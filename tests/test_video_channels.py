from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_channels import (
    MASKED_PASSWORD,
    ChannelValidationError,
    SOURCE_TYPE_CAMERA_INDEX_CODE,
    SOURCE_TYPE_LOCAL_FILE,
    SOURCE_TYPE_RTSP,
    VideoChannelStore,
    channel_source,
    mask_stream_url,
    probe_channel,
    probe_channel_bounded,
)


class VideoChannelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "platform.db"
        self.store = VideoChannelStore(self.database_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_imports_configured_camera_index_codes_once(self) -> None:
        first = self.store.list_channels()
        self.assertEqual(3, len(first))
        self.store.import_configured_camera_index_codes()
        second = self.store.list_channels()
        self.assertEqual(3, len(second))
        self.assertTrue(all(item["source_type"] == SOURCE_TYPE_CAMERA_INDEX_CODE for item in second))
        self.assertTrue(all(item["reconnect_enabled"] for item in second))
        self.assertTrue(all(item["stream_profile"] == "main" for item in second))

    def test_camera_index_code_stream_profile_is_added_to_source(self) -> None:
        created = self.store.create_channel(
            {
                "name": "Sub Stream Camera",
                "source_type": SOURCE_TYPE_CAMERA_INDEX_CODE,
                "camera_index_code": "sub-camera",
                "stream_profile": "sub",
                "enabled": True,
            }
        )

        source = channel_source(created, lambda code: f"cameraIndexCode:{code}")

        self.assertEqual("sub", created["stream_profile"])
        self.assertEqual("cameraIndexCode:sub-camera?streamType=1", source)

    def test_crud_masks_rtsp_password_and_preserves_masked_update(self) -> None:
        raw_url = "rtsp://operator:secret@example.com:554/live"
        created = self.store.create_channel(
            {
                "name": "Direct RTSP",
                "source_type": SOURCE_TYPE_RTSP,
                "source_url": raw_url,
                "camera_id": "CAM-001",
                "enabled": True,
            }
        )
        self.assertIn(MASKED_PASSWORD, created["source_url"])
        self.assertNotIn("secret", created["source_url"])
        private = self.store.get_channel(created["id"], public=False)
        self.assertEqual(raw_url, private["source_url"])

        updated = self.store.update_channel(
            created["id"],
            {
                "name": "Direct RTSP Updated",
                "source_type": SOURCE_TYPE_RTSP,
                "source_url": created["source_url"],
                "camera_id": "CAM-002",
                "enabled": False,
            },
        )
        self.assertEqual("Direct RTSP Updated", updated["name"])
        self.assertFalse(updated["enabled"])
        self.assertEqual(raw_url, self.store.get_channel(created["id"], public=False)["source_url"])

        self.store.delete_channel(created["id"])
        self.assertEqual(3, len(self.store.list_channels()))

    def test_duplicate_name_is_rejected(self) -> None:
        payload = {
            "name": "Same Name",
            "source_type": SOURCE_TYPE_RTSP,
            "source_url": "rtsp://example.com/one",
            "enabled": True,
        }
        self.store.create_channel(payload)
        payload["source_url"] = "rtsp://example.com/two"
        with self.assertRaises(ChannelValidationError):
            self.store.create_channel(payload)

    def test_local_video_file_is_supported(self) -> None:
        video_path = Path(self.temp_dir.name) / "sample.mp4"
        video_path.touch()
        created = self.store.create_channel(
            {
                "name": "Local Video",
                "source_type": SOURCE_TYPE_LOCAL_FILE,
                "source_url": str(video_path),
                "enabled": True,
            }
        )
        self.assertEqual(str(video_path.resolve()), created["source_url"])

    def test_mask_stream_url(self) -> None:
        masked = mask_stream_url("rtsp://user:p%40ss@127.0.0.1:554/live?x=1")
        self.assertEqual("rtsp://user:******@127.0.0.1:554/live?x=1", masked)


class ProbeChannelTests(unittest.TestCase):
    def test_bounded_probe_returns_timeout_when_capture_backend_blocks(self) -> None:
        channel = {
            "source_type": SOURCE_TYPE_RTSP,
            "camera_index_code": "",
            "source_url": "rtsp://example.com/live",
        }

        with patch("video_channels.probe_channel", side_effect=lambda *args, **kwargs: time.sleep(0.2)):
            started_at = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "0.05 秒"):
                probe_channel_bounded(
                    channel,
                    lambda source, timeout: (source, source),
                    lambda code: code,
                    timeout_seconds=0.05,
                )
            self.assertLess(time.monotonic() - started_at, 0.18)

    def test_camera_index_code_retries_and_returns_snapshot(self) -> None:
        instances = []

        class FakeCapture:
            def __init__(self) -> None:
                self.number = len(instances) + 1
                self.opened = False
                self.released = False
                instances.append(self)

            def set(self, _property: int, _value: float) -> bool:
                return True

            def open(self, _source: str) -> bool:
                self.opened = self.number == 3
                return self.opened

            def isOpened(self) -> bool:
                return self.opened

            def read(self):
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def get(self, _property: int) -> float:
                return 25.0

            def release(self) -> None:
                self.released = True

        channel = {
            "source_type": SOURCE_TYPE_CAMERA_INDEX_CODE,
            "camera_index_code": "camera-code",
            "source_url": "",
        }
        resolver_calls = []

        def resolver(source: str, _timeout_seconds: float):
            resolver_calls.append(source)
            return "rtsp://user:secret@example.com/live", "masked"

        with patch("video_channels.cv2.VideoCapture", FakeCapture):
            result, snapshot = probe_channel(
                channel,
                resolver,
                lambda code: f"cameraIndexCode:{code}",
                timeout_seconds=3.0,
                need_snapshot=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(3, result["attempts"])
        self.assertTrue(result["reconnect_enabled"])
        self.assertEqual(64, result["width"])
        self.assertEqual(48, result["height"])
        self.assertIsNotNone(snapshot)
        self.assertEqual(3, len(resolver_calls))
        self.assertTrue(all(instance.released for instance in instances))


if __name__ == "__main__":
    unittest.main()
