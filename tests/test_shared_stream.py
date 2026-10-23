from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shared_stream import SharedStreamManager
from video_channels import SOURCE_TYPE_CAMERA_INDEX_CODE, SOURCE_TYPE_LOCAL_FILE, VideoChannelStore


class FakeVideoFrame:
    def to_ndarray(self, format="bgr24"):
        assert format == "bgr24"
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakeStream:
    type = "video"
    average_rate = 10


class FakePacket:
    def __init__(self, stream):
        self.stream = stream
        self.size = 1000

    def decode(self):
        time.sleep(0.01)
        return [FakeVideoFrame()]


class FakeContainer:
    def __init__(self):
        self.streams = [FakeStream()]
        self.closed = False

    def demux(self):
        for _ in range(40):
            yield FakePacket(self.streams[0])

    def close(self):
        self.closed = True


class SharedStreamManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = VideoChannelStore(root / "platform.db")
        video_path = root / "shared.mp4"
        video_path.touch()
        self.channel = self.store.create_channel(
            {
                "name": "Shared Camera",
                "source_type": SOURCE_TYPE_LOCAL_FILE,
                "source_url": str(video_path),
                "enabled": True,
            }
        )
        self.manager = SharedStreamManager(
            self.store,
            lambda source, _timeout: (source, source),
            lambda code: f"cameraIndexCode:{code}",
        )

    def tearDown(self):
        self.manager.stop_all()
        self.temp_dir.cleanup()

    def test_two_subscribers_share_one_upstream_and_receive_frames(self):
        opened = []

        def fake_open(*_args, **_kwargs):
            container = FakeContainer()
            opened.append(container)
            return container

        fake_av = types.SimpleNamespace(open=fake_open)
        with patch.dict(sys.modules, {"av": fake_av}):
            first, _ = self.manager.subscribe(self.channel["id"])
            second, _ = self.manager.subscribe(self.channel["id"])
            first_ok, first_frame = first.read()
            second_ok, second_frame = second.read()

            self.assertTrue(first_ok)
            self.assertTrue(second_ok)
            self.assertEqual((48, 64, 3), first_frame.shape)
            self.assertEqual((48, 64, 3), second_frame.shape)
            self.assertEqual(1, len(opened))
            traffic = self.manager.traffic_status(self.channel["id"])
            self.assertEqual(2, traffic["subscriber_count"])
            self.assertGreater(traffic["estimated_bytes_today"], 0)

            first.release()
            second.release()

    def test_hot_restart_keeps_existing_subscriber_alive(self):
        opened = []

        def fake_open(*_args, **_kwargs):
            container = FakeContainer()
            opened.append(container)
            return container

        fake_av = types.SimpleNamespace(open=fake_open)
        with patch.dict(sys.modules, {"av": fake_av}):
            capture, _ = self.manager.subscribe(self.channel["id"])
            first_ok, _ = capture.read()
            self.assertTrue(first_ok)

            self.manager.restart_channel(self.channel["id"])
            deadline = time.monotonic() + 2.0
            while len(opened) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertGreaterEqual(len(opened), 2)
            self.assertTrue(capture.isOpened())
            second_ok, _ = capture.read()
            self.assertTrue(second_ok)
            capture.release()

    def test_on_demand_main_profile_reads_one_frame_and_closes(self):
        camera = self.store.create_channel(
            {
                "name": "Camera confirmation",
                "source_type": SOURCE_TYPE_CAMERA_INDEX_CODE,
                "camera_index_code": "confirm-camera",
                "stream_profile": "sub",
                "enabled": True,
            }
        )
        resolved_sources = []
        self.manager.resolve_capture_source = lambda source, _timeout: (
            resolved_sources.append(source) or "rtsp://test/main",
            "rtsp://test/main",
        )
        opened = []

        def fake_open(*_args, **_kwargs):
            container = FakeContainer()
            opened.append(container)
            return container

        with patch.dict(sys.modules, {"av": types.SimpleNamespace(open=fake_open)}):
            frame = self.manager.capture_profile_frame(camera["id"], "main", 2.0)

        self.assertEqual((48, 64, 3), frame.shape)
        self.assertIn("streamType=0", resolved_sources[0])
        self.assertTrue(opened[0].closed)


if __name__ == "__main__":
    unittest.main()
