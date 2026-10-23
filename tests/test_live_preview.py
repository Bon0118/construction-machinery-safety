import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from live_preview import LivePreviewManager
from video_channels import SOURCE_TYPE_LOCAL_FILE, VideoChannelStore


class LivePreviewManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = VideoChannelStore(root / "platform.db")
        video_path = root / "preview.mp4"
        video_path.touch()
        self.channel = self.store.create_channel(
            {
                "name": "Preview Camera",
                "source_type": SOURCE_TYPE_LOCAL_FILE,
                "source_url": str(video_path),
                "camera_id": "preview-test",
                "enabled": True,
            }
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_two_viewers_share_one_capture_and_receive_mjpeg(self) -> None:
        instances = []

        class FakeCapture:
            def __init__(self) -> None:
                self.opened = False
                self.released = False
                instances.append(self)

            def set(self, _property: int, _value: float) -> bool:
                return True

            def open(self, _source: str) -> bool:
                self.opened = True
                return True

            def isOpened(self) -> bool:
                return self.opened

            def read(self):
                time.sleep(0.005)
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self) -> None:
                self.released = True

        manager = LivePreviewManager(
            self.store,
            lambda source, _timeout: (source, source),
            lambda code: "cameraIndexCode:{}".format(code),
        )

        with patch("live_preview.cv2.VideoCapture", FakeCapture):
            manager.touch_client("browser-one", {self.channel["id"]})
            first_stream = manager.iter_mjpeg(self.channel["id"], "browser-one")
            first_frame = next(first_stream)
            manager.touch_client("browser-two", {self.channel["id"]})
            second_stream = manager.iter_mjpeg(self.channel["id"], "browser-two")
            second_frame = next(second_stream)

            self.assertTrue(first_frame.startswith(b"--frame\r\nContent-Type: image/jpeg"))
            self.assertTrue(second_frame.startswith(b"--frame\r\nContent-Type: image/jpeg"))
            self.assertEqual(1, len(instances))
            status = manager.statuses()[self.channel["id"]]
            self.assertEqual("online", status["state"])
            self.assertEqual(2, status["viewers"])
            self.assertEqual(64, status["width"])
            self.assertEqual(48, status["height"])

            first_stream.close()
            second_stream.close()
            manager.stop_all()
            deadline = time.monotonic() + 1.0
            while not instances[0].released and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(instances[0].released)


if __name__ == "__main__":
    unittest.main()
