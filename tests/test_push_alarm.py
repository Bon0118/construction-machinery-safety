import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import push_alarm


class PushAlarmPayloadTests(unittest.TestCase):
    def test_payload_uses_fixed_report_number_and_algorithm_name(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "alarm.jpg"
            image_path.write_bytes(b"alarm-image")
            payload = push_alarm.build_payload(
                image_path=image_path,
                camera_id="CAM-001",
                event_time=datetime(2026, 8, 17, 16, 0, 0),
                event_type="机械伤害预警",
            )

        self.assertEqual(payload["Violator_LSNumber"], "zidingyi")
        self.assertEqual(payload["EventType"], "机械伤害预警")
        self.assertEqual(payload["CameraID"], "CAM-001")
        self.assertTrue(payload["ViolationPhoto"])


if __name__ == "__main__":
    unittest.main()
