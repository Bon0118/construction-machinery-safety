import io
import json
import unittest
from unittest.mock import patch

from live_alarm_review import LiveAlarmReviewManager, MODEL_NAME


class AlarmModelIdentityTests(unittest.TestCase):
    def test_ready_server_must_serve_configured_model(self):
        for served_name, expected in [(MODEL_NAME, True), ("Qwen3VL-4B", False)]:
            with self.subTest(served_name=served_name):
                health = io.BytesIO(b'{"status":"ok"}')
                health.status = 200
                models = io.BytesIO(json.dumps({"data": [{"id": served_name}]}).encode())
                with patch("live_alarm_review.urllib.request.urlopen", side_effect=[health, models]):
                    manager = LiveAlarmReviewManager(object())
                    self.assertEqual(manager._server_ready(), expected)


if __name__ == "__main__":
    unittest.main()
