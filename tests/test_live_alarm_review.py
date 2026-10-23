import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from live_alarm_review import LiveAlarmReviewManager
from offline_alarm_review import load_prompt_bundle
from video_alarms import VideoAlarmStore
from video_channels import VideoChannelStore
from video_tasks import VideoTaskStore


class LiveAlarmReviewManagerTests(unittest.TestCase):
    def test_review_persists_final_verdict_without_external_report_for_local_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "alarm_outputs"
            event_dir = output / "event_live_review_test"
            event_dir.mkdir(parents=True)
            image = np.full((120, 160, 3), 80, dtype=np.uint8)
            for filename in ("original.jpg", "alarm.jpg", "risk_crop.jpg"):
                self.assertTrue(cv2.imwrite(str(event_dir / filename), image))
            writer = cv2.VideoWriter(
                str(event_dir / "alarm_clip.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full_like(image, 60 + index))
            writer.release()
            metadata = {
                "event_id": event_dir.name,
                "event_type": "机械伤害预警",
                "algorithm_code": "machinery_hazard",
                "algorithm_name": "机械伤害预警",
                "time": "2026-08-24T16:00:00+08:00",
                "short_video_status": "complete",
                "report_url": "",
                "report_status": "local_only",
            }
            (event_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            database_path = root / "platform.db"
            VideoChannelStore(database_path)
            VideoTaskStore(database_path)
            store = VideoAlarmStore(database_path, output)
            store.sync_from_output()
            manager = LiveAlarmReviewManager(store)
            (
                manager._system_prompt,
                manager._algorithm_prompts,
                manager._review_schema,
            ) = load_prompt_bundle(Path("review/prompts"))
            reviewed = {
                "verdict": "confirmed",
                "alarm_probability": 0.9,
                "verdict_confidence": 0.88,
                "evidence_quality": "high",
                "observations": {},
                "summary": "复核通过",
                "reason_codes": ["dangerous_proximity"],
                "needs_manual_review": False,
            }
            with patch.object(manager, "_ensure_model_server"), patch(
                "live_alarm_review.call_review_model",
                return_value=(reviewed, json.dumps(reviewed), {}),
            ):
                manager._review_event(event_dir.name)

            alarm = store.list_alarms(review_status="confirmed")["items"][0]
            self.assertEqual(alarm["report_status"], "local_only")
            self.assertEqual(alarm["review_summary"], "复核通过")
            self.assertTrue((event_dir / "vlm_review" / "review.json").is_file())
            self.assertTrue((event_dir / "vlm_review" / "keyframes_contact.jpg").is_file())
            target_crop = cv2.imread(str(event_dir / "vlm_review" / "review_target_crop.jpg"))
            self.assertIsNotNone(target_crop)
            self.assertGreaterEqual(min(target_crop.shape[:2]), 480)


if __name__ == "__main__":
    unittest.main()
