import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_alarms import AlarmValidationError, VideoAlarmStore
from video_channels import VideoChannelStore
from video_tasks import VideoTaskStore


class VideoAlarmStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "platform.db"
        self.output_dir = self.root / "alarm_outputs"
        self.channels = VideoChannelStore(self.database_path)
        self.channel = self.channels.create_channel(
            {
                "name": "告警通道",
                "source_type": "camera_index_code",
                "camera_index_code": "alarm-camera-code",
                "camera_id": "CAM-ALARM",
                "enabled": True,
            }
        )
        self.tasks = VideoTaskStore(self.database_path)
        self.task = self.tasks.create_task(
            {
                "task_code": "ALARM-TASK",
                "channel_id": self.channel["id"],
                "report_url": "",
                "algorithm_enabled": True,
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_event(self, index, alarm_time):
        event_id = f"event_alarm_{index}"
        event_dir = self.output_dir / event_id
        event_dir.mkdir(parents=True)
        (event_dir / "alarm.jpg").write_bytes(b"fake-jpeg")
        metadata = {
            "event_id": event_id,
            "event_type": "机械伤害风险",
            "camera": "告警通道",
            "source": "cameraIndexCode:alarm-camera-code",
            "time": alarm_time,
            "artifacts": {"annotated_image": "alarm.jpg"},
        }
        (event_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return event_dir

    def test_sync_filters_and_local_only_status(self):
        event_dir = self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        result = store.list_alarms(
            task_id=self.task["id"],
            channel_id=self.channel["id"],
            alarm_date="2026-08-17",
        )
        self.assertEqual(result["total"], 1)
        alarm = result["items"][0]
        self.assertEqual(alarm["task_code"], "ALARM-TASK")
        self.assertEqual(alarm["channel_name"], "告警通道")
        self.assertEqual(alarm["alarm_type"], "机械伤害预警")
        self.assertEqual(alarm["report_status"], "local_only")
        self.assertTrue(store.image_path(alarm["id"]).is_file())
        self.assertEqual(store.delete_alarms([alarm["id"]]), 1)
        self.assertFalse(event_dir.exists())

    def test_limit_deletes_oldest_event(self):
        oldest = self.create_event(1, "2026-08-15T10:00:00+08:00")
        self.create_event(2, "2026-08-16T10:00:00+08:00")
        self.create_event(3, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=2)
        result = store.list_alarms()
        self.assertEqual(result["total"], 2)
        self.assertFalse(oldest.exists())
        self.assertEqual([item["event_id"] for item in result["items"]], ["event_alarm_3", "event_alarm_2"])

    def test_event_without_alarm_jpg_is_not_displayed(self):
        event_dir = self.output_dir / "event_original_only"
        event_dir.mkdir(parents=True)
        (event_dir / "original.jpg").write_bytes(b"fake-original")
        (event_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "event_id": event_dir.name,
                    "time": "2026-08-17T10:00:00+08:00",
                    "artifacts": {"original_image": "original.jpg"},
                }
            ),
            encoding="utf-8",
        )

        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)

        self.assertEqual(store.list_alarms()["total"], 0)

    def test_existing_original_image_index_is_corrected_to_alarm_jpg(self):
        event_dir = self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        alarm = store.list_alarms()["items"][0]
        original_path = event_dir / "original.jpg"
        original_path.write_bytes(b"fake-original")
        with store._connect() as connection:
            connection.execute(
                "UPDATE video_alarms SET image_path = ? WHERE id = ?",
                (str(original_path.resolve()), alarm["id"]),
            )

        store.sync_from_output()

        self.assertEqual(store.image_path(alarm["id"]), (event_dir / "alarm.jpg").resolve())

    def test_configured_alarm_uses_channel_camera_id_and_report_url(self):
        self.tasks.update_task(
            self.task["id"],
            {
                "task_code": self.task["task_code"],
                "channel_id": self.channel["id"],
                "report_url": "http://example.test/alarm",
                "algorithm_enabled": True,
            },
        )
        self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        store.sync_from_output()

        # Reporting is released only after the final, policy-adjusted review
        # verdict is confirmed.
        alarm = store.list_alarms()["items"][0]
        self.assertEqual(store.report_pending(), [])
        store.complete_review(
            alarm["event_id"],
            {
                "verdict": "confirmed",
                "alarm_probability": 0.91,
                "verdict_confidence": 0.88,
                "evidence_quality": "high",
                "summary": "告警证据清晰",
                "reason_codes": ["dangerous_proximity"],
            },
            model="test-vlm",
            prompt_version="test-v2",
        )

        with patch("video_alarms.push_alarm.push_alarm") as send:
            send.return_value = {"success": True, "status": 200, "response": "ok"}
            results = store.report_pending()

        self.assertEqual(results[0]["status"], "success")
        api_url, payload, timeout = send.call_args.args
        self.assertEqual(api_url, "http://example.test/alarm")
        self.assertEqual(payload["CameraID"], "CAM-ALARM")
        self.assertEqual(payload["Violator_LSNumber"], "zidingyi")
        self.assertEqual(payload["EventType"], "机械伤害预警")
        self.assertGreater(timeout, 0)
        alarm = store.list_alarms()["items"][0]
        self.assertEqual(alarm["report_status"], "success")
        self.assertEqual(alarm["review_status"], "confirmed")

    def test_rejected_review_is_filterable_and_never_reported(self):
        self.tasks.update_task(
            self.task["id"],
            {"report_url": "http://example.test/alarm"},
        )
        self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        store.sync_from_output()
        alarm = store.list_alarms()["items"][0]
        store.complete_review(
            alarm["event_id"],
            {
                "verdict": "rejected",
                "alarm_probability": 0.08,
                "verdict_confidence": 0.92,
                "evidence_quality": "high",
                "summary": "明确误检",
                "reason_codes": ["false_person_detection"],
            },
            model="test-vlm",
            prompt_version="test-v2",
        )

        with patch("video_alarms.push_alarm.push_alarm") as send:
            self.assertEqual(store.report_pending(), [])
        send.assert_not_called()
        rejected = store.list_alarms(review_status="rejected")
        self.assertEqual(rejected["total"], 1)
        self.assertEqual(rejected["items"][0]["report_status"], "review_blocked")
        self.assertEqual(store.list_alarms(review_status="confirmed")["total"], 0)

    def test_uncertain_alarm_can_be_manually_reported_after_operator_override(self):
        self.tasks.update_task(
            self.task["id"],
            {"report_url": "http://example.test/alarm"},
        )
        event_dir = self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        store.sync_from_output()
        alarm = store.list_alarms()["items"][0]
        store.complete_review(
            alarm["event_id"],
            {
                "verdict": "uncertain",
                "alarm_probability": 0.5,
                "verdict_confidence": 0.4,
                "evidence_quality": "medium",
                "summary": "需要人工确认",
                "reason_codes": ["insufficient_evidence"],
            },
            model="test-vlm",
            prompt_version="test-v3",
        )
        uncertain = store.list_alarms(review_status="uncertain")["items"][0]
        self.assertTrue(uncertain["can_manual_report"])
        self.assertEqual(store.report_pending(), [])

        with patch("video_alarms.push_alarm.push_alarm") as send:
            send.return_value = {"success": True, "status": 200, "response": "ok"}
            result = store.manual_report_alarm(uncertain["id"])

        self.assertTrue(result["sent"])
        self.assertTrue(result["manual_override"])
        send.assert_called_once()
        updated = store.list_alarms(review_status="uncertain")["items"][0]
        self.assertEqual(updated["report_status"], "success")
        self.assertFalse(updated["can_manual_report"])
        metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["manual_report"])
        self.assertTrue(metadata["manual_report_attempted_at"])

    def test_manual_report_rejects_non_uncertain_or_missing_url(self):
        self.create_event(1, "2026-08-17T10:00:00+08:00")
        store = VideoAlarmStore(self.database_path, self.output_dir, max_records=1000)
        store.sync_from_output()
        alarm = store.list_alarms()["items"][0]
        store.complete_review(
            alarm["event_id"],
            {
                "verdict": "uncertain",
                "alarm_probability": 0.5,
                "verdict_confidence": 0.4,
                "evidence_quality": "low",
                "summary": "需要人工确认",
                "reason_codes": ["insufficient_evidence"],
            },
            model="test-vlm",
            prompt_version="test-v3",
        )
        with self.assertRaisesRegex(AlarmValidationError, "未配置上报地址"):
            store.manual_report_alarm(alarm["id"])

        with store._connect() as connection:
            connection.execute(
                "UPDATE video_alarms SET review_status = 'rejected' WHERE id = ?",
                (alarm["id"],),
            )
        with self.assertRaisesRegex(AlarmValidationError, "只有复核结果为 uncertain"):
            store.manual_report_alarm(alarm["id"])


if __name__ == "__main__":
    unittest.main()
