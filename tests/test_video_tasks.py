import tempfile
import unittest
from pathlib import Path

from video_channels import VideoChannelStore
from video_tasks import TaskValidationError, VideoTaskStore


class VideoTaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "platform.db"
        self.channels = VideoChannelStore(self.database_path)
        self.channel = self.channels.create_channel(
            {
                "name": "任务测试通道",
                "source_type": "rtsp",
                "source_url": "rtsp://127.0.0.1/live",
                "camera_id": "CAM-001",
                "enabled": True,
            }
        )
        self.tasks = VideoTaskStore(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def payload(self, task_code="TASK-001"):
        return {
            "task_code": task_code,
            "channel_id": self.channel["id"],
            "report_url": "http://example.test/alarm",
            "algorithm_enabled": True,
            "report_mapping": {"camera_id": "CameraID"},
        }

    def test_create_update_status_and_delete(self):
        task = self.tasks.create_task(self.payload())
        self.assertEqual(task["task_code"], "TASK-001")
        self.assertEqual(task["channel_name"], "任务测试通道")
        self.assertEqual(task["status"], "stopped")
        self.assertEqual(task["report_mapping"]["camera_id"], "CameraID")

        running = self.tasks.set_status(task["id"], "running")
        self.assertEqual(running["status"], "running")

        updated = self.tasks.update_task(task["id"], {**self.payload("TASK-002"), "algorithm_enabled": False})
        self.assertEqual(updated["task_code"], "TASK-002")
        self.assertFalse(updated["algorithm_enabled"])

        with self.assertRaises(TaskValidationError):
            self.tasks.set_status(task["id"], "running")

        self.tasks.delete_task(task["id"])
        self.assertEqual(self.tasks.list_tasks(), [])

    def test_duplicate_task_code_is_rejected(self):
        self.tasks.create_task(self.payload())
        with self.assertRaises(TaskValidationError):
            self.tasks.create_task(self.payload())

    def test_deleted_channel_is_retained_as_missing_reference(self):
        task = self.tasks.create_task(self.payload())
        self.channels.delete_channel(self.channel["id"])
        missing = self.tasks.get_task(task["id"])
        self.assertIsNone(missing["channel_id"])
        self.assertEqual(missing["channel_name"], "通道已删除")
        with self.assertRaises(TaskValidationError):
            self.tasks.set_status(task["id"], "running")

    def test_invalid_report_url_is_rejected(self):
        payload = self.payload()
        payload["report_url"] = "not-a-url"
        with self.assertRaises(TaskValidationError):
            self.tasks.create_task(payload)

    def test_empty_report_url_is_local_only_configuration(self):
        payload = self.payload()
        payload["report_url"] = ""
        task = self.tasks.create_task(payload)
        self.assertEqual(task["report_url"], "")

    def test_multiple_algorithm_codes_are_saved_and_restored(self):
        payload = self.payload("TASK-MULTI")
        payload.pop("algorithm_enabled", None)
        payload["algorithm_codes"] = ["safety_helmet", "fire_detection", "traffic_cone"]
        task = self.tasks.create_task(payload)

        self.assertEqual(
            task["algorithm_codes"],
            ["safety_helmet", "fire_detection", "traffic_cone"],
        )
        self.assertEqual(
            [item["name"] for item in task["algorithms"]],
            ["安全帽检测", "明火检测", "交通锥检测"],
        )

        updated = self.tasks.update_task(
            task["id"],
            {**payload, "algorithm_codes": ["ship_detection", "life_jacket"]},
        )
        self.assertEqual(updated["algorithm_codes"], ["ship_detection", "life_jacket"])

    def test_unknown_algorithm_code_is_rejected(self):
        payload = self.payload("TASK-UNKNOWN")
        payload["algorithm_codes"] = ["does_not_exist"]
        with self.assertRaises(TaskValidationError):
            self.tasks.create_task(payload)

    def test_each_algorithm_alarm_interval_is_saved_and_validated(self):
        payload = self.payload("TASK-INTERVALS")
        payload["algorithm_codes"] = ["machinery_hazard", "fainting_detection"]
        payload["algorithm_alarm_intervals"] = {
            "machinery_hazard": 15,
            "fainting_detection": 120,
        }
        task = self.tasks.create_task(payload)

        self.assertEqual(
            task["algorithm_alarm_intervals"],
            {"machinery_hazard": 15.0, "fainting_detection": 120.0},
        )

        updated = self.tasks.update_task(
            task["id"],
            {"algorithm_alarm_intervals": {"machinery_hazard": 5, "fainting_detection": 45}},
        )
        self.assertEqual(updated["algorithm_alarm_intervals"]["machinery_hazard"], 5.0)
        self.assertEqual(updated["algorithm_alarm_intervals"]["fainting_detection"], 45.0)

        with self.assertRaises(TaskValidationError):
            self.tasks.update_task(
                task["id"],
                {"algorithm_alarm_intervals": {"machinery_hazard": -1}},
            )

    def test_alarm_interval_defaults_match_each_algorithm(self):
        payload = self.payload("TASK-INTERVAL-DEFAULTS")
        payload["algorithm_codes"] = ["machinery_hazard", "fainting_detection"]
        task = self.tasks.create_task(payload)

        self.assertEqual(task["algorithm_alarm_intervals"]["machinery_hazard"], 30.0)
        self.assertEqual(task["algorithm_alarm_intervals"]["fainting_detection"], 60.0)

    def test_algorithm_parameters_are_saved_with_current_defaults(self):
        payload = self.payload("TASK-PARAMETERS")
        payload["algorithm_codes"] = [
            "machinery_hazard",
            "safety_helmet",
            "cigarette_detection",
        ]
        payload["algorithm_parameters"] = {
            "machinery_hazard": {
                "person_confidence_threshold": 0.65,
                "confidence_threshold": 0.4,
                "overlap_ratio_threshold": 0.05,
                "risk_consecutive_frames": 3,
            },
            "safety_helmet": {
                "person_confidence_threshold": 0.5,
                "bare_head_confidence_threshold": 0.6,
                "helmet_confidence_threshold": 0.2,
                "violation_consecutive_hits": 2,
            },
            "cigarette_detection": {
                "confidence_threshold": 0.55,
                "person_confidence_threshold": 0.4,
                "alarm_window_size": 4,
                "alarm_min_hits": 3,
            },
        }

        task = self.tasks.create_task(payload)

        self.assertEqual(
            task["algorithm_parameters"]["machinery_hazard"]["risk_consecutive_frames"],
            3,
        )
        self.assertEqual(
            task["algorithm_parameters"]["safety_helmet"]["helmet_confidence_threshold"],
            0.2,
        )
        self.assertEqual(
            task["algorithm_parameters"]["cigarette_detection"]["alarm_min_hits"],
            3,
        )

        defaults = self.tasks.create_task(
            {
                **self.payload("TASK-PARAMETER-DEFAULTS"),
                "algorithm_codes": ["cellphone_usage", "fainting_detection"],
            }
        )["algorithm_parameters"]
        self.assertEqual(defaults["cellphone_usage"]["confidence_threshold"], 0.3)
        self.assertEqual(defaults["cellphone_usage"]["alarm_min_hits"], 2)
        self.assertEqual(defaults["fainting_detection"]["confidence_threshold"], 0.35)
        self.assertEqual(defaults["fainting_detection"]["alarm_min_duration_seconds"], 0.0)

    def test_algorithm_parameter_dependencies_are_validated(self):
        payload = self.payload("TASK-BAD-PARAMETERS")
        payload["algorithm_codes"] = ["cigarette_detection"]
        payload["algorithm_parameters"] = {
            "cigarette_detection": {
                "alarm_window_size": 2,
                "alarm_min_hits": 3,
            }
        }
        with self.assertRaises(TaskValidationError):
            self.tasks.create_task(payload)

    def test_runtime_policy_is_saved_and_validated(self):
        payload = self.payload("TASK-POLICY")
        payload["runtime_policy"] = {
            "schedule_enabled": True,
            "schedule_start": "22:00",
            "schedule_end": "06:00",
            "event_wakeup_enabled": True,
            "event_wakeup_seconds": 600,
            "main_stream_confirmation_enabled": True,
            "small_target_area_ratio": 0.01,
            "confirmation_timeout_seconds": 12,
        }
        task = self.tasks.create_task(payload)

        self.assertTrue(task["runtime_policy"]["schedule_enabled"])
        self.assertEqual(task["runtime_policy"]["schedule_start"], "22:00")
        self.assertEqual(task["runtime_policy"]["event_wakeup_seconds"], 600)
        self.assertTrue(task["runtime_policy"]["main_stream_confirmation_enabled"])

        payload["runtime_policy"]["schedule_start"] = "25:00"
        with self.assertRaises(TaskValidationError):
            self.tasks.create_task({**payload, "task_code": "TASK-BAD-POLICY"})

    def test_channel_delete_requires_deleting_tasks_first(self):
        task = self.tasks.create_task(self.payload())
        reason = self.tasks.channel_delete_block_reason(self.channel["id"])
        self.assertIn("视频任务", reason)
        self.tasks.set_status(task["id"], "running")
        running_reason = self.tasks.channel_delete_block_reason(self.channel["id"])
        self.assertIn("已启动任务", running_reason)
        self.tasks.delete_task(task["id"])
        self.assertIsNone(self.tasks.channel_delete_block_reason(self.channel["id"]))


if __name__ == "__main__":
    unittest.main()
