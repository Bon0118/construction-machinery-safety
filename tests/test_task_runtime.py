import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from task_runtime import TaskExecutionManager
from video_channels import VideoChannelStore
from video_tasks import VideoTaskStore


class FakeDetector:
    code = "machinery_hazard"
    name = "机械伤害预警"

    def __init__(self):
        self.started = threading.Event()
        self.source = ""
        self.context = {}
        self.alarm_intervals = {}
        self.algorithm_parameters = {}

    def process_video(self, source, **kwargs):
        self.source = source
        self.context = dict(kwargs.get("event_context") or {})
        self.alarm_intervals = dict(kwargs.get("algorithm_alarm_intervals") or {})
        self.algorithm_parameters = dict(kwargs.get("algorithm_parameters") or {})
        self.started.set()
        kwargs["stop_event"].wait(2.0)
        return {"frames": 12, "alarm_count": 1, "stop_reason": "task_stopped"}

    def config_snapshot(self):
        return {"config_version": 1, "confidence_threshold": 0.25}


class FakeAlarmStore:
    def sync_from_output(self):
        return 0

    def report_pending(self, limit=20):
        return []


class FakeStreamManager:
    def __init__(self):
        self.calls = []

    def capture_profile_frame(self, channel_id, profile, timeout):
        self.calls.append((channel_id, profile, timeout))
        return np.ones((200, 200, 3), dtype=np.uint8)


class TaskExecutionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "platform.db"
        self.channels = VideoChannelStore(self.database_path)
        self.channel = self.channels.create_channel(
            {
                "name": "任务运行通道",
                "source_type": "camera_index_code",
                "camera_index_code": "runtime-camera",
                "camera_id": "CAM-RUNTIME",
                "enabled": True,
            }
        )
        self.tasks = VideoTaskStore(self.database_path)
        self.task = self.tasks.create_task(
            {
                "task_code": "RUNTIME-TASK",
                "channel_id": self.channel["id"],
                "report_url": "",
                "algorithm_enabled": True,
            }
        )
        self.detector = FakeDetector()
        self.manager = TaskExecutionManager(
            task_store=self.tasks,
            channel_store=self.channels,
            detector_factory=lambda _codes: self.detector,
            make_camera_source=lambda code: f"cameraIndexCode:{code}",
            alarm_store=FakeAlarmStore(),
        )

    def tearDown(self):
        self.manager.stop_all(preserve_desired_status=False)
        self.temp_dir.cleanup()

    def test_start_runs_selected_channel_and_stop_releases_task(self):
        started = self.manager.start_task(self.task["id"])
        self.assertEqual(started["status"], "running")
        self.assertTrue(self.detector.started.wait(1.0))

        for _ in range(50):
            running = self.manager.get_task(self.task["id"])
            if running["runtime_state"] == "running":
                break
            time.sleep(0.01)
        self.assertEqual(running["runtime_state"], "running")
        self.assertEqual(self.detector.source, "cameraIndexCode:runtime-camera?streamType=0")
        self.assertEqual(self.detector.context["task_id"], self.task["id"])
        self.assertEqual(self.detector.context["camera_id"], "CAM-RUNTIME")
        self.assertEqual(self.detector.context["algorithm_name"], "机械伤害预警")
        self.assertEqual(self.detector.context["algorithm_code"], "machinery_hazard")
        self.assertEqual(self.detector.context["algorithm_config"]["config_version"], 1)
        self.assertEqual(self.detector.alarm_intervals, {"machinery_hazard": 30.0})
        self.assertEqual(
            self.detector.algorithm_parameters["machinery_hazard"]["person_confidence_threshold"],
            0.5,
        )

        stopped = self.manager.stop_task(self.task["id"])
        self.assertEqual(stopped["status"], "stopped")
        self.assertIn(stopped["runtime_state"], {"stopped", "stopping"})

    def test_outside_schedule_waits_disconnected_until_camera_event(self):
        now = datetime.now()
        start = (now + timedelta(hours=6)).strftime("%H:%M")
        end = (now + timedelta(hours=7)).strftime("%H:%M")
        self.tasks.update_task(
            self.task["id"],
            {
                "runtime_policy": {
                    "schedule_enabled": True,
                    "schedule_start": start,
                    "schedule_end": end,
                    "event_wakeup_enabled": True,
                    "event_wakeup_seconds": 30,
                }
            },
        )

        self.manager.start_task(self.task["id"])
        time.sleep(0.2)
        waiting = self.manager.get_task(self.task["id"])
        self.assertEqual(waiting["runtime_state"], "waiting_event")
        self.assertFalse(self.detector.started.is_set())

        result = self.manager.wake_from_camera_event(
            {"camera_id": "CAM-RUNTIME", "event_type": "motion"}
        )
        self.assertEqual(result["awakened_count"], 1)
        self.assertTrue(self.detector.started.wait(1.0))
        running = self.manager.get_task(self.task["id"])
        self.assertEqual(running["runtime_state"], "event_running")
        self.assertEqual(running["runtime_wake_reason"], "motion")

    def test_small_target_uses_one_main_stream_confirmation_frame(self):
        stream_manager = FakeStreamManager()
        self.manager.stream_manager = stream_manager
        channel = self.channels.update_channel(
            self.channel["id"], {"stream_profile": "sub"}
        )
        session = self.manager._session(self.task["id"])
        if session is None:
            from task_runtime import TaskRuntimeSession
            session = TaskRuntimeSession(task_id=self.task["id"])
        callback = self.manager._confirmation_callback(
            session,
            self.channels.get_channel(channel["id"], public=False),
            {
                "main_stream_confirmation_enabled": True,
                "small_target_area_ratio": 0.02,
                "confirmation_timeout_seconds": 8,
            },
        )

        result = callback(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            candidate_boxes=[(0, 0, 5, 5)],
            verify=lambda main_frame: {"confirmed": bool(main_frame.any())},
        )

        self.assertTrue(result["confirmed"])
        self.assertEqual(stream_manager.calls, [(channel["id"], "main", 8.0)])
        self.assertEqual(session.confirmation_requests, 1)
        self.assertEqual(session.confirmation_passed, 1)


if __name__ == "__main__":
    unittest.main()
