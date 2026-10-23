import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from manual_reviews import ManualReviewStore
from video_alarms import AlarmValidationError, VideoAlarmStore
from video_channels import VideoChannelStore
from video_tasks import VideoTaskStore


class ManualReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "platform.db"
        self.output_dir = self.root / "alarm_outputs"
        self.dataset_dir = self.root / "dataset_outputs"
        channels = VideoChannelStore(self.database_path)
        self.channel = channels.create_channel(
            {
                "name": "复核通道",
                "source_type": "camera_index_code",
                "camera_index_code": "review-camera-code",
                "camera_id": "CAM-REVIEW",
                "enabled": True,
            }
        )
        tasks = VideoTaskStore(self.database_path)
        self.task = tasks.create_task(
            {
                "task_code": "REVIEW-TASK",
                "channel_id": self.channel["id"],
                "report_url": "",
                "algorithm_enabled": True,
            }
        )
        self.event_id = "event_review_001"
        self.event_dir = self.output_dir / self.event_id
        self.event_dir.mkdir(parents=True)
        Image.new("RGB", (320, 240), (80, 90, 100)).save(self.event_dir / "original.jpg")
        Image.new("RGB", (320, 240), (80, 90, 100)).save(self.event_dir / "alarm.jpg")
        metadata = {
            "event_id": self.event_id,
            "event_type": "安全帽检测",
            "camera": "复核通道",
            "source": "cameraIndexCode:review-camera-code",
            "time": "2026-08-27T16:24:43+08:00",
            "task_id": self.task["id"],
            "task_code": self.task["task_code"],
            "channel_id": self.channel["id"],
            "channel_name": self.channel["name"],
            "artifacts": {"annotated_image": "alarm.jpg", "original_image": "original.jpg"},
            "detections": [{"class_name": "person", "box": [80, 40, 180, 220], "confidence": 0.9}],
        }
        (self.event_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        self.alarm_store = VideoAlarmStore(self.database_path, self.output_dir, max_records=100)
        self.alarm = self.alarm_store.list_alarms()["items"][0]
        self.alarm_store.complete_review(
            self.event_id,
            {
                "verdict": "rejected",
                "alarm_probability": 0.1,
                "verdict_confidence": 0.92,
                "evidence_quality": "high",
                "summary": "可见黄色安全帽，原告警不成立。",
                "reason_codes": ["helmet_visible"],
            },
            model="Qwen3-VL-4B",
            prompt_version="safety-helmet-v3",
        )
        annotation_dir = self.dataset_dir / "teacher_run" / "annotations"
        annotation_dir.mkdir(parents=True)
        self.crop_id = f"sample_001_person_01_{self.event_id}"
        (annotation_dir / f"sample_001_{self.event_id}.json").write_text(
            json.dumps(
                {
                    "event_id": self.event_id,
                    "crop_ids": [self.crop_id],
                    "labels": [
                        {"class_name": "person", "box": [80, 40, 180, 220], "confidence": 0.95},
                        {"class_name": "head", "box": [105, 42, 150, 92], "confidence": 0.6},
                    ],
                }
            ),
            encoding="utf-8",
        )
        crop_annotation_dir = self.dataset_dir / "teacher_run" / "crops" / "annotations"
        crop_annotation_dir.mkdir(parents=True)
        self.crop_annotation_path = crop_annotation_dir / f"{self.crop_id}.json"
        self.crop_annotation_path.write_text(
            json.dumps(
                {
                    "crop_id": self.crop_id,
                    "manual_review_reasons": ["teacher_prelabel_requires_manual_review"],
                    "crop_transform": {
                        "source_person_box_original": [80, 40, 180, 220],
                        "crop_box_original": [60, 20, 200, 240],
                        "output_size": [640, 640],
                    },
                    "qwen_semantic_decision": {
                        "head_state": "head",
                        "confidence": 0.9,
                        "reason": "可见未佩戴安全帽的人头。",
                    },
                    "qwen_runtime": {"status": "completed"},
                    "labels": [
                        {"class_name": "person", "box": [90, 60, 550, 620]},
                        {"class_name": "head", "box": [205, 64, 410, 210]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.store = ManualReviewStore(self.alarm_store, self.database_path, self.dataset_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detail_uses_original_image_and_teacher_candidates(self):
        detail = self.store.detail(self.alarm["id"])

        self.assertEqual(detail["image"]["width"], 320)
        self.assertEqual(detail["image"]["height"], 240)
        self.assertEqual(detail["alarm"]["review_status"], "rejected")
        self.assertEqual([item["class_name"] for item in detail["candidate_annotations"]], ["person", "head"])
        person = detail["candidate_annotations"][0]
        self.assertEqual(person["head_status"]["code"], "labeled")
        self.assertEqual(person["head_status"]["label"], "已标注 head")
        self.assertEqual(detail["review"]["workflow_status"], "pending")

    def test_person_head_status_distinguishes_small_filter_and_suspected_missing(self):
        teacher_path = self.dataset_dir / "teacher_run" / "annotations" / f"sample_001_{self.event_id}.json"
        teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
        teacher["labels"] = [teacher["labels"][0]]
        teacher_path.write_text(json.dumps(teacher), encoding="utf-8")

        detail = self.store.detail(self.alarm["id"])
        status = detail["candidate_annotations"][0]["head_status"]
        self.assertEqual(status["code"], "filtered_small")
        self.assertGreater(status["training_short_edge"], 20)
        self.assertIsNotNone(status["original_short_edge"])

        crop = json.loads(self.crop_annotation_path.read_text(encoding="utf-8"))
        crop["labels"] = [crop["labels"][0]]
        crop["manual_review_reasons"] = ["qwen_person_only", "qwen_review_failed"]
        crop["qwen_semantic_decision"] = {
            "head_state": "person_only",
            "confidence": 0,
            "reason": "教师模型候选框选择失败。",
        }
        crop["qwen_runtime"] = {"status": "fallback", "error": "候选框选择失败"}
        self.crop_annotation_path.write_text(json.dumps(crop), encoding="utf-8")

        detail = self.store.detail(self.alarm["id"])
        status = detail["candidate_annotations"][0]["head_status"]
        self.assertEqual(status["code"], "suspected_missing")
        self.assertIn("失败", status["reason"])

    def test_draft_submission_history_sidecar_and_export(self):
        payload = {
            "annotations": [
                {"id": "p1", "class_name": "person", "box": [80, 40, 180, 220], "truncated": False},
                {"id": "h1", "class_name": "helmet", "box": [103, 40, 153, 96], "color": "yellow"},
            ],
            "decision": "corrected",
            "reason_codes": ["wrong_class"],
            "comment": "候选 head 修正为 helmet。",
            "reviewer": "测试审核员",
        }

        draft = self.store.save(self.alarm["id"], payload, submit=False)
        self.assertEqual(draft["review"]["workflow_status"], "draft")
        self.assertEqual(self.store.list_queue(human_status="draft")["total"], 1)
        submitted = self.store.save(self.alarm["id"], payload, submit=True)

        self.assertEqual(submitted["review"]["workflow_status"], "completed")
        self.assertEqual(submitted["review"]["version"], 1)
        self.assertTrue((self.event_dir / "human_review" / "annotation_v001.json").is_file())
        self.assertTrue((self.event_dir / "human_review" / "current.json").is_file())
        history = self.store.list_history(decision="corrected")
        self.assertEqual(history["total"], 1)
        self.assertEqual(history["items"][0]["training_eligibility"], "positive")
        archive, count = self.store.export_archive()
        self.assertEqual(count, 1)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            names = set(bundle.namelist())
            self.assertIn(f"images/{self.event_id}.jpg", names)
            self.assertIn(f"labels/{self.event_id}.txt", names)
            label_text = bundle.read(f"labels/{self.event_id}.txt").decode("utf-8")
            self.assertIn("0 ", label_text)
            self.assertIn("2 ", label_text)

    def test_head_and_helmet_are_mutually_exclusive_for_same_head(self):
        payload = {
            "annotations": [
                {"class_name": "head", "box": [100, 40, 150, 95]},
                {"class_name": "helmet", "box": [98, 38, 152, 98]},
            ],
            "decision": "corrected",
            "reason_codes": ["wrong_class"],
            "comment": "冲突标注",
        }

        with self.assertRaisesRegex(AlarmValidationError, "不能同时标注"):
            self.store.save(self.alarm["id"], payload, submit=True)

    def test_small_head_is_rejected_and_no_target_is_negative(self):
        with self.assertRaisesRegex(AlarmValidationError, "短边小于 20"):
            self.store.save(
                self.alarm["id"],
                {
                    "annotations": [{"class_name": "head", "box": [10, 10, 25, 28]}],
                    "decision": "confirmed",
                    "reason_codes": [],
                    "comment": "",
                },
                submit=True,
            )

        result = self.store.save(
            self.alarm["id"],
            {
                "annotations": [{"class_name": "person", "box": [80, 40, 180, 220]}],
                "decision": "no_target",
                "reason_codes": ["non_person_target"],
                "comment": "画面主体不是人员。",
            },
            submit=True,
        )
        self.assertEqual(result["review"]["annotations"], [])
        self.assertEqual(self.store.list_history()["items"][0]["training_eligibility"], "negative")

    def test_queue_and_detail_exclude_alarm_without_teacher_prelabel(self):
        event_id = "event_review_without_teacher"
        event_dir = self.output_dir / event_id
        event_dir.mkdir(parents=True)
        Image.new("RGB", (320, 240), (70, 80, 90)).save(event_dir / "original.jpg")
        Image.new("RGB", (320, 240), (70, 80, 90)).save(event_dir / "alarm.jpg")
        (event_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "event_id": event_id,
                    "event_type": "安全帽检测",
                    "time": "2026-08-27T16:30:00+08:00",
                    "task_id": self.task["id"],
                    "task_code": self.task["task_code"],
                    "channel_id": self.channel["id"],
                    "channel_name": self.channel["name"],
                    "artifacts": {"annotated_image": "alarm.jpg", "original_image": "original.jpg"},
                    "detections": [{"class_name": "person", "box": [90, 50, 170, 220]}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.alarm_store.sync_from_output()
        alarms = {item["event_id"]: item for item in self.alarm_store.list_alarms() ["items"]}

        self.assertEqual(self.store.list_queue()["total"], 1)
        self.assertNotIn(event_id, [item["event_id"] for item in self.store.list_queue()["items"]])
        with self.assertRaisesRegex(AlarmValidationError, "仅处理具有教师模型候选预标注"):
            self.store.detail(alarms[event_id]["id"])


if __name__ == "__main__":
    unittest.main()
