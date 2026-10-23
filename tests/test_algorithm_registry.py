import unittest
from pathlib import Path

from algorithms import algorithm_registry
from algorithms.machinery_hazard.defaults import ALGORITHM_CODE, load_default_settings
from algorithms.yolov5_objects.defaults import OBJECT_ALGORITHM_DEFINITIONS, load_object_settings


class FakeCore:
    def __init__(self, model_path=None, settings=None):
        self.model_path = model_path
        self.settings = settings

    def process_video(self, source, **kwargs):
        return {"source": source, "kind": "video"}

    def process_image_array(self, image, **kwargs):
        return {"image": image, "kind": "image"}


class AlgorithmRegistryTests(unittest.TestCase):
    def test_machinery_hazard_is_registered_and_created_through_adapter(self):
        settings = load_default_settings()
        algorithm = algorithm_registry.create(
            ALGORITHM_CODE,
            settings=settings,
            core_factory=FakeCore,
        )

        self.assertEqual(algorithm.code, "machinery_hazard")
        self.assertEqual(algorithm.name, "机械伤害预警")
        self.assertEqual(algorithm.process_video("stream")["kind"], "video")
        self.assertIsInstance(algorithm.settings.model_path, Path)
        self.assertEqual(algorithm.config_snapshot()["config_version"], 5)

    def test_default_settings_are_validated_and_serializable(self):
        settings = load_default_settings()

        settings.validate()
        snapshot = settings.to_dict()
        self.assertIsInstance(snapshot["model_path"], str)
        self.assertIsInstance(snapshot["person_class_ids"], list)
        self.assertGreaterEqual(snapshot["risk_consecutive_frames"], 1)
        self.assertIn("machinery_track_iou_threshold", snapshot)
        self.assertIn("risk_zone_class_scales", snapshot)
        self.assertEqual(snapshot["device"], "0")

    def test_imported_yolov5_algorithms_are_registered(self):
        expected = {
            "safety_helmet",
            "ship_detection",
            "hanging_basket",
            "life_jacket",
            "fire_detection",
            "cigarette_detection",
            "smoke_detection",
            "cellphone_usage",
            "rebar_detection",
            "fainting_detection",
            "vehicle_detection",
            "traffic_cone",
        }
        registered = {item["code"] for item in algorithm_registry.list_public()}
        self.assertTrue(expected.issubset(registered))

        settings = load_object_settings("life_jacket")
        self.assertEqual(settings.class_ids, (3, 4))
        self.assertTrue(settings.model_path.is_file())

        helmet_settings = load_object_settings("safety_helmet")
        self.assertEqual(helmet_settings.class_ids, (0, 1, 2))
        self.assertEqual(helmet_settings.alarm_rule, "missing_helmet")

        cigarette_settings = load_object_settings("cigarette_detection")
        self.assertEqual(cigarette_settings.class_ids, (2,))
        self.assertEqual(cigarette_settings.alarm_rule, "person_with_cigarette")
        self.assertEqual(cigarette_settings.person_association_expand_ratio, 0.15)
        self.assertEqual(cigarette_settings.person_association_upper_body_ratio, 0.70)

        cellphone_settings = load_object_settings("cellphone_usage")
        self.assertEqual(cellphone_settings.alarm_rule, "person_with_cellphone")
        self.assertEqual(cellphone_settings.alarm_window_size, 3)
        self.assertEqual(cellphone_settings.alarm_min_hits, 2)

        fainting_settings = load_object_settings("fainting_detection")
        self.assertEqual(fainting_settings.alarm_cooldown_seconds, 60.0)
        self.assertEqual(fainting_settings.alarm_temporal_iou_threshold, 0.20)

        self.assertIn("machinery_hazard_supplement", OBJECT_ALGORITHM_DEFINITIONS)
        self.assertNotIn("machinery_hazard_supplement", registered)


if __name__ == "__main__":
    unittest.main()
