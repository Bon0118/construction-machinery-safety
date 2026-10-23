import unittest
from unittest.mock import Mock, patch

from algorithms.composite import create_task_algorithm_runner
from algorithms.machinery_supplement import (
    MachinerySupplementBroker,
    map_supplement_detection,
)
from server import Detection, MachineryHazardDetector


class MachinerySupplementTests(unittest.TestCase):
    def test_model2_classes_map_to_canonical_machinery_classes(self):
        excavator = map_supplement_detection(
            Mock(class_id=7, class_name="Excavator", confidence=0.8, box=(1, 2, 30, 40))
        )
        crane_arm = map_supplement_detection(
            Mock(class_id=9, class_name="CraneArm", confidence=0.7, box=(5, 6, 50, 60))
        )

        self.assertEqual((excavator.class_id, excavator.class_name), (6, "excavator"))
        self.assertEqual((crane_arm.class_id, crane_arm.class_name), (13, "crane_arm"))

    def test_broker_expires_stale_results(self):
        broker = MachinerySupplementBroker()
        item = map_supplement_detection(
            Mock(class_id=7, class_name="Excavator", confidence=0.8, box=(1, 2, 30, 40))
        )
        broker.publish("task:1", [item], observed_at=10.0)

        self.assertEqual(len(broker.latest("task:1", 1.5, now=11.0)), 1)
        self.assertEqual(broker.latest("task:1", 1.5, now=12.0), ())

    def test_overlapping_excavator_is_merged_and_crane_arm_is_added(self):
        primary = [Detection(6, "excavator", 0.70, (100, 100, 300, 300))]
        supplemental = [
            Detection(6, "excavator", 0.85, (110, 105, 310, 305)),
            Detection(13, "crane_arm", 0.75, (250, 40, 500, 140)),
        ]

        merged = MachineryHazardDetector._merge_supplemental_machinery(primary, supplemental)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].box, (100, 100, 310, 305))
        self.assertEqual(merged[0].confidence, 0.85)
        self.assertEqual(merged[1].class_id, 13)

    @patch("algorithms.composite.YoloV5ObjectTaskRunner")
    def test_machinery_task_automatically_enables_hidden_supplement(self, object_runner):
        object_runner.return_value = Mock(code="yolov5_objects")
        machinery_runner = Mock(code="machinery_hazard")

        runner = create_task_algorithm_runner(["machinery_hazard"], machinery_runner)

        object_runner.assert_called_once_with(["machinery_hazard_supplement"])
        self.assertEqual(runner.algorithm_codes, ["machinery_hazard"])


if __name__ == "__main__":
    unittest.main()
