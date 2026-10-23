import ast
import unittest
from pathlib import Path

import config
from algorithms.machinery_hazard import defaults as machinery_defaults


class ConfigBoundaryTests(unittest.TestCase):
    def test_root_config_does_not_define_algorithm_defaults(self):
        config_path = Path(config.__file__)
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        assigned = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertNotIn("MODEL_PATH", assigned)
        self.assertNotIn("CONFIDENCE_THRESHOLD", assigned)
        self.assertNotIn("MACHINERY_MOTION_ENABLED", assigned)
        self.assertNotIn("RISK_ZONE_CLASS_SCALES", assigned)
        self.assertNotIn("ALARM_RISK_TEXT", assigned)

    def test_legacy_reads_are_delegated_to_machinery_defaults(self):
        self.assertEqual(config.MODEL_PATH, machinery_defaults.MODEL_PATH)
        self.assertEqual(config.CONFIDENCE_THRESHOLD, machinery_defaults.CONFIDENCE_THRESHOLD)
        self.assertEqual(config.RISK_CONSECUTIVE_FRAMES, machinery_defaults.RISK_CONSECUTIVE_FRAMES)


if __name__ == "__main__":
    unittest.main()
