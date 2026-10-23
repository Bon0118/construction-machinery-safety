import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from dataset_annotation.annotation.rules import (
    PrelabelBox,
    eligible_head_size,
    enforce_head_helmet_exclusivity,
)
from dataset_annotation.annotation.head_shoulder_crop import (
    associated_with_person_head_region,
    create_head_shoulder_crop,
)
from dataset_annotation.annotation.teacher_models import (
    TeacherCandidate,
    _candidate_kind,
    build_label_candidates,
    candidate_matches_target_person,
    fallback_decision,
    mask_box,
)
from dataset_annotation.quality.image_quality import evaluate_image


class SafetyHelmetAnnotationRuleTests(unittest.TestCase):
    def test_head_short_side_uses_letterboxed_training_scale(self):
        self.assertFalse(eligible_head_size((0, 0, 44, 49), 2560, 1440, 640, 640, 20))
        self.assertTrue(eligible_head_size((0, 0, 80, 90), 2560, 1440, 640, 640, 20))

    def test_reviewed_helmet_wins_over_overlapping_head(self):
        head = PrelabelBox(1, "head", 0.8, (100, 100, 180, 200))
        helmet = PrelabelBox(2, "helmet", 0.7, (105, 105, 185, 205))

        result, issues = enforce_head_helmet_exclusivity([head, helmet], "helmet")

        self.assertEqual([item.class_name for item in result], ["helmet"])
        self.assertIn("head_helmet_overlap", issues)

    def test_reviewed_head_wins_over_overlapping_helmet(self):
        head = PrelabelBox(1, "head", 0.7, (100, 100, 180, 200))
        helmet = PrelabelBox(2, "helmet", 0.9, (105, 105, 185, 205))

        result, _issues = enforce_head_helmet_exclusivity([head, helmet], "head")

        self.assertEqual([item.class_name for item in result], ["head"])

    def test_low_information_grey_frame_is_rejected(self):
        config = {
            "grey_low_saturation_ratio": 0.92,
            "grey_mean_channel_spread": 1.5,
            "grey_max_brightness_std": 25.0,
            "white_pixel_threshold": 245,
            "maximum_white_ratio": 0.08,
            "black_pixel_threshold": 10,
        }
        image = np.full((120, 160, 3), 128, dtype=np.uint8)
        image[:, ::8] = 130
        with TemporaryDirectory() as directory:
            path = Path(directory) / "grey.jpg"
            cv2.imencode(".jpg", image)[1].tofile(str(path))
            quality, _ = evaluate_image(path, config)

        self.assertFalse(quality.accepted)
        self.assertEqual(quality.rejection_reason, "grey_frame")

    def test_head_shoulder_crop_is_square_and_traceable(self):
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        person = PrelabelBox(0, "person", 0.9, (1000, 500, 1100, 800))
        config = {
            "output_size": 640,
            "minimum_source_side": 128,
            "person_width_multiplier": 2.2,
            "person_height_multiplier": 0.8,
            "vertical_center_ratio": 0.28,
            "association_horizontal_expand": 0.25,
            "association_top_expand": 0.2,
            "association_bottom_ratio": 0.65,
        }

        crop = create_head_shoulder_crop(image, person, config)

        self.assertEqual(crop.image.shape, (640, 640, 3))
        self.assertTrue(crop.person_truncated)
        self.assertEqual(crop.source_person_box_original, person.box)
        remapped = crop.remap_original_box((1020, 510, 1080, 570))
        self.assertIsNotNone(remapped)
        self.assertGreater(remapped[2] - remapped[0], 20)
        self.assertTrue(associated_with_person_head_region((250, 80, 380, 190), crop.person_box, config))
        self.assertFalse(associated_with_person_head_region((0, 500, 30, 600), crop.person_box, config))

        restored = crop.remap_crop_box(remapped)
        self.assertIsNotNone(restored)
        for expected, actual in zip((1020, 510, 1080, 570), restored):
            self.assertLessEqual(abs(expected - actual), 1)

    def test_teacher_helmet_candidate_unites_shell_and_whole_head(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        person = (220, 80, 420, 640)
        grounded = [
            TeacherCandidate("H01", "head", 0.70, (270, 80, 370, 210), "dino"),
            TeacherCandidate("K01", "helmet_shell", 0.85, (265, 70, 375, 140), "dino+sam"),
        ]

        candidates = build_label_candidates(image, person, grounded)

        helmet = next(item for item in candidates if item.kind == "helmet")
        self.assertEqual(helmet.box, (265, 70, 375, 210))
        self.assertIn("helmet_head_union", helmet.source)

    def test_mask_box_returns_tight_bounds(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[3:10, 7:19] = 1
        self.assertEqual(mask_box(mask), (7, 3, 19, 10))

    def test_motorcycle_helmet_is_not_safety_helmet_candidate(self):
        self.assertEqual(_candidate_kind("motorcycle helmet"), "other_headgear")
        self.assertEqual(_candidate_kind("construction safety helmet"), "helmet_shell")

    def test_adjacent_person_headgear_is_not_associated_with_target(self):
        person = (250, 200, 390, 600)
        self.assertTrue(candidate_matches_target_person((270, 175, 350, 280), person))
        self.assertFalse(candidate_matches_target_person((160, 160, 240, 280), person))

    def test_qwen_failure_does_not_create_head_or_helmet_label(self):
        decision = fallback_decision("helmet", [], "invalid semantic response")
        self.assertEqual(decision.head_state, "person_only")
        self.assertEqual(decision.selected_candidate_id, "")
        self.assertEqual(decision.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
