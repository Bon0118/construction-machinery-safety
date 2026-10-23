import json
import unittest
from pathlib import Path

from offline_alarm_review import load_prompt_bundle, validate_review


def helmet_metadata(short_edge=44):
    return {
        "algorithm_code": "safety_helmet",
        "detections": [
            {
                "class_id": 1,
                "box": [100, 100, 100 + short_edge, 100 + short_edge],
            }
        ],
    }


def helmet_review(headgear_type, *, verdict="uncertain", probability=0.5, quality="medium", shell="no", outline="no", summary=""):
    return {
        "verdict": verdict,
        "alarm_probability": probability,
        "verdict_confidence": 0.9,
        "evidence_quality": quality,
        "observations": {
            "person_visible": "yes",
            "head_visible": "yes",
            "headgear_type": headgear_type,
            "hard_shell_visible": shell,
            "helmet_outline_visible": outline,
            "multi_frame_consistent": "yes",
            "image_quality": quality,
        },
        "reason_codes": [],
        "summary": summary,
        "needs_manual_review": False,
    }


class AlarmReviewPolicyTests(unittest.TestCase):
    def test_v3_prompt_and_schema_override_are_loaded(self):
        _system, prompts, schemas = load_prompt_bundle(Path("review/prompts"))
        self.assertIn("普通帽不等于安全帽", prompts["safety_helmet"])
        observations = schemas["safety_helmet"]["properties"]["observations"]
        self.assertIn("headgear_type", observations["required"])

    def test_real_162443_response_can_no_longer_be_rejected(self):
        raw = {
            "verdict": "rejected",
            "alarm_probability": 0.0,
            "verdict_confidence": 0.95,
            "evidence_quality": "high",
            "observations": {
                "person_visible": "yes",
                "target_visible": "yes",
                "person_role": "unclear",
                "spatial_relation": "head covering that appears to be a cap or helmet",
                "motion_evidence": "person is walking",
                "physical_barrier": "no",
                "keyframe_consistency": "consistent",
                "image_quality": "high",
            },
            "reason_codes": ["helmet_visible", "person_visible"],
            "summary": "Person has a head covering, likely a cap or helmet.",
            "needs_manual_review": False,
        }

        result = validate_review(raw, "safety_helmet", helmet_metadata(44))

        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(result["evidence_quality"], "medium")
        self.assertEqual(result["target_short_edge_pixels"], 44)
        self.assertTrue(result["needs_manual_review"])
        self.assertTrue(any("helmet_ambiguous_description" in value for value in result["policy_adjustments"]))

    def test_bare_head_and_ordinary_hat_confirm_alarm(self):
        for headgear_type in ("bare_head", "ordinary_hat"):
            with self.subTest(headgear_type=headgear_type):
                result = validate_review(
                    helmet_review(headgear_type),
                    "safety_helmet",
                    helmet_metadata(80),
                )
                self.assertEqual(result["verdict"], "confirmed")
                self.assertGreaterEqual(result["alarm_probability"], 0.75)

    def test_safety_helmet_requires_visible_structure(self):
        unclear = validate_review(
            helmet_review("safety_helmet", verdict="rejected", probability=0.05),
            "safety_helmet",
            helmet_metadata(80),
        )
        self.assertEqual(unclear["verdict"], "uncertain")

        clear = validate_review(
            helmet_review(
                "safety_helmet",
                verdict="rejected",
                probability=0.05,
                shell="yes",
                outline="yes",
            ),
            "safety_helmet",
            helmet_metadata(80),
        )
        self.assertEqual(clear["verdict"], "rejected")

    def test_small_head_without_specific_multiframe_evidence_is_uncertain(self):
        result = validate_review(
            helmet_review("bare_head", verdict="confirmed", probability=0.95, quality="high"),
            "safety_helmet",
            helmet_metadata(28),
        )
        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(result["evidence_quality"], "medium")


if __name__ == "__main__":
    unittest.main()
