import unittest
from pathlib import Path

from offline_alarm_review import load_prompt_bundle, validate_review


def review(*, position="in_front", attention="yes", active="yes", verdict="confirmed", probability=0.95):
    return {
        "verdict": verdict,
        "alarm_probability": probability,
        "verdict_confidence": 0.9,
        "evidence_quality": "high",
        "observations": {
            "person_visible": "yes",
            "target_visible": "yes",
            "same_person": "yes",
            "phone_position": position,
            "attention_to_phone": attention,
            "active_use": active,
        },
        "reason_codes": [],
        "summary": "candidate person and phone",
        "needs_manual_review": False,
    }


class CellphoneActionReviewTests(unittest.TestCase):
    def test_dedicated_prompt_and_schema_are_loaded(self):
        _, prompts, schemas = load_prompt_bundle(Path("review/prompts"))
        self.assertIn("不能把人和手机同时出现", prompts["cellphone_usage"])
        required = schemas["cellphone_usage"]["properties"]["observations"]["required"]
        self.assertIn("attention_to_phone", required)
        self.assertIn("active_use", required)

    def test_held_behind_back_is_filtered_even_if_model_confirms(self):
        result = validate_review(review(position="behind_back", attention="no", active="no"), "cellphone_usage")
        self.assertEqual(result["verdict"], "rejected")
        self.assertLessEqual(result["alarm_probability"], 0.25)

    def test_archived_case_rejects_behind_back_despite_uncertain_phone_identity(self):
        raw = review(position="behind_back", attention="unclear", active="no",
                     verdict="rejected", probability=0.0)
        raw["evidence_quality"] = "low"
        raw["observations"].update(target_visible="unclear", same_person="no",
                                   keyframe_consistency="consistent")
        result = validate_review(raw, "cellphone_usage")
        self.assertEqual(result["verdict"], "rejected")
        self.assertIn("cellphone_behind_back_no_action:uncertain->rejected", result["policy_adjustments"])

    def test_carrying_phone_without_clear_viewing_is_not_confirmed(self):
        for attention, active in (("unclear", "unclear"), ("yes", "unclear")):
            with self.subTest(attention=attention, active=active):
                result = validate_review(review(attention=attention, active=active), "cellphone_usage")
                self.assertEqual(result["verdict"], "uncertain")
                self.assertTrue(result["needs_manual_review"])

    def test_clear_viewing_can_be_confirmed(self):
        result = validate_review(review(), "cellphone_usage")
        self.assertEqual(result["verdict"], "confirmed")


if __name__ == "__main__":
    unittest.main()
