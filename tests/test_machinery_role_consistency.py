import unittest
from offline_alarm_review import validate_review


class RoleConsistencyTests(unittest.TestCase):
    def review(self, role, verdict, probability, codes):
        return validate_review({
            "verdict": verdict, "alarm_probability": probability,
            "verdict_confidence": .95, "evidence_quality": "high",
            "observations": {"person_role": role}, "reason_codes": codes,
            "summary": "original explanation", "needs_manual_review": False,
        }, "machinery_hazard")

    def test_conflicting_cab_confirmation_requires_review(self):
        result = self.review("operator_in_cab", "confirmed", .85, ["active_operation_visible"])
        self.assertEqual(result["verdict"], "uncertain")
        self.assertTrue(result["needs_manual_review"])
        self.assertIn("矛盾", result["summary"])
        self.assertEqual(result["model_verdict"], "confirmed")

    def test_clear_cab_rejection_remains_rejected(self):
        self.assertEqual(self.review("operator_in_cab", "rejected", .1, [])['verdict'], 'rejected')

    def test_external_worker_remains_confirmed(self):
        self.assertEqual(self.review("pedestrian", "confirmed", .85, ["active_operation_visible"])['verdict'], 'confirmed')

    def test_conflicting_reason_code_requires_review(self):
        self.assertEqual(self.review("pedestrian", "confirmed", .85, ["operator_in_cab"])['verdict'], 'uncertain')
