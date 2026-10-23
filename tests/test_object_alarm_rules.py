import unittest
from collections import deque
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from algorithms.yolov5_objects.defaults import load_object_settings
from algorithms.yolov5_objects.detector import (
    ObjectDetection,
    YoloV5ObjectAlgorithm,
    apply_algorithm_parameters,
    select_alarm_detections,
    update_alarm_temporal_history,
)


def detection(class_id, box, confidence=0.9, class_name=None):
    names = {0: "person", 1: "head", 2: "helmet"}
    return ObjectDetection(class_id, class_name or names.get(class_id, "target"), confidence, box)


class ObjectAlarmRuleTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_object_settings("safety_helmet")

    def test_empty_frame_does_not_raise_missing_helmet_alarm(self):
        self.assertEqual(select_alarm_detections(self.settings, []), [])

    def test_person_without_helmet_raises_alarm(self):
        person = detection(0, (100, 50, 300, 500))

        result = select_alarm_detections(self.settings, [person])

        self.assertEqual(result, [person])

    def test_person_with_helmet_in_head_zone_is_safe(self):
        person = detection(0, (100, 50, 300, 500))
        helmet = detection(2, (155, 55, 235, 125))

        result = select_alarm_detections(self.settings, [person, helmet])

        self.assertEqual(result, [])

    def test_unrelated_helmet_does_not_make_person_safe(self):
        person = detection(0, (100, 50, 300, 500))
        unrelated_helmet = detection(2, (500, 50, 580, 125))

        result = select_alarm_detections(self.settings, [person, unrelated_helmet])

        self.assertEqual(result, [person])

    def test_bare_head_uses_person_box_without_duplicate(self):
        person = detection(0, (100, 50, 300, 500))
        head = detection(1, (160, 65, 230, 145))

        result = select_alarm_detections(self.settings, [person, head])

        self.assertEqual(result, [person])

    def test_bare_head_without_person_does_not_emit_head_box(self):
        head = detection(1, (160, 65, 230, 145))
        self.assertEqual(select_alarm_detections(self.settings, [head]), [])

    def test_multiple_heads_for_same_person_emit_one_person_box(self):
        person = detection(0, (100, 50, 300, 500))
        heads = [detection(1, (160, 65, 230, 145)), detection(1, (165, 70, 235, 150))]
        self.assertEqual(select_alarm_detections(self.settings, [person] + heads), [person])

    def test_helmet_on_same_head_suppresses_person_alarm(self):
        person = detection(0, (100, 50, 300, 500))
        head = detection(1, (160, 65, 230, 145))
        helmet = detection(2, head.box)
        self.assertEqual(select_alarm_detections(self.settings, [person, head, helmet]), [])

    def test_other_algorithms_alarm_when_target_is_detected(self):
        fire_settings = load_object_settings("fire_detection")
        fire = detection(0, (20, 20, 80, 90), class_name="fire")

        result = select_alarm_detections(fire_settings, [fire])

        self.assertEqual(result, [fire])

    def test_cigarette_without_person_does_not_raise_alarm(self):
        settings = load_object_settings("cigarette_detection")
        cigarette = detection(2, (180, 120, 205, 150), class_name="cigarette")

        result = select_alarm_detections(settings, [cigarette])

        self.assertEqual(result, [])

    def test_person_without_cigarette_does_not_raise_alarm(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")

        result = select_alarm_detections(settings, [person])

        self.assertEqual(result, [])

    def test_person_and_cigarette_together_raise_smoking_alarm(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cigarette = detection(2, (180, 120, 205, 150), class_name="cigarette")

        result = select_alarm_detections(settings, [cigarette, person])

        self.assertEqual(result, [person, cigarette])

    def test_distant_person_and_cigarette_do_not_raise_smoking_alarm(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cigarette = detection(2, (700, 120, 725, 150), class_name="cigarette")

        result = select_alarm_detections(settings, [person, cigarette])

        self.assertEqual(result, [])

    def test_cigarette_slightly_outside_person_box_matches_expanded_zone(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cigarette = detection(2, (305, 120, 325, 145), class_name="cigarette")

        result = select_alarm_detections(settings, [person, cigarette])

        self.assertEqual(result, [person, cigarette])

    def test_cigarette_below_upper_body_zone_does_not_raise_alarm(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cigarette = detection(2, (180, 430, 205, 455), class_name="cigarette")

        result = select_alarm_detections(settings, [person, cigarette])

        self.assertEqual(result, [])

    def test_alarm_evidence_only_keeps_person_associated_with_cigarette(self):
        settings = load_object_settings("cigarette_detection")
        associated_person = detection(0, (100, 50, 300, 500), class_name="person")
        unrelated_person = detection(0, (600, 50, 800, 500), class_name="person")
        cigarette = detection(2, (180, 120, 205, 150), class_name="cigarette")

        result = select_alarm_detections(
            settings,
            [associated_person, unrelated_person, cigarette],
        )

        self.assertEqual(result, [associated_person, cigarette])

    def test_fire_is_not_mistaken_for_person_in_smoking_rule(self):
        settings = load_object_settings("cigarette_detection")
        fire = detection(0, (20, 20, 80, 90), class_name="fire")
        cigarette = detection(2, (180, 120, 205, 150), class_name="cigarette")

        result = select_alarm_detections(settings, [fire, cigarette])

        self.assertEqual(result, [])

    def test_cellphone_near_person_upper_body_raises_candidate(self):
        settings = load_object_settings("cellphone_usage")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cellphone = detection(6, (210, 180, 245, 230), class_name="cellphone")

        result = select_alarm_detections(settings, [person, cellphone])

        self.assertEqual(result, [person, cellphone])

    def test_distant_cellphone_does_not_raise_candidate(self):
        settings = load_object_settings("cellphone_usage")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        cellphone = detection(6, (600, 180, 640, 230), class_name="cellphone")

        result = select_alarm_detections(settings, [person, cellphone])

        self.assertEqual(result, [])

    def test_cellphone_requires_two_hits_in_three_frame_window(self):
        settings = load_object_settings("cellphone_usage")
        history = deque(maxlen=settings.alarm_window_size)
        cellphone = detection(6, (210, 180, 245, 230), class_name="cellphone")

        self.assertFalse(update_alarm_temporal_history(settings, [cellphone], history))
        self.assertFalse(update_alarm_temporal_history(settings, [], history))
        self.assertTrue(update_alarm_temporal_history(settings, [cellphone], history))

    def test_rebar_temporal_rule_requires_same_area(self):
        settings = load_object_settings("rebar_detection")
        history = deque(maxlen=settings.alarm_window_size)
        first = detection(10, (100, 100, 300, 180), class_name="Rebar")
        nearby = detection(10, (110, 102, 310, 182), class_name="Rebar")

        self.assertFalse(update_alarm_temporal_history(settings, [first], history))
        self.assertTrue(update_alarm_temporal_history(settings, [nearby], history))

    def test_rebar_temporal_rule_rejects_unrelated_area(self):
        settings = load_object_settings("rebar_detection")
        history = deque(maxlen=settings.alarm_window_size)
        first = detection(10, (100, 100, 300, 180), class_name="Rebar")
        distant = detection(10, (500, 300, 700, 380), class_name="Rebar")

        self.assertFalse(update_alarm_temporal_history(settings, [first], history))
        self.assertFalse(update_alarm_temporal_history(settings, [distant], history))

    def test_fainting_uses_temporal_confirmation_and_longer_cooldown(self):
        settings = load_object_settings("fainting_detection")
        history = deque(maxlen=settings.alarm_window_size)
        fainting = detection(11, (100, 300, 400, 450), class_name="fainting")

        self.assertEqual(settings.alarm_cooldown_seconds, 60.0)
        self.assertFalse(update_alarm_temporal_history(settings, [fainting], history))
        self.assertTrue(update_alarm_temporal_history(settings, [fainting], history))

    def test_helmet_is_not_mistaken_for_cigarette_in_smoking_rule(self):
        settings = load_object_settings("cigarette_detection")
        person = detection(0, (100, 50, 300, 500), class_name="person")
        helmet = detection(2, (155, 55, 235, 125), class_name="helmet")

        result = select_alarm_detections(settings, [person, helmet])

        self.assertEqual(result, [])

    def test_helmet_uses_independent_person_head_and_helmet_thresholds(self):
        settings = replace(
            self.settings,
            person_confidence_threshold=0.6,
            bare_head_confidence_threshold=0.7,
            helmet_confidence_threshold=0.2,
        )
        low_person = detection(0, (100, 50, 300, 500), confidence=0.5)
        low_head = detection(1, (160, 65, 230, 145), confidence=0.65)
        valid_helmet = detection(2, (155, 55, 235, 125), confidence=0.25)

        self.assertEqual(
            select_alarm_detections(settings, [low_person, low_head, valid_helmet]),
            [],
        )
        person = detection(0, (100, 50, 300, 500), confidence=0.8)
        self.assertEqual(select_alarm_detections(settings, [person, valid_helmet]), [])

    def test_cigarette_uses_independent_target_and_person_thresholds(self):
        settings = replace(
            load_object_settings("cigarette_detection"),
            confidence_threshold=0.7,
            person_confidence_threshold=0.4,
        )
        person = detection(0, (100, 50, 300, 500), confidence=0.5, class_name="person")
        weak_cigarette = detection(2, (180, 120, 205, 150), confidence=0.65, class_name="cigarette")
        strong_cigarette = detection(2, (180, 120, 205, 150), confidence=0.8, class_name="cigarette")

        self.assertEqual(select_alarm_detections(settings, [person, weak_cigarette]), [])
        self.assertEqual(
            select_alarm_detections(settings, [person, strong_cigarette]),
            [person, strong_cigarette],
        )

    def test_minimum_target_area_filters_small_direct_detections(self):
        settings = replace(
            load_object_settings("fire_detection"),
            minimum_target_area_ratio=0.02,
        )
        small = detection(0, (0, 0, 10, 10), class_name="fire")
        large = detection(0, (0, 0, 50, 50), class_name="fire")

        self.assertEqual(select_alarm_detections(settings, [small], (100, 100, 3)), [])
        self.assertEqual(select_alarm_detections(settings, [large], (100, 100, 3)), [large])

    def test_consecutive_hit_parameters_expand_temporal_window(self):
        helmet = apply_algorithm_parameters(
            self.settings,
            {"violation_consecutive_hits": 3},
        )
        fire = apply_algorithm_parameters(
            load_object_settings("fire_detection"),
            {"alarm_min_hits": 4},
        )

        self.assertEqual((helmet.alarm_window_size, helmet.alarm_min_hits), (3, 3))
        self.assertEqual((fire.alarm_window_size, fire.alarm_min_hits), (4, 4))

    @patch("algorithms.yolov5_objects.detector.MODEL_POOL.get")
    def test_cigarette_image_processing_loads_person_evidence_model(self, model_get):
        cigarette = detection(2, (40, 30, 55, 48), class_name="cigarette")
        person = detection(0, (20, 10, 80, 95), class_name="person")
        maritime_runtime = Mock()
        maritime_runtime.predict.return_value = [cigarette]
        helmet_runtime = Mock()
        helmet_runtime.predict.return_value = [person]
        model_get.side_effect = lambda settings: (
            helmet_runtime if settings.model_key == "helmet" else maritime_runtime
        )

        result = YoloV5ObjectAlgorithm("cigarette_detection").process_image_array(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )

        maritime_runtime.predict.assert_called_once()
        helmet_runtime.predict.assert_called_once()
        self.assertEqual(
            {item["class_name"] for item in result["alarm_detections"]},
            {"person", "cigarette"},
        )


if __name__ == "__main__":
    unittest.main()
