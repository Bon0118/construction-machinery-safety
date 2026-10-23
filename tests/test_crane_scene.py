import copy
import json
import tempfile
import threading
import unittest
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from algorithms.machinery_hazard.defaults import load_default_settings, ALARM_RISK_TEXT
from algorithms.yolov5_objects.defaults import load_object_settings
from offline_alarm_review import load_prompt_bundle, review_scene_key, validate_review, prompt_for
from server import Detection, MachineryHazardDetector, AlarmImageStore


class CraneSceneTests(unittest.TestCase):
    def setUp(self):
        self.detector = MachineryHazardDetector.__new__(MachineryHazardDetector)
        self.detector.settings = load_default_settings()
        self.detector._analysis_lock = threading.RLock()
        self.detector._draw_alarm_overlay = lambda frame, *args, **kwargs: frame
        self.frame = np.zeros((400, 600, 3), np.uint8)
        self.person = Detection(0, 'person', .9, (200, 280, 220, 360))
        self.crane = Detection(3, 'mobile_crane', .9, (100, 30, 400, 200))
        self.metadata = {'scene_mode': 'crane_boom_person', 'risks': [
            {'person': self.person.to_dict(), 'machinery': self.crane.to_dict()}]}
        self.raw = dict(verdict='confirmed', alarm_probability=.9, verdict_confidence=.9,
                        evidence_quality='high', summary='展开吊臂下的人员',
                        reason_codes=['crane_boom_area_hazard'], needs_manual_review=False,
                        observations=dict(candidate_index=0, person_visible='yes', target_visible='yes',
                                          person_role='pedestrian', physical_barrier='no',
                                          crane_class_match='yes', boom_visible='yes',
                                          crane_state='deployed', boom_person_relation='under_boom',
                                          same_physical_space='yes', motion_evidence='静止'))

    def analyze(self, detections, supplements=(), analyzer=None):
        self.detector._predict = Mock(return_value=detections)
        with patch('config.ALARM_REVIEW_ENABLED', True):
            return self.detector.analyze_frame(self.frame, supplemental_detections=supplements,
                                               motion_analyzer=analyzer)

    def test_defaults_restrict_classes_and_disable_old_gates(self):
        settings = self.detector.settings
        self.assertEqual(settings.machinery_class_ids, (1, 3))
        self.assertFalse(settings.machinery_motion_enabled)
        self.assertFalse(settings.rider_filter_enabled)
        self.assertEqual(ALARM_RISK_TEXT, '吊装下禁止站人')
        self.assertEqual(load_object_settings('machinery_hazard_supplement').class_ids, (9,))

    def test_yolo_receives_only_person_cranes_and_auxiliary_hook(self):
        self.detector.model = Mock()
        self.detector.model.predict.return_value = [Mock(boxes=None)]
        self.detector._predict(self.frame)
        self.assertEqual(self.detector.model.predict.call_args.kwargs['classes'], [0, 1, 2, 3])

    def test_saved_event_routes_to_dedicated_prompt(self):
        analysis = self.analyze([self.person, self.crane])
        with tempfile.TemporaryDirectory() as folder:
            record = AlarmImageStore(Path(folder)).create_event(
                self.frame, self.frame, analysis, 'test', 'test', datetime.now(),
                event_context={'algorithm_code': 'machinery_hazard', 'alarm_content': '旧文字'})
            metadata = json.loads(record.metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['scene_mode'], 'crane_boom_person')
            self.assertEqual(metadata['alarm_content'], '吊装下禁止站人')
            candidate = Mock(metadata=metadata, algorithm_code='machinery_hazard',
                             algorithm_name='机械伤害预警', event_id='test', channel_name='test',
                             event_time='test', event_dir=record.event_dir)
            _, prompts, _ = load_prompt_bundle(Path('review/prompts'))
            self.assertIn('此任务是区域禁入判断', prompt_for(candidate, prompts))

    def test_stationary_crane_without_hook_or_motion_is_candidate(self):
        analyzer = Mock()
        result = self.analyze([self.person, self.crane], analyzer=analyzer)
        self.assertEqual(len(result.risks), 1)
        self.assertEqual(result.risks[0].motion_reason, 'crane_boom_area_review')
        analyzer.update_detections.assert_not_called()
        analyzer.evaluate_riders.assert_not_called()

    def test_hook_arm_and_other_machines_cannot_trigger(self):
        for class_id in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            with self.subTest(class_id=class_id):
                item = Detection(class_id, 'other', .9, self.crane.box)
                self.assertFalse(self.analyze([self.person, item], [item]).risks)

    def test_tower_crane_triggers_but_auxiliary_arm_is_not_risk_machine(self):
        tower = Detection(1, 'tower_crane', .9, self.crane.box)
        arm = Detection(13, 'crane_arm', .9, (200, 40, 450, 100))
        excavator = Detection(6, 'excavator', .9, self.crane.box)
        result = self.analyze([self.person, tower], [arm, excavator])
        self.assertEqual([r.machinery.class_id for r in result.risks], [1])
        self.assertIn(13, [d.class_id for d in result.detections])
        self.assertNotIn(6, [d.class_id for d in result.detections])

    def test_person_outside_candidate_region_is_not_candidate(self):
        outside = Detection(0, 'person', .9, (520, 280, 540, 360))
        self.assertFalse(self.analyze([outside, self.crane]).risks)
        self.detector.settings = replace(self.detector.settings, overlap_ratio_threshold=0)
        self.assertFalse(self.analyze([outside, self.crane]).risks)

    def test_disabled_review_cannot_emit_unverified_area_alarm(self):
        self.detector._predict = Mock(return_value=[self.person, self.crane])
        with patch('config.ALARM_REVIEW_ENABLED', False):
            result = self.detector.analyze_frame(self.frame)
        self.assertFalse(result.risks)
        self.assertEqual(result.suppressed_risks[0].suppression_reason, 'crane_scene_requires_review')

    def test_static_under_and_near_boom_can_confirm_without_hook(self):
        for relation in ['under_boom', 'near_boom']:
            self.raw['observations']['boom_person_relation'] = relation
            self.assertEqual(validate_review(self.raw, 'machinery_hazard', self.metadata)['verdict'], 'confirmed')

    def test_incomplete_or_conflicting_confirmation_is_blocked(self):
        for field, value in [('person_visible', 'no'), ('crane_class_match', 'no'),
                             ('boom_visible', 'unclear'), ('crane_state', 'stowed_or_transport'),
                             ('boom_person_relation', 'outside'), ('same_physical_space', 'unclear'),
                             ('physical_barrier', 'yes'), ('candidate_index', 2),
                             ('candidate_index', True), ('person_role', 'operator_in_cab')]:
            with self.subTest(field=field):
                raw = copy.deepcopy(self.raw)
                raw['observations'][field] = value
                result = validate_review(raw, 'machinery_hazard', self.metadata)
                self.assertEqual(result['verdict'], 'uncertain')
                self.assertTrue(result['needs_manual_review'])

    def test_explicit_counterevidence_rejection_is_preserved(self):
        self.raw.update(verdict='rejected', alarm_probability=.1)
        self.raw['observations']['crane_state'] = 'stowed_or_transport'
        self.assertEqual(validate_review(self.raw, 'machinery_hazard', self.metadata)['verdict'], 'rejected')

    def test_cab_exemption_requires_visual_fields_and_independent_verification(self):
        raw = copy.deepcopy(self.raw)
        raw.update(verdict='rejected', alarm_probability=.1, reason_codes=['operator_in_cab'])
        raw['observations'].update(person_role='operator_in_cab', boom_person_relation='outside')
        self.assertEqual(validate_review(raw, 'machinery_hazard', self.metadata)['verdict'], 'uncertain')
        raw['observations'].update(cab_visible='yes', person_inside_cab='yes',
                                   cab_belongs_to_crane='yes', riding_two_wheeler='no')
        self.assertEqual(validate_review(raw, 'machinery_hazard', self.metadata)['verdict'], 'uncertain')
        meta = dict(self.metadata, cab_verification={'verified': True})
        self.assertEqual(validate_review(raw, 'machinery_hazard', meta)['verdict'], 'rejected')
        raw['observations']['riding_two_wheeler'] = 'yes'
        self.assertEqual(validate_review(raw, 'machinery_hazard', meta)['verdict'], 'uncertain')

    def test_old_events_keep_old_prompt_and_new_schema_is_loaded(self):
        _, prompts, schemas = load_prompt_bundle(Path('review/prompts'))
        self.assertEqual(review_scene_key('machinery_hazard', {}), 'machinery_hazard')
        self.assertEqual(review_scene_key('machinery_hazard', self.metadata), 'crane_boom_person')
        self.assertIn('crane_boom_person', prompts)
        self.assertIn('boom_person_relation', schemas['crane_boom_person']['properties']['observations']['required'])


if __name__ == '__main__':
    unittest.main()
