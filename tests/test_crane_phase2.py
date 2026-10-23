import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from algorithms.machinery_hazard.crane_regions import build_regions, deduplicate, foot_in_region
from crane_target_review import review_pairs, render_confirmed_alarm
from helmet_target_evidence import save_image
from offline_alarm_review import AlarmCandidate, validate_review, call_review_model, load_prompt_bundle
from server import Detection


class CraneRegionTests(unittest.TestCase):
    def setUp(self):
        self.crane = Detection(3, 'mobile_crane', .9, (100, 100, 300, 220))
        self.arm = Detection(13, 'crane_arm', .9, (220, 80, 450, 200))

    def test_extended_boom_recalls_person_outside_vehicle_box(self):
        region = build_regions([self.crane], [self.arm], (500, 600, 3))[0]
        self.assertEqual(region['source'], 'associated_boom')
        self.assertTrue(foot_in_region((410, 220, 430, 300), region))
        self.assertFalse(foot_in_region((90, 220, 110, 300), region))

    def test_overlapping_cranes_do_not_silently_claim_same_boom(self):
        other = Detection(1, 'tower_crane', .9, self.crane.box)
        regions = build_regions([self.crane, other], [self.arm], (500, 600, 3))
        for region in regions:
            self.assertEqual(region['source'], 'ambiguous_fallback')
            self.assertEqual(region['auxiliary_evidence'], [])
            self.assertEqual(len(region['ambiguous_evidence']), 1)
            self.assertTrue(foot_in_region((410, 220, 430, 300), region))

    def test_remote_auxiliary_is_not_borrowed(self):
        remote = Detection(13, 'crane_arm', .9, (400, 0, 550, 40))
        region = build_regions([self.crane], [remote], (500, 600, 3))[0]
        self.assertEqual(region['source'], 'crane_fallback')
        self.assertEqual(region['auxiliary_evidence'], [])

    def test_weak_attachment_keeps_fallback_instead_of_narrowing_region(self):
        weak = Detection(13, 'crane_arm', .9, (299, 100, 490, 140))
        region = build_regions([self.crane], [weak], (500, 600, 3))[0]
        self.assertEqual(region['source'], 'ambiguous_fallback')
        self.assertTrue(foot_in_region((140, 220, 160, 300), region))

    def test_hook_alone_does_not_create_boom_region(self):
        hook = Detection(2, 'hook', .9, (150, 150, 160, 160))
        region = build_regions([self.crane], [hook], (500, 600, 3))[0]
        self.assertEqual(region['source'], 'crane_fallback')

    def test_low_confidence_boom_cannot_narrow_region(self):
        weak = Detection(13, 'crane_arm', .3, self.arm.box)
        region = build_regions([self.crane], [weak], (500, 600, 3))[0]
        self.assertEqual(region['source'], 'crane_fallback')

    def test_duplicate_boxes_removed_but_adjacent_people_kept(self):
        person = Detection(0, 'person', .9, (10, 20, 30, 80))
        duplicate = Detection(0, 'person', .7, (10, 20, 30, 80))
        adjacent = Detection(0, 'person', .8, (25, 20, 45, 80))
        self.assertEqual(deduplicate([person, duplicate, adjacent]), [person, adjacent])


class CranePairReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        save_image(self.root/'original.jpg', np.zeros((200, 300, 3), np.uint8))
        crane = Detection(3, 'mobile_crane', .9, (20, 20, 180, 100)).to_dict()
        self.pairs = [dict(person=Detection(0, 'person', .9, (30+i*30, 110, 45+i*30, 180)).to_dict(),
                           machinery=crane) for i in range(4)]
        self.candidate = AlarmCandidate(event_id='test', event_dir=self.root,
                                       metadata_path=self.root/'metadata.json',
                                       metadata={'algorithm_code': 'machinery_hazard',
                                                 'scene_mode': 'crane_boom_person', 'risks': self.pairs})
        self.gate = Mock(return_value=({'passed': True, 'rejected': False, 'confidence': .95}, '{}', {}))
        self.seen = []

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, child, *args):
        self.seen.append(child.metadata)
        raw = dict(verdict='confirmed', alarm_probability=.9, verdict_confidence=.9,
                   evidence_quality='high', summary='该人员在该起重机吊臂下', reason_codes=[],
                   needs_manual_review=False, observations=dict(
                       candidate_index=0, review_pair_id=child.metadata['review_pair_id'],
                       person_visible='yes', target_visible='yes', person_role='pedestrian',
                       physical_barrier='no', crane_class_match='yes', boom_visible='yes',
                       crane_state='deployed', boom_person_relation='under_boom',
                       same_physical_space='yes', boom_belongs_to_crane='yes'))
        return validate_review(raw, 'machinery_hazard', child.metadata), json.dumps(raw), {}

    def run_review(self, **kwargs):
        return review_pairs(self.candidate, [], self.root/'sheet.jpg', '', {}, {}, 'unused',
                            kwargs.get('timeout', 60), 1000, None, 'test',
                            kwargs.get('invoke', self.invoke), self.gate)[0]

    def test_every_pair_including_fourth_has_own_evidence(self):
        result = self.run_review()
        self.assertEqual(result['confirmed_risk_indices'], [0, 1, 2, 3])
        self.assertEqual(len(self.seen), 4)
        self.assertEqual([m['review_pair_id'] for m in self.seen], ['pair_1','pair_2','pair_3','pair_4'])
        for index, meta in enumerate(self.seen):
            self.assertEqual(meta['risks'], [self.pairs[index]])
            self.assertTrue(Path(meta['crane_person_crop']).is_file())
            self.assertEqual(meta['crane_local_frame_count'], 0)

    def test_production_dispatch_uses_pair_reviews_for_new_scene(self):
        with patch('offline_alarm_review.verify_person', self.gate), patch(
                'offline_alarm_review._call_review_model_once', side_effect=self.invoke):
            result, _, _ = call_review_model(self.candidate, [], self.root/'sheet.jpg', '', {}, {},
                                            'unused', 60, 1000)
        self.assertEqual(result['confirmed_risk_indices'], [0, 1, 2, 3])

    def test_http_payload_contains_pair_schema_and_ordered_local_images(self):
        from offline_alarm_review import _call_review_model_once
        original_invoke = self.invoke
        payloads = []
        def invoke_http(child, frames, sheet, *args):
            _, raw, _ = original_invoke(child)
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = json.dumps({'choices':[{'message':{'content':raw}}]}).encode()
            def urlopen(request, timeout):
                payloads.append(json.loads(request.data.decode('utf-8')))
                return response
            system, prompts, schemas = load_prompt_bundle(Path('review/prompts'))
            with patch('offline_alarm_review.urllib.request.urlopen', side_effect=urlopen):
                return _call_review_model_once(child, frames, sheet, system, prompts, schemas,
                                               'http://localhost/test', 10, 1000, args[-2], 'test')
        result = self.run_review(invoke=invoke_http)
        self.assertEqual(result['verdict'], 'confirmed')
        self.assertEqual(len(payloads), 4)
        content = payloads[0]['messages'][1]['content']
        image_titles = [content[i-1]['text'] for i, block in enumerate(content) if block['type']=='image_url']
        self.assertIn('本次唯一人员', image_titles[0])
        self.assertIn('没有可靠的跟踪关键帧', image_titles[2])
        schema = payloads[0]['response_format']['json_schema']['schema']
        self.assertIn('boom_belongs_to_crane', schema['properties']['observations']['required'])

    def test_false_person_never_reaches_relation_review(self):
        self.gate.return_value = {'passed': False, 'rejected': True, 'confidence': .95}, '{}', {}
        result = self.run_review()
        self.assertEqual(result['verdict'], 'rejected')
        self.assertEqual(self.seen, [])

    def test_cab_claim_uses_same_pair_gate_and_keeps_uncorroborated_result_uncertain(self):
        self.candidate.metadata['risks'] = [self.pairs[0]]
        def cab_claim(child, *args):
            raw = dict(verdict='rejected', alarm_probability=.1, verdict_confidence=.95,
                       evidence_quality='high', summary='驾驶室人员', reason_codes=['operator_in_cab'],
                       needs_manual_review=False, observations=dict(
                           candidate_index=0, review_pair_id='pair_1', person_visible='yes',
                           target_visible='yes', person_role='operator_in_cab', cab_visible='yes',
                           person_inside_cab='yes', cab_belongs_to_crane='yes', riding_two_wheeler='no',
                           crane_class_match='yes', boom_visible='yes', crane_state='deployed',
                           boom_person_relation='outside', same_physical_space='yes',
                           boom_belongs_to_crane='yes', physical_barrier='no'))
            return validate_review(raw, 'machinery_hazard', child.metadata), json.dumps(raw), {}
        gate = (dict(review_pair_id='pair_1', verified=False, riding_two_wheeler='yes'), '{}', {})
        with patch('crane_target_review.verify_crane_cab_role', return_value=gate) as verify:
            result = self.run_review(invoke=cab_claim)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['confirmed_risk_indices'], [])
        self.assertEqual(result['pair_reviews'][0]['cab_verification']['riding_two_wheeler'], 'yes')
        self.assertEqual(verify.call_args.args[2], 'pair_1')

    def test_timeout_marks_all_unreviewed_pairs_uncertain(self):
        result = self.run_review(timeout=0)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(len(result['pair_reviews']), 4)
        self.gate.assert_not_called()

    def test_person_gate_reused_only_for_same_box(self):
        self.candidate.metadata['risks'] = [self.pairs[0], self.pairs[0], self.pairs[1]]
        self.run_review()
        self.assertEqual(self.gate.call_count, 2)

    def test_wrong_pair_identity_and_unclear_boom_cannot_confirm(self):
        def wrong(child, *args):
            _, raw, _ = self.invoke(child, *args)
            value = json.loads(raw)
            value['observations'].update(review_pair_id='another_pair', boom_belongs_to_crane='unclear')
            return validate_review(value, 'machinery_hazard', child.metadata), json.dumps(value), {}
        result = self.run_review(invoke=wrong)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['confirmed_risk_indices'], [])

    def test_partial_failure_does_not_hide_confirmed_or_unreviewed_pairs(self):
        def fail_one(child, *args):
            if child.metadata['review_pair_id'] == 'pair_1':
                raise TimeoutError('timeout')
            return self.invoke(child, *args)
        result = self.run_review(invoke=fail_one)
        self.assertEqual(result['verdict'], 'confirmed')
        self.assertEqual(result['confirmed_risk_indices'], [1, 2, 3])
        self.assertTrue(result['needs_manual_review'])

    def test_alarm_redraw_excludes_rejected_pair_and_preserves_candidate_image(self):
        save_image(self.root/'alarm.jpg', np.full((200, 300, 3), 77, np.uint8))
        old_bytes = (self.root/'alarm.jpg').read_bytes()
        review = dict(confirmed_risk_indices=[3], pair_reviews=[
            dict(source_risk_index=0, verdict='rejected'), dict(source_risk_index=3, verdict='confirmed')])
        with patch('server.MachineryHazardDetector._draw_alarm_overlay', return_value=np.zeros((200,300,3),np.uint8)) as draw:
            render_confirmed_alarm(self.candidate, review)
            risks = draw.call_args.args[1]
            self.assertEqual(len(risks), 1)
            self.assertEqual(risks[0].person.box, tuple(self.pairs[3]['person']['box']))
        self.assertEqual((self.root/'alarm_candidates.jpg').read_bytes(), old_bytes)

    def test_unconfirmed_index_cannot_be_drawn_as_confirmed(self):
        with self.assertRaises(ValueError):
            render_confirmed_alarm(self.candidate, dict(confirmed_risk_indices=[0], pair_reviews=[]))
