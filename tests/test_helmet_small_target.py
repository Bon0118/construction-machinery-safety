import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from helmet_review_policy import head_measurement, single_target_size
from offline_alarm_review import AlarmCandidate, call_review_model, validate_review
from algorithms.yolov5_objects.defaults import load_object_settings
from algorithms.yolov5_objects.detector import ObjectDetection, select_alarm_detections


def meta(size=24):
    return dict(detections=[dict(class_id=0, box=[10,10,110,210], head_box=[35,20,35+size,20+size])],
                review_target_id='person_1', helmet_available_frames=[
                    dict(id='T1', seconds=0, image_hash='a'), dict(id='T2', seconds=1, image_hash='b')])


def raw():
    return dict(target_id='person_1', verdict='confirmed', alarm_probability=.92, verdict_confidence=.9,
                evidence_quality='medium', observations=dict(person_visible='yes', head_visible='yes',
                    headgear_type='bare_head', hard_shell_visible='no', helmet_outline_visible='no',
                    multi_frame_consistent='yes', visible_head_features=['hair_structure'], evidence_frames=['T1','T2'],
                    image_quality='可以辨认头发结构'), summary='目标人员多个关键帧中可见头发结构',
                reason_codes=['bare_head_visible'], needs_manual_review=False)


class HelmetSmallTargetTests(unittest.TestCase):
    def test_body_box_kept_and_actual_head_saved(self):
        person = ObjectDetection(0, 'person', .8, (10,10,110,210))
        head = ObjectDetection(1, 'head', .8, (35,20,59,44))
        result = select_alarm_detections(load_object_settings('safety_helmet'), [person,head])
        self.assertEqual(result[0].box, person.box)
        self.assertEqual(result[0].to_dict()['head_box'], list(head.box))
        self.assertEqual(head_measurement(result[0].to_dict()), (24, 'detected_head'))

    def test_body_size_is_not_a_head_measurement(self):
        self.assertEqual(head_measurement(dict(class_id=0, box=[0,0,100,300])), (None,'unknown'))

    def test_no_largest_person_size_borrowing(self):
        data = meta()
        data['detections'].append(dict(class_id=1, box=[0,0,100,100]))
        self.assertIsNone(single_target_size(data))
        self.assertEqual(validate_review(raw(), 'safety_helmet', data)['verdict'], 'uncertain')

    def test_small_target_can_confirm_with_all_evidence(self):
        for size in (16,24,31):
            self.assertEqual(validate_review(raw(), 'safety_helmet', meta(size))['verdict'], 'confirmed')

    def test_below_16_remains_uncertain(self):
        result = validate_review(raw(), 'safety_helmet', meta(15))
        self.assertEqual(result['verdict'], 'uncertain')

    def test_missing_duplicate_or_close_frames_do_not_pass(self):
        for kind in ('missing','duplicate','close'):
            data = meta()
            if kind == 'missing': data['helmet_available_frames'] = []
            if kind == 'duplicate': data['helmet_available_frames'][1]['image_hash'] = 'a'
            if kind == 'close': data['helmet_available_frames'][1]['seconds'] = .1
            self.assertEqual(validate_review(raw(), 'safety_helmet', data)['verdict'], 'uncertain')

    def test_wrong_person_no_features_or_unclear_head_do_not_pass(self):
        for kind in ('identity','features','head','confidence'):
            value = raw()
            if kind == 'identity': value['target_id'] = 'person_2'
            if kind == 'features': value['observations']['visible_head_features'] = []
            if kind == 'head': value['observations']['head_visible'] = 'unclear'
            if kind == 'confidence': value['verdict_confidence'] = .8
            self.assertEqual(validate_review(value, 'safety_helmet', meta())['verdict'], 'uncertain')

    def test_unknown_size_requires_same_strong_evidence(self):
        data = meta()
        del data['detections'][0]['head_box']
        self.assertEqual(validate_review(raw(),'safety_helmet',data)['verdict'],'confirmed')
        data['helmet_available_frames'] = []
        self.assertEqual(validate_review(raw(),'safety_helmet',data)['verdict'],'uncertain')

    def test_tiny_person_without_head_box_cannot_bypass_pixel_floor(self):
        data = meta()
        data['detections'] = [dict(class_id=0, box=[10,10,21,60])]
        result = validate_review(raw(), 'safety_helmet', data)
        self.assertIsNone(result['target_short_edge_pixels'])
        self.assertEqual(result['verdict'], 'uncertain')

    def test_multiple_people_are_reviewed_separately_and_failure_does_not_erase_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.random.RandomState(42).randint(0,255,(240,320,3),dtype=np.uint8)
            cv2.imwrite(str(root/'original.jpg'), frame)
            cv2.imwrite(str(root/'frame.jpg'), frame)
            cv2.imwrite(str(root/'frame2.jpg'), cv2.convertScaleAbs(frame, alpha=.99))
            candidate = AlarmCandidate('test', root, root/'metadata.json',
                dict(algorithm_code='safety_helmet', detections=meta()['detections']*2))
            positive = dict(verdict='confirmed', summary='bare head', needs_manual_review=False)
            with patch('offline_alarm_review.verify_person', return_value=(dict(passed=True,rejected=False), '{}', {})), patch('offline_alarm_review._call_review_model_once',
                       side_effect=[(positive, json.dumps(raw()), {}), TimeoutError('test timeout')]) as request:
                result, _, _ = call_review_model(candidate,[dict(path=str(root/'frame.jpg'),seconds=0),dict(path=str(root/'frame2.jpg'),seconds=.5)],
                    root/'contact.jpg','',{}, {},'unused',180,1536,root/'crop.jpg')
            self.assertEqual(request.call_count,2)
            self.assertEqual([c.args[0].metadata['review_target_id'] for c in request.call_args_list],['person_1','person_2'])
            self.assertTrue(all(len(c.args[0].metadata['detections'])==1 for c in request.call_args_list))
            self.assertEqual(result['verdict'],'confirmed')
            self.assertEqual(result['target_reviews'][1]['verdict'],'uncertain')
