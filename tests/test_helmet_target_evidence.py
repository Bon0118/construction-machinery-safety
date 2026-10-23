import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from helmet_target_evidence import build_target_evidence, read_image, save_image
from offline_alarm_review import AlarmCandidate, call_review_model


class TargetEvidenceTests(unittest.TestCase):
    def test_person_gate_blocks_nonhuman_and_unclear_without_helmet_call(self):
        for rejected in (True,False):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                save_image(root/'original.jpg',np.zeros((100,100,3),np.uint8))
                candidate = AlarmCandidate('wall',root,root/'metadata.json',dict(
                    algorithm_code='safety_helmet',detections=[dict(class_id=0,box=[10,10,80,90])]))
                gate = dict(passed=False,rejected=rejected,confidence=.99,summary='墙体或无法确认人体')
                with patch('offline_alarm_review.verify_person',return_value=(gate,'{}',{})), patch('offline_alarm_review._call_review_model_once') as helmet:
                    result,_,_ = call_review_model(candidate,[],root/'contact.jpg','',{}, {},'unused',180,100)
                helmet.assert_not_called()
                self.assertEqual(result['verdict'],'rejected' if rejected else 'uncertain')
                self.assertEqual(result['needs_manual_review'],not rejected)

    def test_tracking_translation_and_tight_crop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rng = np.random.RandomState(14)
            image = rng.randint(0,255,(240,320,3),dtype=np.uint8)
            original = save_image(root/'original.jpg',image)
            moved = cv2.warpAffine(image,np.float32([[1,0,4],[0,1,3]]),(320,240))
            frame = save_image(root/'frame.jpg',moved)
            crop, context, frames, diagnostics = build_target_evidence(original,[80,60,160,180],
                [dict(path=frame,seconds=.5)],root,'person_1')
            self.assertEqual(read_image(crop).shape[:2],(120,80))
            self.assertEqual(len(frames),1)
            self.assertAlmostEqual(frames[0]['target_box'][0],84,delta=1)
            self.assertAlmostEqual(frames[0]['target_box'][1],63,delta=1)

    def test_textureless_target_cannot_claim_tracking(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = save_image(root/'original.jpg',np.zeros((240,320,3),np.uint8))
            _,_,frames,_ = build_target_evidence(original,[80,60,160,180],
                [dict(path=original,seconds=.5)],root,'person_1')
            self.assertEqual(frames,[])
