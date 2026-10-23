import unittest
from unittest.mock import patch
import cv2
import numpy as np
from server import MachineryMotionAnalyzer, Detection
from algorithms.machinery_hazard import defaults as cfg
from algorithms.machinery_hazard.motion_evidence import BodyVisualEvidence
from offline_alarm_review import validate_review


class MotionEvidenceTests(unittest.TestCase):
    def texture(self):
        rng = np.random.RandomState(7)
        return cv2.GaussianBlur(rng.randint(0, 256, (180, 220), dtype=np.uint8), (3, 3), 0)

    def test_static_texture_cannot_confirm_box_motion(self):
        gray = self.texture()
        evidence = BodyVisualEvidence()
        for i in range(8):
            evidence.observe(gray, gray, (10+i*2, 10, 190+i*2, 160), [], i*.25, True)
        self.assertFalse(evidence.confirms(1.75, (1, 0)))

    def test_coherent_image_translation_is_confirmed(self):
        gray = self.texture()
        shifted = cv2.warpAffine(gray, np.float32([[1, 0, 1], [0, 1, 0]]), (220, 180))
        evidence = BodyVisualEvidence()
        for i in range(5):
            evidence.observe(gray, shifted, (10, 10, 210, 170), [], i*.25, True)
        self.assertTrue(evidence.confirms(1, (1, 0)))
        self.assertFalse(evidence.confirms(1, (-1, 0)))
        self.assertFalse(evidence.confirms(3, (1, 0)))

    def test_person_local_motion_does_not_confirm_whole_body(self):
        gray = self.texture()
        moved = gray.copy()
        moved[70:100, 80:110] = np.roll(moved[70:100, 80:110], 3, axis=1)
        evidence = BodyVisualEvidence()
        for i in range(5):
            evidence.observe(gray, moved, (10, 10, 210, 170), [(78, 68, 112, 102)], i*.25, True)
        self.assertFalse(evidence.confirms(1, (1, 0)))

    def test_unreliable_camera_clears_visual_votes(self):
        evidence = BodyVisualEvidence()
        evidence.samples.append((0, .1, 0, True))
        gray = self.texture()
        evidence.observe(gray, gray, (10, 10, 210, 170), [], 1, False)
        self.assertFalse(evidence.reliable)
        self.assertFalse(evidence.samples)

    def trajectory(self, xs, sizes=None):
        analyzer = MachineryMotionAnalyzer('test')
        analyzer._camera_reliable = True
        values = []
        for i, x in enumerate(xs):
            size = sizes[i] if sizes else 100
            boxes, state = analyzer.update_detections([Detection(3, 'mobile_crane', .9, (x, 30, x+size, 130))], [], i*.5)
            values.append(state[0])
        return analyzer, values

    def test_box_back_and_forth_rejected_even_with_visual_vote(self):
        with patch.object(BodyVisualEvidence, 'confirms', return_value=True):
            analyzer, states = self.trajectory([30, 34, 30, 34, 30, 34])
        self.assertFalse(any(s.body_moving for s in states))

    def test_slow_consistent_translation_with_visual_evidence_passes(self):
        with patch.object(BodyVisualEvidence, 'confirms', return_value=True):
            _, states = self.trajectory(list(range(30, 42)))
        self.assertTrue(any(s.body_moving for s in states))

    def test_size_jump_resets_old_motion(self):
        with patch.object(BodyVisualEvidence, 'confirms', return_value=True):
            _, states = self.trajectory([30, 34, 38, 42, 46, 50], [100, 100, 100, 100, 100, 150])
        self.assertTrue(states[-2].body_moving)
        self.assertFalse(states[-1].body_moving)
        self.assertFalse(states[-1].box_stable)

    def test_static_review_has_independent_entry_but_excludes_giant_box(self):
        analyzer, _ = self.trajectory([30]*7)
        person = Detection(0, 'person', .9, (40, 150, 60, 190))
        machine = Detection(3, 'mobile_crane', .9, (30, 30, 130, 130), track_id=1)
        hook = Detection(2, 'hook', .8, (40, 70, 60, 90))
        self.assertFalse(analyzer.static_operation_candidate(machine, person, (400, 500, 3)))
        self.assertTrue(analyzer.static_operation_candidate(machine, person, (400, 500, 3), [hook]))
        giant = Detection(3, 'mobile_crane', .9, (0, 0, 500, 400), track_id=1)
        self.assertFalse(analyzer.static_operation_candidate(giant, person, (400, 500, 3), [hook]))
        with patch('config.ALARM_REVIEW_ENABLED', False):
            self.assertFalse(analyzer.static_operation_candidate(machine, person, (400, 500, 3), [hook]))

    def test_static_review_cannot_confirm_without_path_b(self):
        raw = dict(verdict='confirmed', alarm_probability=.9, verdict_confidence=.9,
                   evidence_quality='high', observations={'person_role': 'pedestrian'},
                   summary='test', reason_codes=['dangerous_proximity'], needs_manual_review=False)
        metadata = {'risks': [{'motion_reason': 'static_operation_review'}]}
        result = validate_review(raw, 'machinery_hazard', metadata)
        self.assertEqual(result['verdict'], 'uncertain')
        raw['reason_codes'] = ['active_operation_visible']
        self.assertEqual(validate_review(raw, 'machinery_hazard', metadata)['verdict'], 'confirmed')

    def image_sequence(self, body_speed=0.0, camera_speed=0.0, box_drift=0.0):
        rng = np.random.RandomState(42)
        background = cv2.GaussianBlur(rng.randint(0, 256, (360, 480), dtype=np.uint8), (3, 3), 0)
        texture = cv2.GaussianBlur(rng.randint(0, 256, (100, 100), dtype=np.uint8), (3, 3), 0)
        analyzer = MachineryMotionAnalyzer('image_sequence')
        states = []
        for i in range(33):
            t = i*.25
            body_dx, camera_dx = body_speed*t, camera_speed*t
            patch_image = np.zeros_like(background)
            patch_image[100:200, 120:220] = texture
            patch_mask = np.zeros_like(background)
            patch_mask[100:200, 120:220] = 255
            transform = np.float32([[1, 0, body_dx], [0, 1, 0]])
            moving = cv2.warpAffine(patch_image, transform, (480, 360))
            mask = cv2.warpAffine(patch_mask, transform, (480, 360)).astype(float)/255
            gray = (background*(1-mask)+moving).clip(0,255).astype(np.uint8)
            gray = cv2.warpAffine(gray, np.float32([[1, 0, camera_dx], [0, 1, 0]]), (480, 360), borderMode=cv2.BORDER_REFLECT)
            analyzer.observe_frame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1+t)
            if i % 2 == 0:
                x = round(120+body_dx+camera_dx+box_drift*t)
                _, assessment = analyzer.update_detections([Detection(3, 'mobile_crane', .9, (x,100,x+100,200))], [], 1+t)
                states.append((t, assessment[0]))
        return states

    def test_image_sequence_slow_vehicle_is_not_lost(self):
        states = self.image_sequence(body_speed=2.0)
        self.assertTrue(any(s.body_moving for _, s in states))
        self.assertLessEqual(next(t for t, s in states if s.body_moving), 4.0)

    def test_image_sequence_camera_pan_is_not_vehicle_motion(self):
        self.assertFalse(any(s.body_moving for _, s in self.image_sequence(camera_speed=8.0)))

    def test_image_sequence_drifting_box_on_static_scene_is_rejected(self):
        self.assertFalse(any(s.body_moving for _, s in self.image_sequence(box_drift=6.0)))


if __name__ == '__main__':
    unittest.main()
