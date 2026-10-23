"""Per-person original-pixel measurements and small-head evidence checks."""
import hashlib
from pathlib import Path
from typing import Mapping
import cv2
import numpy as np

MIN_HEAD_PIXELS = 16
SMALL_HEAD_PIXELS = 32
SMALL_HEAD_MIN_CONFIDENCE = .85
MIN_FRAME_SEPARATION = .30


def head_measurement(detection):
    box = detection.get('head_box')
    source = 'detected_head'
    if box is None and detection.get('class_id') == 1:
        box = detection.get('box')
        source = 'legacy_head'
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            w, h = int(box[2])-int(box[0]), int(box[3])-int(box[1])
            if w > 0 and h > 0:
                # Bound malformed head coordinates to their associated body.
                body = detection.get('box')
                if detection.get('class_id') == 0 and body:
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if not (body[0] <= cx <= body[2] and body[1] <= cy <= body[1]+.42*(body[3]-body[1])):
                        return None, 'unknown'
                return min(w, h), source
        except (TypeError, ValueError):
            pass
    return None, 'unknown'


def single_target_size(metadata):
    detections = (metadata or {}).get('detections', [])
    # Never borrow the largest person's size for another person's decision.
    if len(detections) != 1 or not isinstance(detections[0], Mapping):
        return None
    return head_measurement(detections[0])[0]


def head_size_upper_bound(metadata):
    """Body envelope bound for tiny targets, never a measured head size."""
    detections = (metadata or {}).get('detections', [])
    if len(detections) != 1 or detections[0].get('class_id') != 0:
        return None
    box = detections[0].get('box', [])
    try:
        edge = min(int(box[2])-int(box[0]), int(box[3])-int(box[1]))
        return edge if edge > 0 else None
    except (IndexError, TypeError, ValueError):
        return None


def frame_manifest(keyframes):
    frames = []
    for index, item in enumerate(keyframes, 1):
        image = cv2.imdecode(np.frombuffer(Path(item['path']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or item.get('seconds') is None:
            continue
        frames.append(dict(id='T'+str(index), seconds=float(item['seconds']),
                           image_hash=hashlib.sha256(image.tobytes()).hexdigest()))
    return frames


def small_target_supported(value, metadata):
    observations = value.get('observations', {})
    if any(observations.get(key) != 'yes' for key in ('person_visible', 'head_visible', 'multi_frame_consistent')):
        return False
    if value.get('verdict_confidence', 0) < SMALL_HEAD_MIN_CONFIDENCE:
        return False
    kind = observations.get('headgear_type')
    features = set(observations.get('visible_head_features', []))
    allowed = {'bare_head': {'hair_structure', 'scalp_contour'},
               'ordinary_hat': {'fabric_cap', 'soft_hat_brim'},
               'safety_helmet': {'rigid_shell', 'helmet_outline'}}
    if not features.intersection(allowed.get(kind, set())):
        return False
    if kind in ('bare_head', 'ordinary_hat') and (value.get('verdict') != 'confirmed' or value.get('alarm_probability', 0) < .85):
        return False
    available = {f['id']: f for f in (metadata or {}).get('helmet_available_frames', [])}
    chosen = [available[key] for key in set(observations.get('evidence_frames', [])) if key in available]
    return any(a['image_hash'] != b['image_hash'] and abs(a['seconds']-b['seconds']) >= MIN_FRAME_SEPARATION
               for i, a in enumerate(chosen) for b in chosen[i+1:])
