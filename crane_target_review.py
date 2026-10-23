"""Review each crane/person pair using its own images and a shared time budget."""
import json
import time
import shutil
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

from helmet_target_evidence import (build_target_evidence, crop_target, read_image,
                                    save_image, target_contact_sheet)


def verify_crane_cab_role(context_path, pair_path, pair_id, server_url, model_name, timeout):
    """Check whether the marked person is inside this pair's crane cab."""
    from offline_alarm_review import image_data_url

    choices = ['yes', 'no', 'unclear']
    schema = dict(type='object', additionalProperties=False,
                  required=['review_pair_id', 'cab_visible', 'person_inside_cab',
                            'cab_belongs_to_crane', 'riding_two_wheeler', 'confidence', 'summary'],
                  properties={
                      'review_pair_id': dict(type='string'),
                      **{key: dict(type='string', enum=choices) for key in
                         ('cab_visible', 'person_inside_cab', 'cab_belongs_to_crane', 'riding_two_wheeler')},
                      'confidence': dict(type='number', minimum=0, maximum=1),
                      'summary': dict(type='string'),
                  })
    prompt = (
        '只核验红框内的同一人员是否身体处于蓝框起重机的驾驶室内部，不判断吊臂危险。'
        '第一张图为标框原图，第二张为这对目标的局部上下文。必须看到可辨识驾驶室边界、'
        '红框人员在其内部、且驾驶室属于蓝框起重机，才能分别回答yes。'
        '骑电动车、自行车、摩托车回答riding_two_wheeler=yes，不能称为驾驶室人员。'
        '坐姿、安全帽、车辆靠近、其他机械的驾驶室均不足以证明。看不清回答unclear。'
        'review_pair_id原样返回：' + pair_id
    )
    body = dict(model=model_name, temperature=0, max_tokens=300,
                chat_template_kwargs={'enable_thinking': False},
                response_format={'type': 'json_schema',
                                 'json_schema': {'name': 'crane_cab_gate', 'strict': True, 'schema': schema}},
                messages=[dict(role='user', content=[
                    dict(type='text', text=prompt),
                    dict(type='image_url', image_url={'url': image_data_url(context_path)}),
                    dict(type='image_url', image_url={'url': image_data_url(pair_path)}),
                ])])
    request = urllib.request.Request(server_url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                     headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    raw = payload['choices'][0]['message']['content']
    result = json.loads(raw)
    result['verified'] = bool(result.get('review_pair_id') == pair_id
                              and result.get('confidence', 0) >= .85
                              and all(result.get(key) == 'yes' for key in
                                      ('cab_visible', 'person_inside_cab', 'cab_belongs_to_crane'))
                              and result.get('riding_two_wheeler') == 'no')
    return result, raw, payload.get('usage', {})


def incomplete(reason):
    return dict(verdict='uncertain', alarm_probability=.5, verdict_confidence=0,
                evidence_quality='low', observations={}, reason_codes=['insufficient_evidence'],
                summary=reason, needs_manual_review=True, policy_adjustments=['pair_review_incomplete'])


def pair_images(original_path, pair, root, pair_id):
    image = read_image(original_path)
    marked = image.copy()
    boxes = []
    for name, color in [('person', (0, 0, 255)), ('machinery', (255, 0, 0))]:
        box = list(map(int, pair[name]['box']))
        crop_target(image, box)  # Validate before marking/cropping.
        boxes.append(box)
        cv2.rectangle(marked, tuple(box[:2]), tuple(box[2:]), color, 2)
        cv2.putText(marked, pair_id+' '+name, (box[0], max(18, box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    region = pair.get('candidate_region') or {}
    for aux in region.get('auxiliary_evidence', []) + region.get('ambiguous_evidence', []):
        boxes.append(aux['box'])
    h, w = image.shape[:2]
    left, top = max(0, int(min(b[0] for b in boxes))-20), max(0, int(min(b[1] for b in boxes))-20)
    right, bottom = min(w, int(max(b[2] for b in boxes))+20), min(h, int(max(b[3] for b in boxes))+20)
    context = save_image(root/(pair_id+'_context.jpg'), marked)
    crop = save_image(root/(pair_id+'_pair.jpg'), image[top:bottom, left:right])
    return context, crop, [left, top, right, bottom]


def review_pairs(candidate, keyframes, contact_sheet_path, system_prompt, algorithm_prompts,
                 review_schema, server_url, timeout, max_tokens, target_crop_path, model_name,
                 invoke, verify_person):
    root = Path(target_crop_path or contact_sheet_path).parent / 'crane_pairs'
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+timeout
    results, raw_results, usage_total, person_cache = [], [], {}, {}
    pairs = candidate.metadata.get('risks', [])

    def collect(usage):
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                usage_total[key] = usage_total.get(key, 0)+value

    for index, pair in enumerate(pairs):
        pair_id = 'pair_'+str(index+1)
        reviewed = incomplete('候选复核尚未完成')
        try:
            remaining = deadline-time.monotonic()
            if remaining <= 1:
                raise TimeoutError('逐对复核总时间预算已用尽；未复核候选保留为不确定')
            if pair['person']['class_id'] != 0 or pair['machinery']['class_id'] not in (1, 3):
                raise ValueError('候选目标类别不属于人员与两类起重机')
            # Cache only exactly identical person boxes, never nearby people.
            person_key = tuple(pair['person']['box'])
            if person_key not in person_cache:
                person_id = 'person_'+str(len(person_cache)+1)
                tight, _, frames, tracking = build_target_evidence(
                    candidate.event_dir/'original.jpg', person_key, keyframes[:6], root, person_id)
                gate, raw, usage = verify_person(tight, person_id, server_url, model_name,
                                                max(.1, (deadline-time.monotonic())/2))
                collect(usage)
                raw_results.append(dict(pair_id=pair_id, stage='person_gate', response=raw))
                person_cache[person_key] = tight, frames, tracking, gate
            tight, frames, tracking, gate = person_cache[person_key]
            if not gate['passed']:
                reviewed = incomplete('独立人体核验：'+str(gate.get('summary', '证据不足')))
                if gate['rejected']:
                    reviewed.update(verdict='rejected', alarm_probability=0,
                                    verdict_confidence=gate.get('confidence', 0), evidence_quality='medium',
                                    reason_codes=['false_person_detection'], needs_manual_review=False)
            else:
                context, crop, crop_box = pair_images(candidate.event_dir/'original.jpg', pair, root, pair_id)
                # No untracked full-frame worker is allowed to stand in for this person.
                sheet = target_contact_sheet(frames, root/(pair_id+'_person_frames.jpg')) if frames else tight
                region = pair.get('candidate_region') or {}
                meta = dict(candidate.metadata, risks=[pair], risk_count=1, review_pair_id=pair_id,
                            crane_context_image=str(context), crane_person_crop=str(tight),
                            crane_pair_crop_box=crop_box, crane_local_frame_count=len(frames),
                            operation_evidence=region.get('auxiliary_evidence', []),
                            person_verification=gate)
                meta.pop('detections', None)
                child = replace(candidate, metadata=meta)
                remaining = deadline-time.monotonic()
                if remaining <= 1:
                    raise TimeoutError('人体核验后复核预算已用尽')
                reviewed, raw, usage = invoke(child, frames, sheet, system_prompt, algorithm_prompts,
                                              review_schema, server_url, remaining, max_tokens, crop, model_name)
                collect(usage)
                raw_results.append(dict(pair_id=pair_id, stage='pair_review', response=raw))
                observations = reviewed.get('observations') or {}
                if (observations.get('person_role') == 'operator_in_cab'
                        or 'operator_in_cab' in reviewed.get('reason_codes', [])):
                    cab_gate = {'verified': False, 'summary': '驾驶室专项核验未完成'}
                    try:
                        remaining = deadline-time.monotonic()
                        if remaining <= 1:
                            raise TimeoutError('驾驶室专项核验预算已用尽')
                        cab_gate, cab_raw, cab_usage = verify_crane_cab_role(
                            context, crop, pair_id, server_url, model_name, remaining)
                        collect(cab_usage)
                        raw_results.append(dict(pair_id=pair_id, stage='cab_role_gate', response=cab_raw))
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        cab_gate['summary'] = '驾驶室专项核验未完成：'+str(exc)
                        raw_results.append(dict(pair_id=pair_id, stage='cab_role_gate', error=str(exc)))
                    from offline_alarm_review import validate_review
                    reviewed = validate_review(json.loads(raw), child.algorithm_code,
                                               dict(meta, cab_verification=cab_gate))
                    reviewed['cab_verification'] = cab_gate
            reviewed.update(person_verification=gate, target_tracking=tracking)
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            reviewed = incomplete('候选复核未完成：'+str(exc))
            raw_results.append(dict(pair_id=pair_id, error=str(exc)))
        reviewed.update(pair_id=pair_id, source_risk_index=index)
        results.append(reviewed)
    if not results:
        results = [dict(incomplete('无有效人员—起重机候选'), pair_id='none', source_risk_index=-1)]
    confirmed = [r for r in results if r['verdict'] == 'confirmed']
    uncertain = [r for r in results if r['verdict'] == 'uncertain']
    selected = (confirmed or uncertain or results)[0]
    aggregate = dict(selected, pair_reviews=results,
                     confirmed_risk_indices=[r['source_risk_index'] for r in confirmed],
                     needs_manual_review=bool(uncertain) or any(r.get('needs_manual_review') for r in results),
                     summary='；'.join(r['pair_id']+'：'+r['summary'] for r in results))
    return aggregate, json.dumps(raw_results, ensure_ascii=False), usage_total


def render_confirmed_alarm(candidate, review):
    """Publishable image contains only individually confirmed pairs; retain candidate image."""
    from server import BufferGeometry, Detection, MachineryHazardDetector, RiskMatch
    indices = review.get('confirmed_risk_indices', [])
    allowed = {r['source_risk_index'] for r in review.get('pair_reviews', []) if r['verdict'] == 'confirmed'}
    pairs = candidate.metadata.get('risks', [])
    if not indices or any(type(i) is not int or i not in allowed or not 0 <= i < len(pairs) for i in indices):
        raise ValueError('没有可绘制的逐对确认结果')
    risks = []
    for index in indices:
        pair = pairs[index]
        def detection(name):
            item = pair[name]
            return Detection(item['class_id'], item['class_name'], item['confidence'], tuple(item['box']))
        risks.append(RiskMatch(detection('person'), detection('machinery'),
                               BufferGeometry('rectangle', (0, 0), (1, 1)), pair.get('overlap_ratio', 0),
                               motion_reason='crane_boom_area_review'))
    detector = MachineryHazardDetector.__new__(MachineryHazardDetector)
    image = detector._draw_alarm_overlay(read_image(candidate.event_dir/'original.jpg'), risks,
                                        candidate.channel_name, datetime.now())
    temporary = save_image(candidate.event_dir/'alarm_confirmed.tmp.jpg', image)
    original_alarm = candidate.event_dir/'alarm.jpg'
    backup = candidate.event_dir/'alarm_candidates.jpg'
    if original_alarm.is_file() and not backup.exists():
        shutil.copy2(original_alarm, backup)
    temporary.replace(original_alarm)
    return original_alarm
