"""Image-space crane/boom association; never a calibrated physical danger zone."""
import math


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def deduplicate(detections, threshold=.85):
    """Only merge very high-overlap same-class boxes; adjacent workers stay separate."""
    kept = []
    for item in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(item.class_id == other.class_id and
               intersection(item.box, other.box) / max(1, area(item.box)+area(other.box)-intersection(item.box, other.box))
               >= threshold for other in kept):
            continue
        kept.append(item)
    return kept


def association_score(aux, crane, gap_ratio=.08):
    a, c = aux.box, crane.box
    gap = math.hypot(max(c[0]-a[2], a[0]-c[2], 0), max(c[1]-a[3], a[1]-c[3], 0))
    scale = max(1, math.hypot(c[2]-c[0], c[3]-c[1]))
    if gap > gap_ratio*scale:
        return None
    overlap = intersection(a, c)/max(1, min(area(a), area(c)))
    return .7*overlap + .3*(1-min(1, gap/max(1, gap_ratio*scale)))


def build_regions(cranes, detections, image_shape, margin_ratio=.12, score_margin=.15, min_score=.45,
                  min_aux_confidence=.40):
    """Assign auxiliaries only with a clear score winner; keep ambiguous recall regions."""
    height, width = image_shape[:2]
    assigned = [[] for _ in cranes]
    ambiguous = [[] for _ in cranes]
    for aux in detections:
        if aux.class_id not in {2, 13} or aux.confidence < min_aux_confidence:
            continue
        scores = [(association_score(aux, crane), i) for i, crane in enumerate(cranes)]
        scores = sorted((item for item in scores if item[0] is not None), reverse=True)
        if not scores:
            continue
        if scores[0][0] >= min_score and (len(scores) == 1 or scores[0][0]-scores[1][0] >= score_margin):
            assigned[scores[0][1]].append(aux)
        else:
            for _, i in scores:
                ambiguous[i].append(aux)
    regions = []
    for i, crane in enumerate(cranes):
        arms = [a for a in assigned[i] if a.class_id == 13]
        uncertain_arms = [a for a in ambiguous[i] if a.class_id == 13]
        # Ambiguous or absent booms must not turn a body-only box into a hard exclusion.
        sources = arms if arms and not uncertain_arms else [crane]+arms+uncertain_arms
        x1 = min(a.box[0] for a in sources)
        y1 = min(a.box[1] for a in sources)
        x2 = max(a.box[2] for a in sources)
        margin = max(4, (x2-x1)*margin_ratio)
        box = [max(0, math.floor(x1-margin)), max(0, y1), min(width, math.ceil(x2+margin)), height]
        regions.append(dict(
            source='associated_boom' if arms and not uncertain_arms else 'ambiguous_fallback' if uncertain_arms else 'crane_fallback',
            box=box, boom_span=[x1, x2],
            association='ambiguous' if ambiguous[i] else 'candidate' if assigned[i] else 'missing',
            auxiliary_evidence=[a.to_dict() for a in assigned[i]],
            ambiguous_evidence=[a.to_dict() for a in ambiguous[i]],
            geometry_only=True,
        ))
    return regions


def foot_in_region(person_box, region):
    x = (person_box[0]+person_box[2])/2
    y = person_box[3]
    left, top, right, bottom = region['box']
    return left <= x <= right and top <= y <= bottom
