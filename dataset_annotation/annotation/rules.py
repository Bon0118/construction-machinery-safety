"""Safety-helmet v1 box rules and conflict checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PrelabelBox:
    class_id: int
    class_name: str
    confidence: float
    box: Box
    truncated: bool = False
    source: str = "model1_best.pt"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["box"] = list(self.box)
        value["confidence"] = round(float(self.confidence), 5)
        return value


def training_scale(width: int, height: int, target_width: int, target_height: int) -> float:
    return min(float(target_width) / max(1, width), float(target_height) / max(1, height))


def eligible_head_size(
    box: Box,
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    minimum_short_side: float,
) -> bool:
    scale = training_scale(image_width, image_height, target_width, target_height)
    box_width = max(0, box[2] - box[0]) * scale
    box_height = max(0, box[3] - box[1]) * scale
    return min(box_width, box_height) >= float(minimum_short_side)


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def enforce_head_helmet_exclusivity(
    detections: Sequence[PrelabelBox],
    semantic_bucket: str,
) -> Tuple[List[PrelabelBox], List[str]]:
    kept = list(detections)
    issues: List[str] = []
    heads = [item for item in kept if item.class_name == "head"]
    helmets = [item for item in kept if item.class_name == "helmet"]
    for head in heads:
        conflicts = [helmet for helmet in helmets if box_iou(head.box, helmet.box) >= 0.30]
        for helmet in conflicts:
            issues.append("head_helmet_overlap")
            if semantic_bucket == "head":
                if helmet in kept:
                    kept.remove(helmet)
            elif semantic_bucket == "helmet":
                if head in kept:
                    kept.remove(head)
            elif head.confidence >= helmet.confidence:
                if helmet in kept:
                    kept.remove(helmet)
            elif head in kept:
                kept.remove(head)
    return kept, sorted(set(issues))


def attach_truncation(detections: Iterable[PrelabelBox], width: int, height: int) -> List[PrelabelBox]:
    result = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        result.append(
            PrelabelBox(
                class_id=item.class_id,
                class_name=item.class_name,
                confidence=item.confidence,
                box=item.box,
                truncated=x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1,
                source=item.source,
            )
        )
    return result

