"""Create traceable square head-and-shoulder crops from full-frame person boxes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from dataset_annotation.annotation.rules import Box, PrelabelBox


@dataclass(frozen=True)
class HeadShoulderCrop:
    image: np.ndarray
    crop_box_original: Box
    source_person_box_original: Box
    person_box: Box
    person_truncated: bool
    output_size: int

    def metadata(self) -> Dict[str, Any]:
        return {
            "crop_box_original": list(self.crop_box_original),
            "source_person_box_original": list(self.source_person_box_original),
            "person_box_in_crop": list(self.person_box),
            "person_truncated": self.person_truncated,
            "output_size": [self.output_size, self.output_size],
        }

    def remap_original_box(self, box: Box) -> Box | None:
        crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_box_original
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        if not (crop_x1 <= center_x <= crop_x2 and crop_y1 <= center_y <= crop_y2):
            return None
        scale = self.output_size / float(crop_x2 - crop_x1)
        x1 = max(box[0], crop_x1)
        y1 = max(box[1], crop_y1)
        x2 = min(box[2], crop_x2)
        y2 = min(box[3], crop_y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (
            max(0, min(self.output_size, int(round((x1 - crop_x1) * scale)))),
            max(0, min(self.output_size, int(round((y1 - crop_y1) * scale)))),
            max(0, min(self.output_size, int(round((x2 - crop_x1) * scale)))),
            max(0, min(self.output_size, int(round((y2 - crop_y1) * scale)))),
        )

    def remap_crop_box(self, box: Box) -> Box | None:
        """Map a crop-space label back to the preserved full-frame image."""
        crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_box_original
        scale = float(crop_x2 - crop_x1) / self.output_size
        x1 = crop_x1 + int(round(max(0, min(self.output_size, box[0])) * scale))
        y1 = crop_y1 + int(round(max(0, min(self.output_size, box[1])) * scale))
        x2 = crop_x1 + int(round(max(0, min(self.output_size, box[2])) * scale))
        y2 = crop_y1 + int(round(max(0, min(self.output_size, box[3])) * scale))
        x1, y1 = max(crop_x1, x1), max(crop_y1, y1)
        x2, y2 = min(crop_x2, x2), min(crop_y2, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


def _clamped_square(
    center_x: float,
    center_y: float,
    side: int,
    image_width: int,
    image_height: int,
) -> Box:
    side = max(1, min(side, image_width, image_height))
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(0, left), image_width - side)
    top = min(max(0, top), image_height - side)
    return left, top, left + side, top + side


def create_head_shoulder_crop(
    image: np.ndarray,
    person: PrelabelBox,
    crop_config: Dict[str, Any],
) -> HeadShoulderCrop:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = person.box
    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)
    side = int(
        round(
            max(
                float(crop_config["minimum_source_side"]),
                person_width * float(crop_config["person_width_multiplier"]),
                person_height * float(crop_config["person_height_multiplier"]),
            )
        )
    )
    center_x = (x1 + x2) / 2.0
    center_y = y1 + person_height * float(crop_config["vertical_center_ratio"])
    crop_box = _clamped_square(center_x, center_y, side, image_width, image_height)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    source = image[crop_y1:crop_y2, crop_x1:crop_x2]
    output_size = int(crop_config["output_size"])
    resized = cv2.resize(source, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    scale = output_size / float(crop_x2 - crop_x1)

    visible_x1 = max(x1, crop_x1)
    visible_y1 = max(y1, crop_y1)
    visible_x2 = min(x2, crop_x2)
    visible_y2 = min(y2, crop_y2)
    person_box = (
        max(0, min(output_size, int(round((visible_x1 - crop_x1) * scale)))),
        max(0, min(output_size, int(round((visible_y1 - crop_y1) * scale)))),
        max(0, min(output_size, int(round((visible_x2 - crop_x1) * scale)))),
        max(0, min(output_size, int(round((visible_y2 - crop_y1) * scale)))),
    )
    person_truncated = (
        visible_x1 != x1 or visible_y1 != y1 or visible_x2 != x2 or visible_y2 != y2
    )
    return HeadShoulderCrop(
        image=resized,
        crop_box_original=crop_box,
        source_person_box_original=person.box,
        person_box=person_box,
        person_truncated=person_truncated,
        output_size=output_size,
    )


def associated_with_person_head_region(candidate: Box, person: Box, crop_config: Dict[str, Any]) -> bool:
    person_x1, person_y1, person_x2, person_y2 = person
    width = max(1, person_x2 - person_x1)
    height = max(1, person_y2 - person_y1)
    center_x = (candidate[0] + candidate[2]) / 2.0
    center_y = (candidate[1] + candidate[3]) / 2.0
    horizontal_expand = width * float(crop_config["association_horizontal_expand"])
    top_expand = height * float(crop_config["association_top_expand"])
    bottom = person_y1 + height * float(crop_config["association_bottom_ratio"])
    return (
        person_x1 - horizontal_expand <= center_x <= person_x2 + horizontal_expand
        and person_y1 - top_expand <= center_y <= bottom
    )
