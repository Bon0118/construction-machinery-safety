"""Run the existing safety-helmet YOLOv5 model as a recall-oriented pre-labeler."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from algorithms.yolov5_objects.defaults import load_object_settings
from algorithms.yolov5_objects.detector import MODEL_POOL
from dataset_annotation.annotation.rules import (
    PrelabelBox,
    attach_truncation,
    eligible_head_size,
    enforce_head_helmet_exclusivity,
)
from dataset_annotation.annotation.head_shoulder_crop import associated_with_person_head_region


class SafetyHelmetPrelabeler:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.settings = load_object_settings(str(config["prelabel"]["model_code"]))
        self.runtime = MODEL_POOL.get(self.settings)

    def raw_labels(self, image: np.ndarray) -> List[PrelabelBox]:
        raw = self.runtime.predict(
            image,
            self.settings.class_ids,
            confidence_threshold=float(self.config["prelabel"]["confidence_threshold"]),
        )
        class_ids = {"person": 0, "head": 1, "helmet": 2}
        labels = []
        for item in raw:
            class_name = str(item.class_name).lower()
            if class_name not in class_ids:
                continue
            labels.append(
                PrelabelBox(
                    class_id=class_ids[class_name],
                    class_name=class_name,
                    confidence=float(item.confidence),
                    box=tuple(int(value) for value in item.box),
                )
            )
        height, width = image.shape[:2]
        return attach_truncation(labels, width, height)

    def predict(
        self,
        image: np.ndarray,
        semantic_bucket: str,
        raw_labels: Sequence[PrelabelBox] | None = None,
    ) -> Tuple[List[PrelabelBox], List[str]]:
        height, width = image.shape[:2]
        raw = list(raw_labels) if raw_labels is not None else self.raw_labels(image)
        labels: List[PrelabelBox] = []
        issues: List[str] = []
        for item in raw:
            class_name = item.class_name
            if class_name not in {"person", "head", "helmet"}:
                continue
            if class_name in {"head", "helmet"} and not eligible_head_size(
                item.box,
                width,
                height,
                int(self.config["training_width"]),
                int(self.config["training_height"]),
                float(self.config["minimum_head_short_side"]),
            ):
                issues.append(f"{class_name}_below_20px_at_training_size")
                continue
            labels.append(
                PrelabelBox(
                    class_id=int(item.class_id),
                    class_name=class_name,
                    confidence=float(item.confidence),
                    box=item.box,
                    source=item.source,
                )
            )
        labels, conflict_issues = enforce_head_helmet_exclusivity(labels, semantic_bucket)
        issues.extend(conflict_issues)
        labels = attach_truncation(labels, width, height)
        if not any(item.class_name == "person" for item in labels):
            issues.append("person_missing")
        if semantic_bucket in {"head", "helmet"} and not any(
            item.class_name == semantic_bucket for item in labels
        ):
            issues.append(f"review_{semantic_bucket}_not_detected")
        return labels, sorted(set(issues))

    def predict_crop(
        self,
        image: np.ndarray,
        person_box: Tuple[int, int, int, int],
        mapped_source_candidates: Sequence[PrelabelBox] = (),
    ) -> Tuple[List[PrelabelBox], List[str]]:
        """Label one person-centered crop while keeping one traceable person box."""
        height, width = image.shape[:2]
        labels = [
            PrelabelBox(
                class_id=0,
                class_name="person",
                confidence=1.0,
                box=person_box,
                truncated=(
                    person_box[0] <= 1
                    or person_box[1] <= 1
                    or person_box[2] >= width - 1
                    or person_box[3] >= height - 1
                ),
                source="full_frame_person_crop",
            )
        ]
        issues: List[str] = []
        raw = self.runtime.predict(
            image,
            self.settings.class_ids,
            confidence_threshold=float(self.config["prelabel"]["confidence_threshold"]),
        )
        class_ids = {"person": 0, "head": 1, "helmet": 2}
        head_status_candidates: List[PrelabelBox] = list(mapped_source_candidates)
        for item in raw:
            class_name = str(item.class_name).lower()
            if class_name not in {"head", "helmet"}:
                continue
            box = tuple(int(value) for value in item.box)
            if not associated_with_person_head_region(box, person_box, self.config["crop"]):
                issues.append(f"unassociated_{class_name}_ignored")
                continue
            if not eligible_head_size(
                box,
                width,
                height,
                int(self.config["training_width"]),
                int(self.config["training_height"]),
                float(self.config["minimum_head_short_side"]),
            ):
                issues.append(f"crop_{class_name}_below_20px_at_training_size")
                continue
            head_status_candidates.append(
                PrelabelBox(
                    class_id=class_ids[class_name],
                    class_name=class_name,
                    confidence=float(item.confidence),
                    box=box,
                    source="model1_best.pt_on_head_shoulder_crop",
                )
            )
        if head_status_candidates:
            best_status = max(head_status_candidates, key=lambda item: item.confidence)
            labels.append(best_status)
            if len(head_status_candidates) > 1:
                issues.append("multiple_head_status_candidates_reduced_to_best")
        labels = attach_truncation(labels, width, height)
        if not any(item.class_name in {"head", "helmet"} for item in labels):
            issues.append("crop_head_status_not_detected")
        return labels, sorted(set(issues))
