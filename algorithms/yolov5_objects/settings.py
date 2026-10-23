"""Validated settings for one selectable YOLOv5 object algorithm."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from algorithms.base import AlgorithmSettings


@dataclass(frozen=True)
class ObjectDetectionSettings(AlgorithmSettings):
    algorithm_code: str
    algorithm_name: str
    description: str
    model_key: str
    model_path: Path
    class_ids: Tuple[int, ...]
    class_labels: Mapping[int, str]
    expected_model_names: Tuple[str, ...]
    alarm_rule: str = "detected"
    confidence_threshold: float = 0.25
    person_confidence_threshold: float = 0.25
    bare_head_confidence_threshold: float = 0.25
    helmet_confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "0"
    detection_interval_seconds: float = 0.5
    alarm_cooldown_seconds: float = 30.0
    person_association_expand_ratio: float = 0.15
    person_association_upper_body_ratio: float = 0.70
    alarm_window_size: int = 1
    alarm_min_hits: int = 1
    alarm_temporal_iou_threshold: float = 0.0
    minimum_target_area_ratio: float = 0.0
    alarm_min_duration_seconds: float = 0.0
    config_version: int = 1

    def validate(self) -> None:
        if not self.algorithm_code.strip() or not self.algorithm_name.strip():
            raise ValueError("algorithm_code 和 algorithm_name 不能为空")
        if not self.model_path.is_file():
            raise ValueError(f"模型文件不存在：{self.model_path}")
        if not self.class_ids:
            raise ValueError("class_ids 不能为空")
        if self.alarm_rule not in {
            "detected",
            "missing_helmet",
            "person_with_cigarette",
            "person_with_cellphone",
            "publish_machinery_supplement",
        }:
            raise ValueError(f"不支持的 alarm_rule：{self.alarm_rule}")
        if any(int(value) < 0 for value in self.class_ids):
            raise ValueError("class_ids 不能包含负数")
        for name in (
            "confidence_threshold",
            "person_confidence_threshold",
            "bare_head_confidence_threshold",
            "helmet_confidence_threshold",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if not 0.0 <= float(self.iou_threshold) <= 1.0:
            raise ValueError("iou_threshold 必须在 0 到 1 之间")
        if int(self.image_size) < 1:
            raise ValueError("image_size 必须大于 0")
        if float(self.detection_interval_seconds) < 0:
            raise ValueError("detection_interval_seconds 不能小于 0")
        if float(self.alarm_cooldown_seconds) < 0:
            raise ValueError("alarm_cooldown_seconds 不能小于 0")
        if float(self.person_association_expand_ratio) < 0:
            raise ValueError("person_association_expand_ratio 不能小于 0")
        if not 0.0 < float(self.person_association_upper_body_ratio) <= 1.0:
            raise ValueError("person_association_upper_body_ratio 必须大于 0 且小于等于 1")
        if int(self.alarm_window_size) < 1:
            raise ValueError("alarm_window_size 必须大于等于 1")
        if not 1 <= int(self.alarm_min_hits) <= int(self.alarm_window_size):
            raise ValueError("alarm_min_hits 必须在 1 到 alarm_window_size 之间")
        if not 0.0 <= float(self.alarm_temporal_iou_threshold) <= 1.0:
            raise ValueError("alarm_temporal_iou_threshold 必须在 0 到 1 之间")
        if not 0.0 <= float(self.minimum_target_area_ratio) <= 1.0:
            raise ValueError("minimum_target_area_ratio 必须在 0 到 1 之间")
        if float(self.alarm_min_duration_seconds) < 0:
            raise ValueError("alarm_min_duration_seconds 不能小于 0")

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["model_path"] = str(self.model_path)
        values["class_ids"] = list(self.class_ids)
        values["class_labels"] = {str(key): value for key, value in self.class_labels.items()}
        values["expected_model_names"] = list(self.expected_model_names)
        return values
