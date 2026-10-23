"""Validated settings snapshot for machinery-hazard detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from algorithms.base import AlgorithmSettings


@dataclass(frozen=True)
class MachineryHazardSettings(AlgorithmSettings):
    model_path: Path
    confidence_threshold: float
    person_confidence_threshold: float
    iou_threshold: float
    image_size: int
    device: str
    max_detections: int
    person_class_ids: Tuple[int, ...]
    machinery_class_ids: Tuple[int, ...]
    detection_interval_seconds: float
    overlap_ratio_threshold: float
    risk_consecutive_frames: int
    alarm_cooldown_seconds: float
    machinery_motion_enabled: bool
    rider_filter_enabled: bool
    advanced: Mapping[str, Any] = field(default_factory=dict)
    config_version: int = 3

    def validate(self) -> None:
        for name in ("confidence_threshold", "person_confidence_threshold", "iou_threshold", "overlap_ratio_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if int(self.image_size) < 1:
            raise ValueError("image_size 必须大于 0")
        if int(self.max_detections) < 1:
            raise ValueError("max_detections 必须大于 0")
        if not self.person_class_ids:
            raise ValueError("person_class_ids 不能为空")
        if not self.machinery_class_ids:
            raise ValueError("machinery_class_ids 不能为空")
        if float(self.detection_interval_seconds) < 0:
            raise ValueError("detection_interval_seconds 不能小于 0")
        if int(self.risk_consecutive_frames) < 1:
            raise ValueError("risk_consecutive_frames 必须大于等于 1")
        if float(self.alarm_cooldown_seconds) < 0:
            raise ValueError("alarm_cooldown_seconds 不能小于 0")

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        advanced = dict(values.pop("advanced", {}))
        values["model_path"] = str(self.model_path)
        values["person_class_ids"] = list(self.person_class_ids)
        values["machinery_class_ids"] = list(self.machinery_class_ids)
        for name, value in advanced.items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, set):
                value = sorted(value)
            values[name] = value
        return values
