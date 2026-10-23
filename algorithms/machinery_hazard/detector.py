"""Registry adapter for the existing machinery-hazard detection engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Optional

from algorithms.base import AlgorithmDescriptor, DetectionAlgorithm
from algorithms.machinery_hazard.defaults import (
    ALGORITHM_CODE,
    ALGORITHM_NAME,
    CONFIG_VERSION,
    SCENE_MODE,
    load_default_settings,
)
from algorithms.machinery_hazard.settings import MachineryHazardSettings


MACHINERY_HAZARD_DESCRIPTOR = AlgorithmDescriptor(
    code=ALGORITHM_CODE,
    name=ALGORITHM_NAME,
    config_version=CONFIG_VERSION,
    description=("识别塔式/移动式起重机展开吊臂下方及附近施工人员，吊钩仅作辅助证据"
                 if SCENE_MODE == "crane_boom_person" else "识别人员进入工程机械危险区域的风险事件"),
)


class MachineryHazardAlgorithm(DetectionAlgorithm):
    """Expose the machinery engine through the common algorithm contract."""

    descriptor = MACHINERY_HAZARD_DESCRIPTOR

    def __init__(
        self,
        settings: Optional[MachineryHazardSettings] = None,
        core_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.settings = settings or load_default_settings()
        self.settings.validate()
        if core_factory is None:
            # Lazy import avoids a circular import while server.py still owns
            # common stream, motion, and alarm evidence infrastructure.
            from server import MachineryHazardDetector

            core_factory = MachineryHazardDetector
        self.core = core_factory(model_path=self.settings.model_path, settings=self.settings)

    def process_video(self, source: str, **kwargs: Any) -> Dict[str, Any]:
        context = dict(kwargs.get("event_context") or {})
        effective_config = dict(self.config_snapshot())
        parameter_map = kwargs.get("algorithm_parameters") or {}
        parameters = parameter_map.get(ALGORITHM_CODE, {})
        setting_updates = {
            key: parameters[key]
            for key in (
                "person_confidence_threshold",
                "confidence_threshold",
                "overlap_ratio_threshold",
                "risk_consecutive_frames",
            )
            if key in parameters
        }
        if setting_updates:
            effective_settings = replace(self.settings, **setting_updates)
            effective_settings.validate()
            effective_config = effective_settings.to_dict()
        interval_overrides = kwargs.get("algorithm_alarm_intervals") or {}
        if ALGORITHM_CODE in interval_overrides:
            effective_config["alarm_cooldown_seconds"] = float(
                interval_overrides[ALGORITHM_CODE]
            )
        context.update(
            algorithm_code=ALGORITHM_CODE,
            algorithm_name=ALGORITHM_NAME,
            algorithm_config=effective_config,
        )
        kwargs["event_context"] = context
        return self.core.process_video(source, **kwargs)

    def process_image_array(self, image: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.core.process_image_array(image, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)


def create_machinery_hazard_algorithm(**kwargs: Any) -> MachineryHazardAlgorithm:
    return MachineryHazardAlgorithm(**kwargs)
