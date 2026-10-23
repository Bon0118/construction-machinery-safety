"""Task-level runner that coordinates all algorithms selected in one video task."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Optional

from algorithms.machinery_hazard.defaults import ALGORITHM_CODE as MACHINERY_HAZARD_CODE
from algorithms.yolov5_objects.defaults import OBJECT_ALGORITHM_DEFINITIONS
from algorithms.yolov5_objects.detector import YoloV5ObjectTaskRunner


class CompositeAlgorithmRunner:
    """Run algorithm families together and aggregate their task result."""

    code = "composite"
    name = "多算法检测"

    def __init__(self, runners: Iterable[Any], algorithm_codes: Iterable[str]) -> None:
        self.runners = list(runners)
        self.algorithm_codes = list(dict.fromkeys(str(value) for value in algorithm_codes))
        if not self.runners:
            raise ValueError("请至少选择一个检测算法")

    def config_snapshot(self) -> Dict[str, Any]:
        snapshots: Dict[str, Any] = {}
        for runner in self.runners:
            callback = getattr(runner, "config_snapshot", None)
            if callable(callback):
                snapshots[str(getattr(runner, "code", type(runner).__name__))] = dict(callback())
        return snapshots

    def process_image_array(self, image: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "algorithms": [runner.process_image_array(image, **kwargs) for runner in self.runners],
        }

    def process_video(self, source: str, **kwargs: Any) -> Dict[str, Any]:
        if len(self.runners) == 1:
            return self.runners[0].process_video(source, **kwargs)

        external_stop: Optional[threading.Event] = kwargs.get("stop_event")
        shared_stop = threading.Event()
        results: List[Optional[Dict[str, Any]]] = [None] * len(self.runners)
        errors: List[Optional[BaseException]] = [None] * len(self.runners)

        def run(index: int, runner: Any) -> None:
            local_kwargs = dict(kwargs)
            local_kwargs["stop_event"] = shared_stop
            context = dict(local_kwargs.get("event_context") or {})
            if str(getattr(runner, "code", "")) == MACHINERY_HAZARD_CODE:
                snapshot = getattr(runner, "config_snapshot", None)
                algorithm_config = dict(snapshot()) if callable(snapshot) else {}
                interval_overrides = local_kwargs.get("algorithm_alarm_intervals") or {}
                if MACHINERY_HAZARD_CODE in interval_overrides:
                    algorithm_config["alarm_cooldown_seconds"] = float(
                        interval_overrides[MACHINERY_HAZARD_CODE]
                    )
                context.update(
                    algorithm_code=MACHINERY_HAZARD_CODE,
                    algorithm_name=str(getattr(runner, "name", "机械伤害预警")),
                    algorithm_config=algorithm_config,
                )
            local_kwargs["event_context"] = context
            try:
                results[index] = runner.process_video(source, **local_kwargs)
            except BaseException as exc:
                errors[index] = exc
            finally:
                # If one family fails or reaches end-of-stream, release the
                # other family instead of leaving a half-running task behind.
                shared_stop.set()

        threads = [
            threading.Thread(target=run, args=(index, runner), daemon=True, name=f"algorithm-family-{index}")
            for index, runner in enumerate(self.runners)
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            if external_stop is not None and external_stop.is_set():
                shared_stop.set()
            for thread in threads:
                thread.join(timeout=0.1)
        first_error = next((item for item in errors if item is not None), None)
        if first_error is not None:
            raise RuntimeError(f"多算法任务运行失败：{first_error}") from first_error
        completed = [item or {} for item in results]
        return {
            "source": source,
            "frames": max((int(item.get("frames", 0) or 0) for item in completed), default=0),
            "alarm_count": sum(int(item.get("alarm_count", 0) or 0) for item in completed),
            "stop_reason": "task_stopped" if external_stop is not None and external_stop.is_set() else "completed",
            "algorithm_codes": self.algorithm_codes,
            "algorithm_results": completed,
        }


def create_task_algorithm_runner(
    algorithm_codes: Iterable[str],
    machinery_runner: Optional[Any] = None,
) -> CompositeAlgorithmRunner:
    codes = list(dict.fromkeys(str(value or "").strip() for value in algorithm_codes if str(value or "").strip()))
    unknown = [
        code for code in codes
        if code != MACHINERY_HAZARD_CODE and code not in OBJECT_ALGORITHM_DEFINITIONS
    ]
    if unknown:
        raise ValueError(f"包含未注册的算法：{', '.join(unknown)}")
    runners: List[Any] = []
    if MACHINERY_HAZARD_CODE in codes:
        if machinery_runner is None:
            raise ValueError("机械伤害预警运行器未连接")
        runners.append(machinery_runner)
    object_codes = [code for code in codes if code in OBJECT_ALGORITHM_DEFINITIONS]
    if MACHINERY_HAZARD_CODE in codes:
        # model2 owns the supplementary Excavator/CraneArm classes. Keep this
        # internal runner out of the task selector while enabling it whenever
        # machinery-hazard analysis is selected.
        object_codes.append("machinery_hazard_supplement")
    if object_codes:
        runners.append(YoloV5ObjectTaskRunner(object_codes))
    return CompositeAlgorithmRunner(runners, codes)
