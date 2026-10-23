"""Benchmark the four-algorithm video-task profile on one local GPU.

The benchmark uses a real project image and reproduces the current task shape:
machinery hazard + safety helmet + fire + cigarette.  Fire and cigarette share
one maritime model inference, matching the video runner's model-family reuse.
It does not create alarms, modify platform data, or open camera streams.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from algorithms.machinery_hazard.detector import MachineryHazardAlgorithm
from algorithms.yolov5_objects.defaults import load_object_settings
from algorithms.yolov5_objects.detector import MODEL_POOL


PROFILE_CODES = ("machinery_hazard", "safety_helmet", "fire_detection", "cigarette_detection")


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def summarize_seconds(values: List[float]) -> Dict[str, float]:
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values) * 1000.0, 3),
        "median_ms": round(statistics.median(values) * 1000.0, 3),
        "p95_ms": round(percentile(values, 0.95) * 1000.0, 3),
        "max_ms": round(max(values) * 1000.0, 3),
    }


def read_unicode_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取测试图片：{path}")
    return image


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], Any], repeats: int) -> List[float]:
    durations: List[float] = []
    for _ in range(max(1, repeats)):
        synchronize()
        started = time.perf_counter()
        call()
        synchronize()
        durations.append(time.perf_counter() - started)
    return durations


def nvidia_smi_memory() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip().splitlines()[0]
        name, used, total, utilization = [part.strip() for part in output.split(",")]
        return {
            "name": name,
            "used_mb": float(used),
            "total_mb": float(total),
            "utilization_percent": float(utilization),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-repeats", type=int, default=5)
    parser.add_argument("--channels", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境没有可用 CUDA，无法执行 GPU 容量评测")

    frame = read_unicode_image(args.image.resolve())
    height, width = frame.shape[:2]
    channels = sorted({max(1, int(value)) for value in args.channels.split(",") if value.strip()})

    load_started = time.perf_counter()
    machinery = MachineryHazardAlgorithm()
    object_settings = [load_object_settings(code) for code in PROFILE_CODES[1:]]
    grouped: Dict[str, List[Any]] = {}
    for setting in object_settings:
        grouped.setdefault(setting.model_key, []).append(setting)
    object_groups = []
    for model_key, settings in grouped.items():
        runtime = MODEL_POOL.get(settings[0])
        class_ids = sorted({class_id for setting in settings for class_id in setting.class_ids})
        object_groups.append((model_key, runtime, class_ids))
    synchronize()
    load_seconds = time.perf_counter() - load_started

    def run_machinery() -> Any:
        return machinery.analyze_frame(frame, camera_name="benchmark")

    def run_objects() -> Any:
        return [runtime.predict(frame, class_ids) for _, runtime, class_ids in object_groups]

    # The constructors warm up object models; warm the machinery path and full family once.
    for _ in range(3):
        run_machinery()
        run_objects()
    synchronize()

    component_results: Dict[str, Dict[str, float]] = {
        "machinery_hazard": summarize_seconds(timed(run_machinery, args.repeats)),
    }
    for model_key, runtime, class_ids in object_groups:
        component_results[model_key] = summarize_seconds(
            timed(lambda runtime=runtime, class_ids=class_ids: runtime.predict(frame, class_ids), args.repeats)
        )

    def run_channel_batch(channel_count: int) -> None:
        # Each live task owns two algorithm-family threads.  The machinery model
        # is serialized by its analysis lock; each object model is serialized by
        # its model-pool lock, while the two families can overlap on the GPU.
        with ThreadPoolExecutor(max_workers=max(2, channel_count * 2)) as executor:
            futures = []
            for _ in range(channel_count):
                futures.append(executor.submit(run_machinery))
                futures.append(executor.submit(run_objects))
            for future in futures:
                future.result()

    channel_results = []
    for channel_count in channels:
        durations = timed(lambda count=channel_count: run_channel_batch(count), args.batch_repeats)
        summary = summarize_seconds(durations)
        summary.update(
            channel_count=channel_count,
            deadline_ms=500.0,
            meets_500ms_p95=summary["p95_ms"] <= 500.0,
            utilization_at_2hz_percent=round(summary["mean_ms"] / 500.0 * 100.0, 1),
        )
        channel_results.append(summary)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": {
            "algorithm_codes": list(PROFILE_CODES),
            "unique_model_passes_per_channel_tick": 3,
            "detection_interval_seconds": 0.5,
            "note": "fire_detection 与 cigarette_detection 共享 maritime 模型的一次推理",
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        },
        "input": {"path": str(args.image.resolve()), "width": width, "height": height},
        "model_load_seconds": round(load_seconds, 3),
        "torch_memory": {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        },
        "nvidia_smi_after_benchmark": nvidia_smi_memory(),
        "component_latency": component_results,
        "channel_batch_latency": channel_results,
        "method_caveats": [
            "使用同一张真实告警帧模拟多路输入，不包含 RTSP 解码与网络抖动",
            "不触发告警落盘、视频编码或 VLM 复核",
            "容量判断以每路每 0.5 秒执行一次四算法组合为基准",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
