"""Compatible YOLOv5-v6 inference and alarm generation for imported weights."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from alarm_event_schema import (
    ALARM_CLIP_FILENAME,
    ALARM_IMAGE_FILENAME,
    EVENT_DIR_PREFIX,
    METADATA_FILENAME,
    ORIGINAL_IMAGE_FILENAME,
    RISK_CROP_FILENAME,
)
from algorithms.base import AlgorithmDescriptor, DetectionAlgorithm
from algorithms.machinery_supplement import (
    MACHINERY_SUPPLEMENT_BROKER,
    map_supplement_detection,
    supplement_key,
)
from algorithms.yolov5_objects.defaults import (
    OBJECT_ALGORITHM_DEFINITIONS,
    get_object_algorithm_definition,
    load_object_settings,
)
from algorithms.yolov5_objects.settings import ObjectDetectionSettings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV5_ROOT = PROJECT_ROOT / "vendor" / "yolov5_v6"


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    head_box: Optional[Tuple[int, int, int, int]] = field(default=None, compare=False)

    def to_dict(self, display_name: str = "") -> Dict[str, Any]:
        result = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "display_name": display_name or self.class_name,
            "confidence": round(self.confidence, 5),
            "box": list(self.box),
        }
        if self.head_box is not None:
            result["head_box"] = list(self.head_box)
            result["head_size_source"] = "detected_head"
        return result


@dataclass
class ObjectAlarmRecord:
    event_id: str
    event_dir: Path
    annotated_path: Path
    original_path: Path
    crop_path: Path
    metadata_path: Path
    clip_path: Optional[Path] = None


def _safe_stem(value: str) -> str:
    import re

    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(value or "source")).strip("_")
    return text or "source"


def _save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"图片编码失败：{path.name}")
    encoded.tofile(str(path))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    )
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _annotate(frame: np.ndarray, detections: Sequence[ObjectDetection], labels: Mapping[int, str]) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _load_font(max(16, min(30, round(min(frame.shape[:2]) / 30))))
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = (255, 45, 45)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{labels.get(item.class_id, item.class_name)} {item.confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = max(0, y1 - text_height - 10)
        draw.rectangle((x1, top, min(image.width, x1 + text_width + 12), y1), fill=color)
        draw.text((x1 + 6, top + 3), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _point_inside(box: Tuple[int, int, int, int], point: Tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_ratio(
    box: Tuple[int, int, int, int],
    target: Tuple[int, int, int, int],
) -> float:
    """Return how much of ``box`` is covered by ``target``."""

    x1 = max(box[0], target[0])
    y1 = max(box[1], target[1])
    x2 = min(box[2], target[2])
    y2 = min(box[3], target[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, box[2] - box[0]) * max(1, box[3] - box[1])
    return intersection / area


def _helmet_matches_zone(
    helmet: ObjectDetection,
    zone: Tuple[int, int, int, int],
) -> bool:
    hx1, hy1, hx2, hy2 = helmet.box
    center = ((hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0)
    return _point_inside(zone, center) or _intersection_ratio(helmet.box, zone) >= 0.30


def _person_head_zone(person: ObjectDetection) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = person.box
    # The helmet should be associated with the upper portion of the same person,
    # not with any helmet that happens to exist elsewhere in the frame.
    return x1, y1, x2, int(round(y1 + (y2 - y1) * 0.42))


def _person_smoking_zone(
    person: ObjectDetection,
    expand_ratio: float,
    upper_body_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = person.box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = round(width * max(0.0, float(expand_ratio)))
    pad_y = round(height * max(0.0, float(expand_ratio)))
    upper_bottom = round(y1 + height * min(1.0, max(0.0, float(upper_body_ratio))))
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, upper_bottom


def _detection_center(item: ObjectDetection) -> Tuple[float, float]:
    x1, y1, x2, y2 = item.box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _meets_minimum_area(
    item: ObjectDetection,
    frame_shape: Optional[Sequence[int]],
    minimum_ratio: float,
) -> bool:
    threshold = float(minimum_ratio)
    if threshold <= 0 or not frame_shape or len(frame_shape) < 2:
        return True
    frame_area = max(1, int(frame_shape[0]) * int(frame_shape[1]))
    x1, y1, x2, y2 = item.box
    target_area = max(0, x2 - x1) * max(0, y2 - y1)
    return target_area / frame_area >= threshold


def apply_algorithm_parameters(
    settings: ObjectDetectionSettings,
    parameters: Optional[Mapping[str, Any]],
) -> ObjectDetectionSettings:
    raw = dict(parameters or {})
    updates: Dict[str, Any] = {}
    for key in (
        "confidence_threshold",
        "person_confidence_threshold",
        "bare_head_confidence_threshold",
        "helmet_confidence_threshold",
        "alarm_window_size",
        "alarm_min_hits",
        "minimum_target_area_ratio",
        "alarm_min_duration_seconds",
    ):
        if key in raw:
            updates[key] = raw[key]
    if "violation_consecutive_hits" in raw:
        hits = int(raw["violation_consecutive_hits"])
        updates.update(alarm_window_size=hits, alarm_min_hits=hits)
    if settings.algorithm_code == "fire_detection" and "alarm_min_hits" in raw:
        hits = int(raw["alarm_min_hits"])
        updates.update(alarm_window_size=hits, alarm_min_hits=hits)
    effective = replace(settings, **updates) if updates else settings
    effective.validate()
    return effective


def select_alarm_detections(
    settings: ObjectDetectionSettings,
    detections: Sequence[ObjectDetection],
    frame_shape: Optional[Sequence[int]] = None,
) -> List[ObjectDetection]:
    """Apply the algorithm-specific alarm rule to raw model detections."""

    if settings.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}:
        target_names = {
            settings.expected_model_names[class_id].strip().lower()
            for class_id in settings.class_ids
            if 0 <= class_id < len(settings.expected_model_names)
        }
        persons = [
            item
            for item in detections
            if item.class_name.strip().lower() == "person"
            and item.confidence >= float(settings.person_confidence_threshold)
        ]
        cigarettes = [
            item
            for item in detections
            if item.class_id in settings.class_ids
            and item.class_name.strip().lower() in target_names
            and item.confidence >= float(settings.confidence_threshold)
            and _meets_minimum_area(
                item, frame_shape, settings.minimum_target_area_ratio
            )
        ]
        if not persons or not cigarettes:
            return []

        matched_persons: List[ObjectDetection] = []
        matched_cigarettes: List[ObjectDetection] = []
        for person in persons:
            association_zone = _person_smoking_zone(
                person,
                settings.person_association_expand_ratio,
                settings.person_association_upper_body_ratio,
            )
            associated = [
                cigarette
                for cigarette in cigarettes
                if _point_inside(association_zone, _detection_center(cigarette))
            ]
            if not associated:
                continue
            matched_persons.append(person)
            for cigarette in associated:
                if cigarette not in matched_cigarettes:
                    matched_cigarettes.append(cigarette)

        # Person-associated small-object rules are AND + spatial-association
        # rules. Unrelated targets elsewhere in the frame must not alarm.
        return matched_persons + matched_cigarettes

    if settings.alarm_rule != "missing_helmet":
        return [
            item
            for item in detections
            if item.class_id in settings.class_ids
            and item.confidence >= float(settings.confidence_threshold)
            and _meets_minimum_area(
                item, frame_shape, settings.minimum_target_area_ratio
            )
        ]

    persons = [
        item for item in detections
        if item.class_id == 0
        and item.confidence >= float(settings.person_confidence_threshold)
    ]
    bare_heads = [
        item for item in detections
        if item.class_id == 1
        and item.confidence >= float(settings.bare_head_confidence_threshold)
    ]
    helmets = [
        item for item in detections
        if item.class_id == 2
        and item.confidence >= float(settings.helmet_confidence_threshold)
    ]
    violations: List[ObjectDetection] = []
    heads_by_person: Dict[ObjectDetection, ObjectDetection] = {}

    # Associate bare-head evidence with a detected body. Never draw a head-only
    # alarm or invent a body box when the person detector missed the body.
    for head in bare_heads:
        if any(_helmet_matches_zone(helmet, head.box) for helmet in helmets):
            continue
        center = _detection_center(head)
        candidates = [
            person for person in persons
            if _point_inside(_person_head_zone(person), center)
        ]
        if candidates:
            # In overlapping crowds, prefer the tightest enclosing person.
            person = min(candidates, key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]))
            if person not in heads_by_person or head.confidence > heads_by_person[person].confidence:
                heads_by_person[person] = head
            if person not in violations:
                violations.append(person)

    for person in persons:
        head_zone = _person_head_zone(person)
        if any(_helmet_matches_zone(helmet, head_zone) for helmet in helmets):
            continue
        if person in violations:
            continue
        violations.append(person)

    # Every alarm target is a detected person, including bare-head violations.
    return [replace(person, head_box=heads_by_person[person].box)
            if person in heads_by_person else person for person in violations]


def _with_required_person_evidence(
    settings: ObjectDetectionSettings,
    primary_detections: Sequence[ObjectDetection],
    person_detections: Sequence[ObjectDetection],
) -> List[ObjectDetection]:
    evidence = list(primary_detections)
    if settings.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}:
        evidence.extend(
            item for item in person_detections if item.class_name.strip().lower() == "person"
        )
    return evidence


def _box_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _alarm_anchor_boxes(
    settings: ObjectDetectionSettings,
    detections: Sequence[ObjectDetection],
) -> Tuple[Tuple[int, int, int, int], ...]:
    target_names = {
        settings.expected_model_names[class_id].strip().lower()
        for class_id in settings.class_ids
        if 0 <= class_id < len(settings.expected_model_names)
    }
    return tuple(
        item.box
        for item in detections
        if item.class_id in settings.class_ids
        and item.class_name.strip().lower() in target_names
    )


def update_alarm_temporal_history(
    settings: ObjectDetectionSettings,
    detections: Sequence[ObjectDetection],
    history: Deque[Tuple[Tuple[int, int, int, int], ...]],
) -> bool:
    anchors = _alarm_anchor_boxes(settings, detections)
    history.append(anchors)
    if not anchors:
        return False
    minimum_hits = int(settings.alarm_min_hits)
    if minimum_hits <= 1:
        return True
    iou_threshold = float(settings.alarm_temporal_iou_threshold)
    if iou_threshold <= 0:
        return sum(bool(frame_boxes) for frame_boxes in history) >= minimum_hits
    for current_box in anchors:
        matching_frames = sum(
            any(_box_iou(current_box, previous_box) >= iou_threshold for previous_box in frame_boxes)
            for frame_boxes in history
        )
        if matching_frames >= minimum_hits:
            return True
    return False


def _crop_detections(frame: np.ndarray, detections: Sequence[ObjectDetection]) -> np.ndarray:
    if not detections:
        return frame.copy()
    height, width = frame.shape[:2]
    x1 = min(item.box[0] for item in detections)
    y1 = min(item.box[1] for item in detections)
    x2 = max(item.box[2] for item in detections)
    y2 = max(item.box[3] for item in detections)
    pad_x = max(8, round((x2 - x1) * 0.12))
    pad_y = max(8, round((y2 - y1) * 0.12))
    return frame[max(0, y1 - pad_y):min(height, y2 + pad_y), max(0, x1 - pad_x):min(width, x2 + pad_x)].copy()


def _write_clip(path: Path, encoded_frames: Sequence[bytes]) -> bool:
    decoded = [cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR) for value in encoded_frames]
    frames = [frame for frame in decoded if frame is not None]
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    handle = tempfile.NamedTemporaryFile(prefix="alarm_clip_", suffix=".mp4", delete=False)
    temporary = Path(handle.name)
    handle.close()
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*str(getattr(config, "ALARM_CLIP_CODEC", "mp4v"))),
        max(1.0, float(getattr(config, "ALARM_CLIP_MAX_FPS", 10.0))),
        (width, height),
    )
    if not writer.isOpened():
        temporary.unlink(missing_ok=True)
        return False
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temporary), str(path))
    return path.is_file() and path.stat().st_size > 0


class ObjectAlarmStore:
    """Write generic detector evidence using the platform-wide event schema."""

    def __init__(self, output_dir: Path = config.ALARM_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self._lock = threading.RLock()

    def create_event(
        self,
        frame: np.ndarray,
        detections: Sequence[ObjectDetection],
        settings: ObjectDetectionSettings,
        source: str,
        frame_index: int,
        event_context: Optional[Dict[str, Any]],
        clip_frames: Sequence[bytes],
    ) -> ObjectAlarmRecord:
        captured_at = datetime.now().astimezone()
        context = dict(event_context or {})
        camera_name = str(context.get("channel_name") or "Camera")
        timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")
        event_id = f"{EVENT_DIR_PREFIX}_{_safe_stem(camera_name)}_{timestamp}_frame_{frame_index:06d}"
        event_dir = self.output_dir / event_id
        with self._lock:
            event_dir.mkdir(parents=True, exist_ok=False)
            annotated_path = event_dir / ALARM_IMAGE_FILENAME
            original_path = event_dir / ORIGINAL_IMAGE_FILENAME
            crop_path = event_dir / RISK_CROP_FILENAME
            metadata_path = event_dir / METADATA_FILENAME
            clip_path = event_dir / ALARM_CLIP_FILENAME
            annotated = _annotate(frame, detections, settings.class_labels)
            _save_jpeg(annotated_path, annotated)
            _save_jpeg(original_path, frame)
            _save_jpeg(crop_path, _crop_detections(annotated, detections))
            clip_ok = _write_clip(clip_path, clip_frames)
            report_url = str(context.get("report_url", "") or "").strip()
            metadata = {
                "event_id": event_id,
                "event_type": settings.algorithm_name,
                "camera": camera_name,
                "source": source,
                "time": captured_at.isoformat(timespec="milliseconds"),
                "frame_index": frame_index,
                "detections": [
                    item.to_dict(settings.class_labels.get(item.class_id, "")) for item in detections
                ],
                "artifacts": {
                    "annotated_image": ALARM_IMAGE_FILENAME,
                    "original_image": ORIGINAL_IMAGE_FILENAME,
                    "risk_crop": RISK_CROP_FILENAME,
                    "metadata": METADATA_FILENAME,
                    "short_video": ALARM_CLIP_FILENAME if clip_ok else None,
                },
                "short_video_status": "completed" if clip_ok else "no_frames",
                "task_id": context.get("task_id"),
                "task_code": str(context.get("task_code", "") or ""),
                "channel_id": context.get("channel_id"),
                "channel_name": str(context.get("channel_name", "") or camera_name),
                "camera_id": str(context.get("camera_id", "") or ""),
                "algorithm_code": settings.algorithm_code,
                "algorithm_name": settings.algorithm_name,
                "algorithm_config": settings.to_dict(),
                "report_url": report_url,
                "report_status": "pending" if report_url else "local_only",
                "alarm_content": (
                    "检测到人员未佩戴安全帽"
                    if settings.alarm_rule == "missing_helmet"
                    else "检测到人员吸烟"
                    if settings.alarm_rule == "person_with_cigarette"
                    else "检测到人员使用手机"
                    if settings.alarm_rule == "person_with_cellphone"
                    else f"检测到{settings.algorithm_name.replace('检测', '')}目标"
                ),
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return ObjectAlarmRecord(
            event_id=event_id,
            event_dir=event_dir,
            annotated_path=annotated_path,
            original_path=original_path,
            crop_path=crop_path,
            metadata_path=metadata_path,
            clip_path=clip_path if clip_ok else None,
        )


def _import_yolov5_runtime() -> Tuple[Any, Any, Any, Any]:
    if not YOLOV5_ROOT.is_dir():
        raise RuntimeError(f"YOLOv5 v6.0 兼容运行库不存在：{YOLOV5_ROOT}")
    root_text = str(YOLOV5_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    return attempt_load, letterbox, non_max_suppression, scale_coords


class YoloV5ModelRuntime:
    def __init__(self, settings: ObjectDetectionSettings) -> None:
        settings.validate()
        attempt_load, letterbox, non_max_suppression, scale_coords = _import_yolov5_runtime()
        requested_device = str(settings.device or "cpu").strip().lower()
        if requested_device != "cpu" and torch.cuda.is_available():
            device_index = int(requested_device) if requested_device.isdigit() else 0
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")
        self.settings = settings
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.model = attempt_load(str(settings.model_path), map_location=self.device)
        self.half = self.device.type == "cuda"
        self.model.half() if self.half else self.model.float()
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.names = tuple(str(value) for value in self.model.names)
        if self.names != tuple(settings.expected_model_names):
            raise RuntimeError(
                f"模型类别不匹配：{settings.model_path.name}，期望 {settings.expected_model_names}，实际 {self.names}"
            )
        self._lock = threading.RLock()
        if self.device.type == "cuda":
            warmup = torch.zeros(1, 3, settings.image_size, settings.image_size, device=self.device)
            warmup = warmup.half() if self.half else warmup.float()
            with torch.inference_mode():
                self.model(warmup)

    def predict(
        self,
        frame: np.ndarray,
        class_ids: Iterable[int],
        confidence_threshold: Optional[float] = None,
    ) -> List[ObjectDetection]:
        selected = sorted({int(value) for value in class_ids})
        if not selected:
            return []
        with self._lock, torch.inference_mode():
            resized = self.letterbox(
                frame,
                new_shape=self.settings.image_size,
                stride=self.stride,
                auto=True,
            )[0]
            tensor_image = resized[:, :, ::-1].transpose(2, 0, 1)
            tensor_image = np.ascontiguousarray(tensor_image)
            tensor = torch.from_numpy(tensor_image).to(self.device)
            tensor = tensor.half() if self.half else tensor.float()
            tensor /= 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)
            prediction = self.model(tensor, augment=False)[0]
            output = self.non_max_suppression(
                prediction,
                conf_thres=float(
                    self.settings.confidence_threshold
                    if confidence_threshold is None
                    else confidence_threshold
                ),
                iou_thres=float(self.settings.iou_threshold),
                classes=selected,
                agnostic=False,
                max_det=300,
            )[0]
            if output is None or len(output) == 0:
                return []
            output[:, :4] = self.scale_coords(tensor.shape[2:], output[:, :4], frame.shape).round()
            detections: List[ObjectDetection] = []
            for x1, y1, x2, y2, confidence, class_id in output.detach().cpu().tolist():
                normalized_id = int(class_id)
                detections.append(
                    ObjectDetection(
                        class_id=normalized_id,
                        class_name=self.names[normalized_id],
                        confidence=float(confidence),
                        box=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
            return detections


class YoloV5ModelPool:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: Dict[Tuple[str, str], YoloV5ModelRuntime] = {}

    def get(self, settings: ObjectDetectionSettings) -> YoloV5ModelRuntime:
        key = (str(settings.model_path.resolve()), str(settings.device))
        with self._lock:
            runtime = self._models.get(key)
            if runtime is None:
                LOGGER.info("[MODEL] loading_yolov5 path=%s device=%s", settings.model_path, settings.device)
                runtime = YoloV5ModelRuntime(settings)
                self._models[key] = runtime
                LOGGER.info("[MODEL] loaded_yolov5 path=%s classes=%s", settings.model_path, runtime.names)
            return runtime

    def loaded_models(self) -> List[str]:
        with self._lock:
            return [key[0] for key in self._models]


MODEL_POOL = YoloV5ModelPool()


class YoloV5ObjectAlgorithm(DetectionAlgorithm):
    def __init__(self, code: str, settings: Optional[ObjectDetectionSettings] = None) -> None:
        self.settings = settings or load_object_settings(code)
        self.settings.validate()
        self.descriptor = AlgorithmDescriptor(
            code=self.settings.algorithm_code,
            name=self.settings.algorithm_name,
            config_version=self.settings.config_version,
            description=self.settings.description,
        )

    def process_image_array(self, image: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        parameter_map = kwargs.get("algorithm_parameters") or {}
        settings = apply_algorithm_parameters(
            self.settings,
            parameter_map.get(self.code, parameter_map),
        )
        primary_confidence = (
            min(
                float(settings.confidence_threshold),
                float(settings.person_confidence_threshold),
                float(settings.bare_head_confidence_threshold),
                float(settings.helmet_confidence_threshold),
            )
            if settings.alarm_rule == "missing_helmet"
            else float(settings.confidence_threshold)
        )
        primary_detections = MODEL_POOL.get(settings).predict(
            image,
            settings.class_ids,
            confidence_threshold=primary_confidence,
        )
        person_detections: Sequence[ObjectDetection] = ()
        if settings.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}:
            person_settings = load_object_settings("safety_helmet")
            person_detections = MODEL_POOL.get(person_settings).predict(
                image,
                (0,),
                confidence_threshold=float(settings.person_confidence_threshold),
            )
        detections = _with_required_person_evidence(
            settings,
            primary_detections,
            person_detections,
        )
        alarm_detections = select_alarm_detections(settings, detections, image.shape)
        display_labels = (
            {0: "人员", 1: "头部", 2: "安全帽"}
            if settings.alarm_rule == "missing_helmet"
            else settings.class_labels
        )
        return {
            "algorithm_code": self.code,
            "algorithm_name": self.name,
            "detections": [item.to_dict(display_labels.get(item.class_id, "")) for item in detections],
            "alarm_detections": [
                item.to_dict(settings.class_labels.get(item.class_id, "")) for item in alarm_detections
            ],
            "annotated_image": _annotate(image, alarm_detections, settings.class_labels),
        }

    def process_video(self, source: str, **kwargs: Any) -> Dict[str, Any]:
        return YoloV5ObjectTaskRunner([self.code]).process_video(source, **kwargs)


class YoloV5ObjectTaskRunner:
    """Run all selected object algorithms while sharing one capture and one inference per model."""

    code = "yolov5_objects"
    name = "YOLOv5目标检测"

    def __init__(self, algorithm_codes: Iterable[str]) -> None:
        ordered = list(dict.fromkeys(str(value or "").strip() for value in algorithm_codes))
        self.settings = [load_object_settings(code) for code in ordered if code in OBJECT_ALGORITHM_DEFINITIONS]
        if not self.settings:
            raise ValueError("未选择 YOLOv5 目标检测算法")
        self.alarm_store = ObjectAlarmStore()

    def config_snapshot(self) -> Dict[str, Any]:
        return {item.algorithm_code: item.to_dict() for item in self.settings}

    def process_image_array(self, image: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        results = []
        for setting in self.settings:
            results.append(YoloV5ObjectAlgorithm(setting.algorithm_code, setting).process_image_array(image, **kwargs))
        return {"algorithms": results}

    def process_video(
        self,
        source: str,
        show: bool = False,
        save_video: bool = False,
        stop_event: Optional[threading.Event] = None,
        event_context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[ObjectAlarmRecord], None]] = None,
        capture_factory: Optional[Callable[[], Tuple[Any, str]]] = None,
        alarm_confirmation: Optional[Callable[..., Dict[str, Any]]] = None,
        algorithm_alarm_intervals: Optional[Dict[str, float]] = None,
        algorithm_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        del show, save_video
        from server import (
            is_camera_index_code_source,
            is_reconnectable_source,
            open_video_capture,
            warmup_video_capture,
        )

        parameter_overrides = algorithm_parameters or {}
        settings = []
        for item in self.settings:
            item_parameters = dict(parameter_overrides.get(item.algorithm_code, {}))
            if item.algorithm_code == "machinery_hazard_supplement":
                machinery_parameters = parameter_overrides.get("machinery_hazard", {})
                if "confidence_threshold" in machinery_parameters:
                    item_parameters["confidence_threshold"] = machinery_parameters[
                        "confidence_threshold"
                    ]
            settings.append(apply_algorithm_parameters(item, item_parameters))
        grouped: Dict[str, List[ObjectDetectionSettings]] = defaultdict(list)
        for setting in settings:
            grouped[setting.model_key].append(setting)
        runtimes = {key: MODEL_POOL.get(values[0]) for key, values in grouped.items()}
        requires_person_evidence = any(
            item.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}
            for item in settings
        )
        person_settings: Optional[ObjectDetectionSettings] = None
        if requires_person_evidence:
            person_settings = load_object_settings("safety_helmet")
            if person_settings.model_key not in runtimes:
                runtimes[person_settings.model_key] = MODEL_POOL.get(person_settings)
        selected_ids_by_model = {
            model_key: {class_id for item in model_settings for class_id in item.class_ids}
            for model_key, model_settings in grouped.items()
        }
        if person_settings is not None:
            selected_ids_by_model.setdefault(person_settings.model_key, set()).add(0)
        confidence_by_model = {
            model_key: min(
                min(
                    float(item.confidence_threshold),
                    float(item.person_confidence_threshold),
                    float(item.bare_head_confidence_threshold),
                    float(item.helmet_confidence_threshold),
                )
                if item.alarm_rule == "missing_helmet"
                else float(item.confidence_threshold)
                for item in model_settings
            )
            for model_key, model_settings in grouped.items()
        }
        if person_settings is not None:
            relation_person_confidence = min(
                float(item.person_confidence_threshold)
                for item in settings
                if item.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}
            )
            confidence_by_model[person_settings.model_key] = min(
                confidence_by_model.get(person_settings.model_key, 1.0),
                relation_person_confidence,
            )
        supplement_setting = next(
            (
                item
                for item in settings
                if item.alarm_rule == "publish_machinery_supplement"
            ),
            None,
        )
        broker_key = supplement_key(source, event_context) if supplement_setting is not None else ""
        detect_interval = min(float(item.detection_interval_seconds) for item in settings)
        interval_overrides = algorithm_alarm_intervals or {}
        cooldowns = {
            item.algorithm_code: float(
                interval_overrides.get(item.algorithm_code, item.alarm_cooldown_seconds)
            )
            for item in settings
        }
        last_alarms = {item.algorithm_code: -10**9 for item in settings}
        first_candidate_at: Dict[str, Optional[float]] = {
            item.algorithm_code: None for item in settings
        }
        alarm_histories = {
            item.algorithm_code: deque(maxlen=max(1, int(item.alarm_window_size)))
            for item in settings
        }
        frame_index = 0
        detected_frames = 0
        alarm_count = 0
        reconnect_attempts = 0
        last_detection_at = 0.0
        last_clip_sample_at = 0.0
        prebuffer: Deque[Tuple[float, bytes]] = deque()
        stop_reason = "end_of_stream"
        should_reconnect = bool(
            config.VIDEO_RECONNECT_ENABLED
            and (is_camera_index_code_source(source) or is_reconnectable_source(source))
        )
        reconnect_interval = float(config.VIDEO_RECONNECT_INTERVAL_SECONDS)
        max_reconnects = int(config.VIDEO_RECONNECT_MAX_ATTEMPTS)
        read_fail_limit = int(config.VIDEO_READ_FAIL_RECONNECT_FRAMES)
        start_at = time.monotonic()
        max_runtime = float(config.STREAM_MAX_RUNTIME_SECONDS)

        while True:
            if stop_event is not None and stop_event.is_set():
                stop_reason = "task_stopped"
                break
            if max_runtime > 0 and time.monotonic() - start_at >= max_runtime:
                stop_reason = "max_runtime_reached"
                break
            try:
                if capture_factory is not None:
                    capture, resolved = capture_factory()
                else:
                    capture, resolved = open_video_capture(source)
            except Exception:
                if not should_reconnect or (max_reconnects > 0 and reconnect_attempts >= max_reconnects):
                    raise
                reconnect_attempts += 1
                if stop_event is not None and stop_event.wait(reconnect_interval):
                    stop_reason = "task_stopped"
                    break
                continue
            if not capture.isOpened():
                capture.release()
                if not should_reconnect or (max_reconnects > 0 and reconnect_attempts >= max_reconnects):
                    raise RuntimeError(f"无法打开视频源：{source}")
                reconnect_attempts += 1
                if stop_event is not None and stop_event.wait(reconnect_interval):
                    stop_reason = "task_stopped"
                    break
                continue
            LOGGER.info("[OBJECT_STREAM] opened=%s algorithms=%s", resolved, [item.algorithm_code for item in settings])
            if not getattr(capture, "is_shared_stream", False):
                warmup_video_capture(_safe_stem(source), capture)
            read_failures = 0
            reconnect_needed = False
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        stop_reason = "task_stopped"
                        break
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if not should_reconnect:
                            stop_reason = "end_of_stream"
                            break
                        read_failures += 1
                        if read_failures >= read_fail_limit:
                            reconnect_needed = True
                            stop_reason = "read_failed"
                            break
                        if stop_event is not None and stop_event.wait(0.05):
                            stop_reason = "task_stopped"
                            break
                        continue
                    read_failures = 0
                    frame_index += 1
                    now = time.monotonic()
                    if now - last_clip_sample_at >= 1.0 / max(1.0, float(config.ALARM_CLIP_MAX_FPS)):
                        ok_jpeg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok_jpeg:
                            prebuffer.append((now, encoded.tobytes()))
                            last_clip_sample_at = now
                            cutoff = now - max(1.0, float(config.ALARM_CLIP_PRE_SECONDS))
                            while prebuffer and prebuffer[0][0] < cutoff:
                                prebuffer.popleft()
                    if last_detection_at > 0 and now - last_detection_at < detect_interval:
                        continue
                    last_detection_at = now
                    detected_frames += 1
                    detections_by_model = {
                        model_key: runtimes[model_key].predict(
                            frame,
                            selected_ids,
                            confidence_threshold=confidence_by_model[model_key],
                        )
                        for model_key, selected_ids in selected_ids_by_model.items()
                    }
                    if supplement_setting is not None:
                        supplement_detections = [
                            mapped
                            for item in detections_by_model[supplement_setting.model_key]
                            if item.class_id in supplement_setting.class_ids
                            and item.confidence >= float(supplement_setting.confidence_threshold)
                            for mapped in (map_supplement_detection(item),)
                            if mapped is not None
                        ]
                        MACHINERY_SUPPLEMENT_BROKER.publish(
                            broker_key,
                            supplement_detections,
                            observed_at=now,
                        )
                    person_detections = (
                        detections_by_model.get(person_settings.model_key, ())
                        if person_settings is not None
                        else ()
                    )
                    for model_key, model_settings in grouped.items():
                        primary_detections = detections_by_model[model_key]
                        for setting in model_settings:
                            if setting.alarm_rule == "publish_machinery_supplement":
                                continue
                            detections = _with_required_person_evidence(
                                setting,
                                primary_detections,
                                person_detections,
                            )
                            matches = select_alarm_detections(setting, detections, frame.shape)
                            temporal_ready = update_alarm_temporal_history(
                                setting,
                                matches,
                                alarm_histories[setting.algorithm_code],
                            )
                            if matches:
                                if first_candidate_at[setting.algorithm_code] is None:
                                    first_candidate_at[setting.algorithm_code] = now
                            else:
                                first_candidate_at[setting.algorithm_code] = None
                            first_seen = first_candidate_at[setting.algorithm_code]
                            duration_ready = (
                                float(setting.alarm_min_duration_seconds) <= 0
                                or (
                                    first_seen is not None
                                    and now - first_seen >= float(setting.alarm_min_duration_seconds)
                                )
                            )
                            if (
                                not matches
                                or not temporal_ready
                                or not duration_ready
                                or now - last_alarms[setting.algorithm_code] < cooldowns[setting.algorithm_code]
                            ):
                                continue
                            alarm_frame = frame
                            alarm_matches = matches
                            if alarm_confirmation is not None:
                                def verify_main(main_frame: np.ndarray) -> Dict[str, Any]:
                                    main_primary_detections = runtimes[model_key].predict(
                                        main_frame,
                                        set(setting.class_ids),
                                        confidence_threshold=confidence_by_model[model_key],
                                    )
                                    main_person_detections: Sequence[ObjectDetection] = ()
                                    if setting.alarm_rule in {"person_with_cigarette", "person_with_cellphone"}:
                                        assert person_settings is not None
                                        main_person_detections = runtimes[
                                            person_settings.model_key
                                        ].predict(
                                            main_frame,
                                            {0},
                                            confidence_threshold=float(
                                                setting.person_confidence_threshold
                                            ),
                                        )
                                    main_detections = _with_required_person_evidence(
                                        setting,
                                        main_primary_detections,
                                        main_person_detections,
                                    )
                                    confirmed_matches = select_alarm_detections(
                                        setting,
                                        main_detections,
                                        main_frame.shape,
                                    )
                                    return {
                                        "confirmed": bool(confirmed_matches),
                                        "frame": main_frame,
                                        "detections": confirmed_matches,
                                    }

                                confirmation = alarm_confirmation(
                                    frame=frame,
                                    candidate_boxes=[item.box for item in matches],
                                    algorithm_code=setting.algorithm_code,
                                    algorithm_name=setting.algorithm_name,
                                    verify=verify_main,
                                )
                                if not confirmation.get("confirmed"):
                                    LOGGER.info(
                                        "[OBJECT_ALARM] algorithm=%s rejected_by_main_stream",
                                        setting.algorithm_code,
                                    )
                                    continue
                                confirmed_frame = confirmation.get("frame")
                                alarm_frame = confirmed_frame if confirmed_frame is not None else frame
                                alarm_matches = confirmation.get("detections") or matches
                            context = dict(event_context or {})
                            effective_config = setting.to_dict()
                            effective_config["alarm_cooldown_seconds"] = cooldowns[
                                setting.algorithm_code
                            ]
                            context.update(
                                algorithm_code=setting.algorithm_code,
                                algorithm_name=setting.algorithm_name,
                                algorithm_config=effective_config,
                            )
                            record = self.alarm_store.create_event(
                                frame=alarm_frame,
                                detections=alarm_matches,
                                settings=setting,
                                source=source,
                                frame_index=frame_index,
                                event_context=context,
                                clip_frames=[value for _timestamp, value in prebuffer],
                            )
                            last_alarms[setting.algorithm_code] = now
                            alarm_histories[setting.algorithm_code].clear()
                            alarm_count += 1
                            LOGGER.info(
                                "[OBJECT_ALARM] algorithm=%s event=%s detections=%s",
                                setting.algorithm_code,
                                record.event_id,
                                len(alarm_matches),
                            )
                            if event_callback is not None:
                                event_callback(record)
                if stop_reason in {"task_stopped", "end_of_stream", "max_runtime_reached"}:
                    break
            finally:
                capture.release()
            if not reconnect_needed:
                break
            if max_reconnects > 0 and reconnect_attempts >= max_reconnects:
                stop_reason = "reconnect_limit_reached"
                break
            reconnect_attempts += 1
            if stop_event is not None and stop_event.wait(reconnect_interval):
                stop_reason = "task_stopped"
                break

        if supplement_setting is not None:
            MACHINERY_SUPPLEMENT_BROKER.clear(broker_key)
        return {
            "source": source,
            "frames": frame_index,
            "detected_frames": detected_frames,
            "alarm_count": alarm_count,
            "stop_reason": stop_reason,
            "reconnect_attempts": reconnect_attempts,
            "algorithm_codes": [item.algorithm_code for item in settings],
        }


def create_object_algorithm(code: str, **kwargs: Any) -> YoloV5ObjectAlgorithm:
    return YoloV5ObjectAlgorithm(code=code, **kwargs)
