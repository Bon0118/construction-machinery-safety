"""工程机械伤害检测服务。

用法示例：
    python server.py --source D:\\test\\image.jpg
    python server.py --source D:\\test\\video.mp4 --save-video
    python server.py --source 0 --show
    python server.py --config-sources
    python server.py --serve
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, time as datetime_time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import config as platform_config
from algorithms.machinery_hazard import defaults as machinery_defaults
from algorithms.machinery_hazard.motion_evidence import BodyVisualEvidence
from algorithms.machinery_supplement import MACHINERY_SUPPLEMENT_BROKER, supplement_key
from alarm_event_schema import (
    ALARM_CLIP_FILENAME,
    ALARM_IMAGE_FILENAME,
    EVENT_DIR_PREFIX,
    METADATA_FILENAME,
    ORIGINAL_IMAGE_FILENAME,
    RISK_CROP_FILENAME,
)


class _ServerConfigView:
    """Resolve algorithm-owned defaults before platform-wide settings."""

    def __getattr__(self, name: str) -> Any:
        if hasattr(machinery_defaults, name):
            return getattr(machinery_defaults, name)
        return getattr(platform_config, name)


config = _ServerConfigView()

# This machine's OpenCV and PyTorch distributions bundle separate Intel OpenMP
# runtimes. The task runner serializes analysis; this flag prevents the second
# runtime from aborting the whole Web process during its first real inference.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def setup_logging() -> logging.Logger:
    log_dir = Path(getattr(config, "LOG_DIR", Path(__file__).resolve().parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / str(getattr(config, "LOG_FILE_NAME", "service.log"))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    logger = logging.getLogger("machinery-hazard")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=int(getattr(config, "LOG_MAX_BYTES", 10 * 1024 * 1024)),
        backupCount=int(getattr(config, "LOG_BACKUP_COUNT", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("log_file=%s", log_path)
    return logger


LOGGER = setup_logging()


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(config.YOLO_CONFIG_DIR))
if getattr(config, "OPENCV_FFMPEG_CAPTURE_OPTIONS", ""):
    # Do not preserve an empty or stale value inherited from a parent process.
    # run_web.bat also sets this before Python starts for Windows OpenCV builds
    # that initialize FFmpeg options during process startup.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(config.OPENCV_FFMPEG_CAPTURE_OPTIONS)

for candidate in (ROOT / "ultralytics", ROOT):
    if (candidate / "ultralytics" / "__init__.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402


Box = Tuple[int, int, int, int]
URL_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
CAMERA_INDEX_SOURCE_PREFIX = "cameraIndexCode:"


@dataclass(frozen=True)
class Detection:
    """一次 YOLO 检测结果。"""

    class_id: int
    class_name: str
    confidence: float
    box: Box
    track_id: Optional[int] = None
    motion_state: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": list(self.box),
        }
        if self.track_id is not None:
            result["track_id"] = self.track_id
        if self.motion_state is not None:
            result["motion_state"] = self.motion_state
        return result


@dataclass(frozen=True)
class BufferGeometry:
    """机械危险缓冲区几何信息。"""

    shape: str
    center: Tuple[int, int]
    axes: Tuple[int, int]
    angle: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        key = "radius" if self.shape == "circle" else "axes"
        value: Any = self.axes[0] if self.shape == "circle" else list(self.axes)
        result = {
            "shape": self.shape,
            "center": list(self.center),
            key: value,
        }
        if self.shape == "ellipse":
            result["angle"] = round(self.angle, 2)
        return result


@dataclass(frozen=True)
class RiskMatch:
    """人员与机械危险缓冲区的一次风险匹配。"""

    person: Detection
    machinery: Detection
    buffer: BufferGeometry
    overlap_ratio: float
    motion_reason: Optional[str] = None
    suppression_reason: Optional[str] = None
    candidate_region: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "person": self.person.to_dict(),
            "machinery": self.machinery.to_dict(),
            "buffer": self.buffer.to_dict(),
            "overlap_ratio": round(self.overlap_ratio, 4),
        }
        if self.motion_reason:
            result["motion_reason"] = self.motion_reason
        if self.candidate_region is not None:
            result["candidate_region"] = self.candidate_region
            result["person_ground_point"] = [(self.person.box[0] + self.person.box[2]) / 2, self.person.box[3]]
        if self.suppression_reason:
            result["suppression_reason"] = self.suppression_reason
        return result


@dataclass(frozen=True)
class MotionAssessment:
    """单台机械当前的车身和机械臂运动状态。"""

    track_id: int
    class_id: int
    class_name: str
    state: str
    body_moving: bool
    arm_working: bool
    body_displacement_ratio: float
    arm_speed_ratio: float
    arm_moving_pixel_ratio: float
    arm_component_ratio: float
    arm_sample_reliable: bool
    body_speed_ratio_per_second: float
    motion_direction: Tuple[float, float]
    body_net_ratio: float = 0.0
    body_direction_consistency: float = 0.0
    body_visual_support: float = 0.0
    body_visual_reliable: bool = False
    box_stable: bool = False

    @property
    def active(self) -> bool:
        return self.body_moving or self.arm_working

    @property
    def reason(self) -> Optional[str]:
        if self.body_moving and self.arm_working:
            return "body_moving+arm_working"
        if self.body_moving:
            return "body_moving"
        if self.arm_working:
            return "arm_working"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state,
            "active": self.active,
            "body_moving": self.body_moving,
            "arm_working": self.arm_working,
            "body_displacement_ratio": round(self.body_displacement_ratio, 5),
            "arm_speed_ratio": round(self.arm_speed_ratio, 5),
            "arm_moving_pixel_ratio": round(self.arm_moving_pixel_ratio, 5),
            "arm_component_ratio": round(self.arm_component_ratio, 5),
            "arm_sample_reliable": self.arm_sample_reliable,
            "body_speed_ratio_per_second": round(self.body_speed_ratio_per_second, 5),
            "motion_direction": [round(value, 5) for value in self.motion_direction],
            "body_net_ratio": round(self.body_net_ratio, 5),
            "body_direction_consistency": round(self.body_direction_consistency, 5),
            "body_visual_support": round(self.body_visual_support, 5),
            "body_visual_reliable": self.body_visual_reliable,
            "box_stable": self.box_stable,
        }


@dataclass(frozen=True)
class RiderAssessment:
    """人员相对某台机械的乘坐状态。"""

    person_track_id: int
    machinery_track_id: int
    state: str
    is_rider: bool
    geometry_match: bool
    containment_ratio: float
    relative_movement: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_track_id": self.person_track_id,
            "machinery_track_id": self.machinery_track_id,
            "state": self.state,
            "is_rider": self.is_rider,
            "geometry_match": self.geometry_match,
            "containment_ratio": round(self.containment_ratio, 5),
            "relative_movement": round(self.relative_movement, 5),
        }


@dataclass
class FrameAnalysis:
    """单帧风险分析结果。"""

    detections: List[Detection]
    persons: List[Detection]
    machinery: List[Detection]
    machinery_buffers: List[Tuple[Detection, BufferGeometry]]
    risks: List[RiskMatch]
    suppressed_risks: List[RiskMatch]
    motion_assessments: List[MotionAssessment]
    rider_assessments: List[RiderAssessment]
    annotated: np.ndarray

    @property
    def has_risk(self) -> bool:
        return bool(self.risks)

    def to_dict(
        self,
        alarm_path: Optional[Path] = None,
        alarm_original_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        return {
            "risk": self.has_risk,
            "alarm_path": str(alarm_path) if alarm_path else None,
            "alarm_original_path": str(alarm_original_path) if alarm_original_path else None,
            "risk_count": len(self.risks),
            "suppressed_risk_count": len(self.suppressed_risks),
            "person_count": len(self.persons),
            "machinery_count": len(self.machinery),
            "detections": [detection.to_dict() for detection in self.detections],
            "risks": [risk.to_dict() for risk in self.risks],
            "suppressed_risks": [risk.to_dict() for risk in self.suppressed_risks],
            "motion_assessments": [assessment.to_dict() for assessment in self.motion_assessments],
            "rider_assessments": [assessment.to_dict() for assessment in self.rider_assessments],
        }


@dataclass
class _MachineryTrack:
    track_id: int
    class_id: int
    class_name: str
    last_box: Box
    smoothed_box: Tuple[float, float, float, float]
    created_at: float
    last_seen_at: float
    position_history: deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    step_history: deque[Tuple[float, float, bool]] = field(default_factory=deque)
    body_state: str = "UNKNOWN"
    body_moving: bool = False
    last_body_motion_at: Optional[float] = None
    body_displacement_ratio: float = 0.0
    body_speed_ratio_per_second: float = 0.0
    motion_direction: Tuple[float, float] = (0.0, 0.0)
    arm_votes: deque[bool] = field(default_factory=deque)
    arm_working: bool = False
    last_arm_motion_at: Optional[float] = None
    arm_speed_ratio: float = 0.0
    arm_moving_pixel_ratio: float = 0.0
    arm_component_ratio: float = 0.0
    arm_sample_reliable: bool = False

    body_visual: BodyVisualEvidence = field(default_factory=BodyVisualEvidence)
    body_net_ratio: float = 0.0
    body_direction_consistency: float = 0.0
    stable_since: float = 0.0
    box_stable: bool = False


@dataclass
class _PersonTrack:
    track_id: int
    last_box: Box
    last_seen_at: float


@dataclass
class _RiderPairTrack:
    person_track_id: int
    machinery_track_id: int
    votes: deque[bool]
    last_relative_position: Optional[Tuple[float, float]] = None
    last_candidate_at: Optional[float] = None
    last_seen_at: float = 0.0
    is_rider: bool = False


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def box_center_distance_ratio(first: Box, second: Box) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    first_diagonal = math.hypot(first[2] - first[0], first[3] - first[1])
    second_diagonal = math.hypot(second[2] - second[0], second[3] - second[1])
    return distance / max(1.0, (first_diagonal + second_diagonal) / 2.0)


class MachineryMotionAnalyzer:
    """按视频源维护机械轨迹，并分析车身位移和挖掘机 ROI 光流。"""

    def __init__(self, source_label: str) -> None:
        self.source_label = source_label
        self._tracks: Dict[int, _MachineryTrack] = {}
        self._next_track_id = 1
        self._person_tracks: Dict[int, _PersonTrack] = {}
        self._next_person_track_id = 1
        self._rider_pairs: Dict[Tuple[int, int], _RiderPairTrack] = {}
        self._last_person_boxes: List[Box] = []
        self._previous_gray: Optional[np.ndarray] = None
        self._last_flow_at = 0.0
        self._camera_transform = np.eye(3, dtype=np.float64)
        self._camera_rebase_required = False
        self._camera_reliable = False

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._person_tracks.clear()
        self._next_person_track_id = 1
        self._rider_pairs.clear()
        self._last_person_boxes = []
        self.reset_flow()

    def reset_flow(self) -> None:
        self._previous_gray = None
        self._last_flow_at = 0.0
        self._camera_transform = np.eye(3, dtype=np.float64)
        self._camera_rebase_required = False
        self._camera_reliable = False
        for track in self._tracks.values():
            self._reset_motion_evidence(track, track.last_seen_at)

    def update_detections(
        self,
        machinery: Sequence[Detection],
        persons: Sequence[Detection],
        timestamp: float,
    ) -> Tuple[List[Detection], List[MotionAssessment]]:
        if self._camera_rebase_required and machinery:
            if getattr(config, "LOG_MACHINERY_MOTION", True):
                LOGGER.warning(
                    "[MOTION] source=%s tracks_reset reason=camera_motion_unreliable",
                    self.source_label,
                )
            self._tracks.clear()
            self._rider_pairs.clear()
            self._camera_rebase_required = False
        self._remove_stale_tracks(timestamp)
        self._last_person_boxes = [person.box for person in persons]

        candidates: List[Tuple[float, int, int]] = []
        min_iou = float(config.MACHINERY_TRACK_IOU_THRESHOLD)
        max_distance = float(config.MACHINERY_TRACK_MAX_CENTER_DISTANCE_RATIO)
        for detection_index, detection in enumerate(machinery):
            for track_id, track in self._tracks.items():
                if detection.class_id != track.class_id:
                    continue
                iou = box_iou(detection.box, track.last_box)
                distance_ratio = box_center_distance_ratio(detection.box, track.last_box)
                if iou < min_iou and distance_ratio > max_distance:
                    continue
                distance_score = max(0.0, 1.0 - distance_ratio / max(max_distance, 1e-6))
                candidates.append((2.0 * iou + distance_score, detection_index, track_id))

        assignments: Dict[int, int] = {}
        used_tracks = set()
        for _score, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in assignments or track_id in used_tracks:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)

        tracked_detections: List[Detection] = []
        assessments: List[MotionAssessment] = []
        for detection_index, detection in enumerate(machinery):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track = self._create_track(detection, timestamp)
            else:
                track = self._tracks[track_id]
                self._update_track_position(track, detection, timestamp)

            assessment = self._assessment(track)
            tracked_detections.append(
                replace(
                    detection,
                    track_id=track.track_id,
                    motion_state=assessment.state,
                )
            )
            assessments.append(assessment)

        return tracked_detections, assessments

    def update_person_detections(
        self,
        persons: Sequence[Detection],
        timestamp: float,
    ) -> List[Detection]:
        self._remove_stale_person_tracks(timestamp)
        min_iou = float(config.PERSON_TRACK_IOU_THRESHOLD)
        max_distance = float(config.PERSON_TRACK_MAX_CENTER_DISTANCE_RATIO)
        candidates: List[Tuple[float, int, int]] = []
        for detection_index, person in enumerate(persons):
            for track_id, track in self._person_tracks.items():
                iou = box_iou(person.box, track.last_box)
                distance_ratio = box_center_distance_ratio(person.box, track.last_box)
                if iou < min_iou and distance_ratio > max_distance:
                    continue
                distance_score = max(0.0, 1.0 - distance_ratio / max(max_distance, 1e-6))
                candidates.append((2.0 * iou + distance_score, detection_index, track_id))

        assignments: Dict[int, int] = {}
        used_tracks = set()
        for _score, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in assignments or track_id in used_tracks:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)

        tracked_persons: List[Detection] = []
        for detection_index, person in enumerate(persons):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_person_track_id
                self._next_person_track_id += 1
                self._person_tracks[track_id] = _PersonTrack(
                    track_id=track_id,
                    last_box=person.box,
                    last_seen_at=timestamp,
                )
            else:
                track = self._person_tracks[track_id]
                track.last_box = person.box
                track.last_seen_at = timestamp
            tracked_persons.append(replace(person, track_id=track_id))
        self._last_person_boxes = [person.box for person in tracked_persons]
        return tracked_persons

    def evaluate_riders(
        self,
        persons: Sequence[Detection],
        machinery: Sequence[Detection],
        timestamp: float,
    ) -> List[RiderAssessment]:
        if not bool(config.RIDER_FILTER_ENABLED):
            return []

        rider_class_ids = set(config.RIDER_FILTER_CLASS_IDS)
        assessments: List[RiderAssessment] = []
        current_pairs = set()
        for person in persons:
            if person.track_id is None:
                continue
            for machine in machinery:
                if machine.track_id is None or machine.class_id not in rider_class_ids:
                    continue
                key = (person.track_id, machine.track_id)
                current_pairs.add(key)
                pair = self._rider_pairs.get(key)
                if pair is None:
                    pair = _RiderPairTrack(
                        person_track_id=person.track_id,
                        machinery_track_id=machine.track_id,
                        votes=deque(maxlen=max(1, int(config.RIDER_CONFIRM_WINDOW_SIZE))),
                    )
                    self._rider_pairs[key] = pair

                geometry_match, containment_ratio, relative_position = self._rider_geometry(person.box, machine)
                relative_movement = 0.0
                if pair.last_relative_position is not None:
                    relative_movement = math.hypot(
                        relative_position[0] - pair.last_relative_position[0],
                        relative_position[1] - pair.last_relative_position[1],
                    )
                stable_relative_position = (
                    pair.last_relative_position is None
                    or relative_movement <= float(config.RIDER_RELATIVE_POSITION_THRESHOLD)
                )
                candidate = geometry_match and stable_relative_position
                pair.votes.append(candidate)
                pair.last_relative_position = relative_position
                pair.last_seen_at = timestamp
                if candidate:
                    pair.last_candidate_at = timestamp

                previous_rider_state = pair.is_rider
                confirm_count = int(config.RIDER_CONFIRM_COUNT)
                if not pair.is_rider and len(pair.votes) >= confirm_count and sum(pair.votes) >= confirm_count:
                    pair.is_rider = True
                elif pair.is_rider:
                    last_candidate_at = pair.last_candidate_at
                    if last_candidate_at is None or timestamp - last_candidate_at >= float(config.RIDER_EXIT_SECONDS):
                        pair.is_rider = False

                if pair.is_rider != previous_rider_state and getattr(config, "LOG_MACHINERY_MOTION", True):
                    LOGGER.info(
                        "[RIDER] source=%s person_track=%s machinery_track=%s %s->%s containment=%.4f relative_movement=%.4f",
                        self.source_label,
                        person.track_id,
                        machine.track_id,
                        "RIDER" if previous_rider_state else "NOT_RIDER",
                        "RIDER" if pair.is_rider else "NOT_RIDER",
                        containment_ratio,
                        relative_movement,
                    )

                state = "RIDER" if pair.is_rider else ("PENDING" if geometry_match else "NOT_RIDER")
                assessments.append(
                    RiderAssessment(
                        person_track_id=person.track_id,
                        machinery_track_id=machine.track_id,
                        state=state,
                        is_rider=pair.is_rider,
                        geometry_match=geometry_match,
                        containment_ratio=containment_ratio,
                        relative_movement=relative_movement,
                    )
                )

        pair_timeout = max(float(config.PERSON_TRACK_LOST_SECONDS), float(config.MACHINERY_TRACK_LOST_SECONDS))
        stale_pairs = [
            key
            for key, pair in self._rider_pairs.items()
            if key not in current_pairs and timestamp - pair.last_seen_at > pair_timeout
        ]
        for key in stale_pairs:
            self._rider_pairs.pop(key, None)
        return assessments

    def _rider_geometry(
        self,
        person_box: Box,
        machine: Detection,
    ) -> Tuple[bool, float, Tuple[float, float]]:
        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = machine.box
        machine_width = max(1, mx2 - mx1)
        machine_height = max(1, my2 - my1)
        person_area = max(1, px2 - px1) * max(1, py2 - py1)
        intersection_width = max(0, min(px2, mx2) - max(px1, mx1))
        intersection_height = max(0, min(py2, my2) - max(py1, my1))
        containment_ratio = intersection_width * intersection_height / person_area

        zone = config.RIDER_CABIN_ZONES.get(machine.class_id, config.RIDER_DEFAULT_CABIN_ZONE)
        zone_x1 = mx1 + float(zone[0]) * machine_width
        zone_y1 = my1 + float(zone[1]) * machine_height
        zone_x2 = mx1 + float(zone[2]) * machine_width
        zone_y2 = my1 + float(zone[3]) * machine_height
        person_center_x = (px1 + px2) / 2.0
        person_center_y = (py1 + py2) / 2.0
        center_in_cabin = zone_x1 <= person_center_x <= zone_x2 and zone_y1 <= person_center_y <= zone_y2
        bottom_ratio = (py2 - my1) / machine_height
        bottom_inside_machine = 0.0 <= bottom_ratio <= float(config.RIDER_PERSON_BOTTOM_MAX_RATIO)
        geometry_match = (
            containment_ratio >= float(config.RIDER_PERSON_CONTAINMENT_THRESHOLD)
            and center_in_cabin
            and bottom_inside_machine
        )
        relative_position = (
            (person_center_x - mx1) / machine_width,
            (py2 - my1) / machine_height,
        )
        return geometry_match, containment_ratio, relative_position

    def observe_frame(self, frame: np.ndarray, timestamp: float) -> None:
        interval = max(0.0, float(config.ARM_FLOW_INTERVAL_SECONDS))
        if self._last_flow_at > 0 and interval > 0 and timestamp - self._last_flow_at < interval:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            self._last_flow_at = timestamp
            return

        elapsed = timestamp - self._last_flow_at
        if elapsed <= 0:
            return

        self._remove_stale_tracks(timestamp)
        if not self._tracks:
            self._previous_gray = gray
            self._last_flow_at = timestamp
            self._camera_transform = np.eye(3, dtype=np.float64)
            return
        excluded_boxes = [track.last_box for track in self._tracks.values()]
        excluded_boxes.extend(self._last_person_boxes)
        aligned_gray, global_motion_reliable, incremental_transform = self._align_current_frame(
            self._previous_gray,
            gray,
            excluded_boxes,
        )
        self._camera_reliable = global_motion_reliable
        if global_motion_reliable:
            incremental_homogeneous = np.vstack((incremental_transform, np.array([0.0, 0.0, 1.0])))
            self._camera_transform = incremental_homogeneous @ self._camera_transform
        elif bool(config.ARM_GLOBAL_MOTION_COMPENSATION):
            self._camera_transform = np.eye(3, dtype=np.float64)
            self._camera_rebase_required = True

        arm_class_ids = set(getattr(config, "ARM_MOTION_CLASS_IDS", set()))
        for track in list(self._tracks.values()):
            track.body_visual.observe(
                self._previous_gray, aligned_gray, track.last_box,
                self._last_person_boxes, timestamp, global_motion_reliable,
            )
            if track.class_id not in arm_class_ids:
                continue
            self._update_arm_flow(
                track,
                self._previous_gray,
                aligned_gray,
                elapsed,
                timestamp,
                global_motion_reliable,
            )

        self._previous_gray = gray
        self._last_flow_at = timestamp

    def assessment_for(self, track_id: Optional[int]) -> Optional[MotionAssessment]:
        if track_id is None:
            return None
        track = self._tracks.get(track_id)
        return self._assessment(track) if track is not None else None

    def _create_track(self, detection: Detection, timestamp: float) -> _MachineryTrack:
        raw_center_x, raw_bottom_y, diagonal = self._bottom_center_and_diagonal(
            tuple(float(value) for value in detection.box)
        )
        motion_center_x, motion_bottom_y = self._camera_stabilized_point(raw_center_x, raw_bottom_y)
        track = _MachineryTrack(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            last_box=detection.box,
            smoothed_box=tuple(float(value) for value in detection.box),
            created_at=timestamp,
            last_seen_at=timestamp,
            stable_since=timestamp,
            arm_votes=deque(maxlen=max(1, int(config.ARM_FLOW_WINDOW_SIZE))),
        )
        self._next_track_id += 1
        track.position_history.append((timestamp, motion_center_x, motion_bottom_y, diagonal))
        self._tracks[track.track_id] = track
        if getattr(config, "LOG_MACHINERY_MOTION", True):
            LOGGER.info(
                "[MOTION] source=%s track=%s class=%s created state=UNKNOWN",
                self.source_label,
                track.track_id,
                track.class_name,
            )
        return track

    def _update_track_position(
        self,
        track: _MachineryTrack,
        detection: Detection,
        timestamp: float,
    ) -> None:
        old_box = track.last_box
        scale_change = max(
            max((detection.box[2]-detection.box[0])/max(1, old_box[2]-old_box[0]),
                (old_box[2]-old_box[0])/max(1, detection.box[2]-detection.box[0])),
            max((detection.box[3]-detection.box[1])/max(1, old_box[3]-old_box[1]),
                (old_box[3]-old_box[1])/max(1, detection.box[3]-detection.box[1])),
        )
        unstable = (scale_change > float(config.MACHINERY_BOX_MAX_SCALE_CHANGE)
                    or box_center_distance_ratio(old_box, detection.box) > float(config.MACHINERY_BOX_MAX_CENTER_STEP))
        if unstable or not self._camera_reliable:
            self._reset_motion_evidence(track, timestamp)
        track.box_stable = not unstable and self._camera_reliable
        alpha = float(config.MACHINERY_BOX_SMOOTHING_ALPHA)
        old_smoothed = track.smoothed_box
        track.smoothed_box = tuple(
            (1.0 - alpha) * old_value + alpha * float(new_value)
            for old_value, new_value in zip(old_smoothed, detection.box)
        )
        track.last_box = detection.box
        track.last_seen_at = timestamp
        track.class_name = detection.class_name

        raw_center_x, raw_bottom_y, diagonal = self._bottom_center_and_diagonal(
            tuple(float(value) for value in detection.box)
        )
        stabilized_x, stabilized_y = self._camera_stabilized_point(raw_center_x, raw_bottom_y)
        center_x = stabilized_x
        bottom_y = stabilized_y
        if track.position_history:
            last_time, last_x, last_y, last_diagonal = track.position_history[-1]
            # Smooth after camera compensation so pan lag is not mistaken for
            # vehicle motion. Raw boxes remain available for jump detection.
            center_x = (1.0-alpha)*last_x + alpha*stabilized_x
            bottom_y = (1.0-alpha)*last_y + alpha*stabilized_y
            delta_x = center_x - last_x
            delta_y = bottom_y - last_y
            step_ratio = math.hypot(delta_x, delta_y) / max(
                1.0,
                (diagonal + last_diagonal) / 2.0,
            )
            moving_step = step_ratio >= float(config.MACHINERY_BODY_STEP_THRESHOLD_RATIO)
            track.step_history.append((timestamp, step_ratio, moving_step))
            if moving_step:
                track.last_body_motion_at = timestamp
                elapsed = max(1e-3, timestamp - last_time)
                current_speed = step_ratio / elapsed
                track.body_speed_ratio_per_second = (
                    0.6 * track.body_speed_ratio_per_second + 0.4 * current_speed
                    if track.body_speed_ratio_per_second > 0
                    else current_speed
                )
                direction_length = math.hypot(delta_x, delta_y)
                if direction_length > 1e-6:
                    current_direction = (delta_x / direction_length, delta_y / direction_length)
                    blended_x = 0.6 * track.motion_direction[0] + 0.4 * current_direction[0]
                    blended_y = 0.6 * track.motion_direction[1] + 0.4 * current_direction[1]
                    blended_length = math.hypot(blended_x, blended_y)
                    if blended_length > 1e-6:
                        track.motion_direction = (blended_x / blended_length, blended_y / blended_length)
            else:
                track.body_speed_ratio_per_second *= 0.7

        track.position_history.append((timestamp, center_x, bottom_y, diagonal))
        window_start = timestamp - float(config.MACHINERY_MOTION_WINDOW_SECONDS)
        while track.position_history and track.position_history[0][0] < window_start:
            track.position_history.popleft()
        while track.step_history and track.step_history[0][0] < window_start:
            track.step_history.popleft()

        positive_steps = [step for _at, step, moving in track.step_history if moving]
        track.body_displacement_ratio = sum(positive_steps)
        _, first_x, first_y, first_diagonal = track.position_history[0]
        net_x, net_y = center_x-first_x, bottom_y-first_y
        track.body_net_ratio = math.hypot(net_x, net_y)/max(1.0, (diagonal+first_diagonal)/2)
        all_travel = sum(step for _at, step, _moving in track.step_history)
        track.body_direction_consistency = min(1.0, track.body_net_ratio/max(all_travel, 1e-9))
        net_length = max(1e-9, math.hypot(net_x, net_y))
        body_direction = (net_x/net_length, net_y/net_length)
        enough_samples = len(track.position_history) >= int(config.MACHINERY_MOTION_MIN_SAMPLES)
        moving_confirmed = (
            enough_samples
            and len(positive_steps) >= int(config.MACHINERY_MOVING_CONFIRM_COUNT)
            and track.body_displacement_ratio >= float(config.MACHINERY_MOVING_THRESHOLD_RATIO)
            and track.box_stable
            and track.body_net_ratio >= float(config.MACHINERY_NET_MOVING_THRESHOLD_RATIO)
            and track.body_direction_consistency >= float(config.MACHINERY_DIRECTION_CONSISTENCY)
            and track.body_visual.confirms(timestamp, body_direction)
        )

        if moving_confirmed and not track.body_moving:
            previous_state = self._state_name(track)
            track.body_moving = True
            track.body_state = "MOVING"
            self._log_transition(track, previous_state)
        elif track.body_moving:
            stop_seconds = float(config.MACHINERY_STOP_CONFIRM_SECONDS)
            if (not track.body_visual.confirms(timestamp, body_direction)
                    or (track.last_body_motion_at is not None and timestamp - track.last_body_motion_at >= stop_seconds)):
                previous_state = self._state_name(track)
                track.body_moving = False
                track.body_state = "STATIC"
                track.body_speed_ratio_per_second = 0.0
                self._log_transition(track, previous_state)
        elif enough_samples:
            track.body_state = "STATIC"

    @staticmethod
    def _reset_motion_evidence(track: _MachineryTrack, timestamp: float) -> None:
        track.position_history.clear()
        track.step_history.clear()
        track.body_visual.clear()
        track.body_moving = track.arm_working = False
        track.body_state = "UNKNOWN"
        track.last_body_motion_at = track.last_arm_motion_at = None
        track.body_displacement_ratio = track.body_net_ratio = 0.0
        track.body_direction_consistency = track.body_speed_ratio_per_second = 0.0
        track.motion_direction = (0.0, 0.0)
        track.arm_votes.clear()
        track.stable_since = timestamp
        track.box_stable = False

    def static_operation_candidate(self, machine: Detection, person: Detection, shape: Tuple[int, ...],
                                   evidence: Sequence[Detection] = ()) -> bool:
        # This is only an entrance to VLM path B; it never confirms an alarm.
        if not config.STATIC_OPERATION_REVIEW_ENABLED or not platform_config.ALARM_REVIEW_ENABLED:
            return False
        track = self._tracks.get(machine.track_id)
        if track is None or machine.class_id not in config.STATIC_OPERATION_REVIEW_CLASS_IDS:
            return False
        if not track.box_stable or track.last_seen_at-track.stable_since < config.STATIC_OPERATION_MIN_TRACK_SECONDS:
            return False
        x1, y1, x2, y2 = machine.box
        area_ratio = (x2-x1)*(y2-y1)/max(1, shape[0]*shape[1])
        if area_ratio > config.STATIC_OPERATION_MAX_FRAME_AREA:
            return False
        # Require a separately detected hook/boom above the person; ordinary
        # static proximity must not bypass the new body-motion filter.
        px = (person.box[0]+person.box[2])/2
        for part in evidence:
            if part.class_id not in config.STATIC_OPERATION_EVIDENCE_CLASS_IDS or part.confidence < config.STATIC_OPERATION_EVIDENCE_CONFIDENCE:
                continue
            if part is machine or part.box == machine.box:
                continue
            ax1, ay1, ax2, ay2 = part.box
            ax, ay = (ax1+ax2)/2, (ay1+ay2)/2
            if (x1 <= ax <= x2 and y1 <= ay <= y2
                    and ax1-(x2-x1)*.15 <= px <= ax2+(x2-x1)*.15
                    and person.box[1] >= ay1):
                return True
        return False

    def _update_arm_flow(
        self,
        track: _MachineryTrack,
        previous_gray: np.ndarray,
        aligned_gray: np.ndarray,
        elapsed: float,
        timestamp: float,
        global_motion_reliable: bool,
    ) -> None:
        if not global_motion_reliable:
            track.arm_sample_reliable = False
            return

        roi_box = self._expanded_roi_box(track.smoothed_box, previous_gray.shape)
        if roi_box is None:
            track.arm_sample_reliable = False
            return
        x1, y1, x2, y2 = roi_box
        previous_roi = previous_gray[y1:y2, x1:x2]
        current_roi = aligned_gray[y1:y2, x1:x2]
        if previous_roi.size == 0 or current_roi.size == 0:
            track.arm_sample_reliable = False
            return

        max_size = max(32, int(config.ARM_FLOW_ROI_MAX_SIZE))
        resize_scale = min(1.0, max_size / max(previous_roi.shape[:2]))
        if resize_scale < 1.0:
            target_size = (
                max(16, int(round(previous_roi.shape[1] * resize_scale))),
                max(16, int(round(previous_roi.shape[0] * resize_scale))),
            )
            previous_roi = cv2.resize(previous_roi, target_size, interpolation=cv2.INTER_AREA)
            current_roi = cv2.resize(current_roi, target_size, interpolation=cv2.INTER_AREA)

        previous_roi = cv2.GaussianBlur(previous_roi, (5, 5), 0)
        current_roi = cv2.GaussianBlur(current_roi, (5, 5), 0)
        brightness_delta = abs(float(previous_roi.mean()) - float(current_roi.mean()))
        if brightness_delta > float(config.ARM_FLOW_MAX_BRIGHTNESS_DELTA):
            track.arm_sample_reliable = False
            return

        flow = cv2.calcOpticalFlowFarneback(
            previous_roi,
            current_roi,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        roi_height, roi_width = previous_roi.shape
        valid_mask = self._make_arm_valid_mask(
            previous_roi,
            roi_box,
            resize_scale,
        )
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < int(config.ARM_FLOW_MIN_VALID_PIXELS):
            track.arm_sample_reliable = False
            return

        normalized_speed = magnitude / max(elapsed, 1e-3) / max(1.0, math.hypot(roi_width, roi_height))
        valid_speeds = normalized_speed[valid_mask]
        top_count = max(1, int(round(valid_speeds.size * 0.05)))
        top_speeds = np.partition(valid_speeds, valid_speeds.size - top_count)[-top_count:]
        speed_score = float(top_speeds.mean())
        moving_mask = valid_mask & (normalized_speed >= float(config.ARM_FLOW_MIN_SPEED_RATIO))
        moving_count = int(np.count_nonzero(moving_mask))
        moving_ratio = moving_count / valid_count

        component_mask = cv2.dilate(moving_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(component_mask, 8)
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
        component_ratio = largest_component / max(1, roi_width * roi_height)

        frame_difference = cv2.absdiff(previous_roi, current_roi)
        corrupted_ratio = float(np.count_nonzero(frame_difference > 80)) / max(1, frame_difference.size)
        if corrupted_ratio > 0.60:
            track.arm_sample_reliable = False
            return

        candidate = (
            speed_score >= float(config.ARM_FLOW_MIN_SPEED_RATIO)
            and moving_ratio >= float(config.ARM_FLOW_MIN_MOVING_PIXEL_RATIO)
            and component_ratio >= float(config.ARM_FLOW_MIN_COMPONENT_RATIO)
        )
        track.arm_speed_ratio = speed_score
        track.arm_moving_pixel_ratio = moving_ratio
        track.arm_component_ratio = component_ratio
        track.arm_sample_reliable = True
        track.arm_votes.append(candidate)
        if candidate:
            track.last_arm_motion_at = timestamp

        confirm_count = int(config.ARM_FLOW_CONFIRM_COUNT)
        if not track.arm_working and len(track.arm_votes) >= confirm_count and sum(track.arm_votes) >= confirm_count:
            previous_state = self._state_name(track)
            track.arm_working = True
            self._log_transition(track, previous_state)
        elif track.arm_working and track.last_arm_motion_at is not None:
            if timestamp - track.last_arm_motion_at >= float(config.ARM_FLOW_STOP_SECONDS):
                previous_state = self._state_name(track)
                track.arm_working = False
                self._log_transition(track, previous_state)

    def _make_arm_valid_mask(
        self,
        previous_roi: np.ndarray,
        roi_box: Box,
        resize_scale: float,
    ) -> np.ndarray:
        roi_height, roi_width = previous_roi.shape
        edges = cv2.Canny(previous_roi, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        valid_mask = edges > 0

        border = max(2, int(round(min(roi_width, roi_height) * 0.02)))
        valid_mask[:border, :] = False
        valid_mask[-border:, :] = False
        valid_mask[:, :border] = False
        valid_mask[:, -border:] = False

        bottom_ratio = float(config.ARM_BODY_EXCLUDE_BOTTOM_RATIO)
        body_y1 = max(0, min(roi_height, int(round(roi_height * (1.0 - bottom_ratio)))))
        body_x1 = int(round(roi_width * 0.20))
        body_x2 = int(round(roi_width * 0.80))
        valid_mask[body_y1:, body_x1:body_x2] = False

        roi_x1, roi_y1, _roi_x2, _roi_y2 = roi_box
        expand_ratio = float(config.ARM_PERSON_MASK_EXPAND_RATIO)
        for person_box in self._last_person_boxes:
            px1, py1, px2, py2 = person_box
            expand_x = (px2 - px1) * expand_ratio
            expand_y = (py2 - py1) * expand_ratio
            local_x1 = int(math.floor((px1 - expand_x - roi_x1) * resize_scale))
            local_y1 = int(math.floor((py1 - expand_y - roi_y1) * resize_scale))
            local_x2 = int(math.ceil((px2 + expand_x - roi_x1) * resize_scale))
            local_y2 = int(math.ceil((py2 + expand_y - roi_y1) * resize_scale))
            local_x1 = max(0, min(roi_width, local_x1))
            local_y1 = max(0, min(roi_height, local_y1))
            local_x2 = max(0, min(roi_width, local_x2))
            local_y2 = max(0, min(roi_height, local_y2))
            if local_x2 > local_x1 and local_y2 > local_y1:
                valid_mask[local_y1:local_y2, local_x1:local_x2] = False
        return valid_mask

    def _align_current_frame(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        excluded_boxes: Sequence[Box],
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        if not bool(config.ARM_GLOBAL_MOTION_COMPENSATION):
            return current_gray, True, identity

        height, width = previous_gray.shape
        scale = min(1.0, 640.0 / max(width, height))
        if scale < 1.0:
            size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
            previous_small = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
            current_small = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        else:
            previous_small = previous_gray
            current_small = current_gray

        mask = np.full(previous_small.shape, 255, dtype=np.uint8)
        for box in excluded_boxes:
            x1, y1, x2, y2 = box
            sx1 = max(0, int(math.floor(x1 * scale)) - 4)
            sy1 = max(0, int(math.floor(y1 * scale)) - 4)
            sx2 = min(mask.shape[1], int(math.ceil(x2 * scale)) + 4)
            sy2 = min(mask.shape[0], int(math.ceil(y2 * scale)) + 4)
            if sx2 > sx1 and sy2 > sy1:
                mask[sy1:sy2, sx1:sx2] = 0

        previous_points = cv2.goodFeaturesToTrack(
            previous_small,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
            blockSize=7,
        )
        min_points = max(4, int(config.ARM_GLOBAL_MOTION_MIN_POINTS))
        if previous_points is None or len(previous_points) < min_points:
            return current_gray, False, identity

        current_points, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous_small,
            current_small,
            previous_points,
            None,
        )
        if current_points is None or status is None:
            return current_gray, False, identity
        valid = status.reshape(-1) == 1
        previous_valid = previous_points.reshape(-1, 2)[valid]
        current_valid = current_points.reshape(-1, 2)[valid]
        if len(previous_valid) < min_points:
            return current_gray, False, identity

        transform, inliers = cv2.estimateAffinePartial2D(
            previous_valid,
            current_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if transform is None or inliers is None or int(inliers.sum()) < min_points:
            return current_gray, False, identity

        inlier_mask = inliers.reshape(-1).astype(bool)
        predicted = previous_valid @ transform[:, :2].T + transform[:, 2]
        residual = np.linalg.norm(predicted-current_valid, axis=1)
        if (float(inlier_mask.mean()) < float(config.CAMERA_MIN_INLIER_RATIO)
                or float(np.median(residual[inlier_mask])) > float(config.CAMERA_MAX_RESIDUAL_PIXELS)):
            return current_gray, False, identity
        # A transform supported by one small background patch is not sufficient.
        points_spread = np.ptp(previous_valid[inlier_mask], axis=0)
        if points_spread[0] < previous_small.shape[1]*0.20 or points_spread[1] < previous_small.shape[0]*0.20:
            return current_gray, False, identity

        transform = transform.astype(np.float64)
        transform[0, 2] /= scale
        transform[1, 2] /= scale
        matrix_scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
        translation_ratio = math.hypot(float(transform[0, 2]), float(transform[1, 2])) / max(
            1.0,
            math.hypot(width, height),
        )
        if not 0.85 <= matrix_scale <= 1.15 or translation_ratio > 0.15:
            return current_gray, False, identity

        inverse_transform = cv2.invertAffineTransform(transform)
        aligned = cv2.warpAffine(
            current_gray,
            inverse_transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, True, transform

    def _expanded_roi_box(
        self,
        box: Tuple[float, float, float, float],
        image_shape: Tuple[int, int],
    ) -> Optional[Box]:
        image_height, image_width = image_shape
        x1, y1, x2, y2 = box
        expand_ratio = float(config.ARM_ROI_EXPAND_RATIO)
        expand_x = (x2 - x1) * expand_ratio
        expand_y = (y2 - y1) * expand_ratio
        roi = (
            max(0, int(math.floor(x1 - expand_x))),
            max(0, int(math.floor(y1 - expand_y))),
            min(image_width, int(math.ceil(x2 + expand_x))),
            min(image_height, int(math.ceil(y2 + expand_y))),
        )
        if roi[2] - roi[0] < 16 or roi[3] - roi[1] < 16:
            return None
        return roi

    def _remove_stale_tracks(self, timestamp: float) -> None:
        timeout = float(config.MACHINERY_TRACK_LOST_SECONDS)
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen_at > timeout
        ]
        for track_id in stale_ids:
            track = self._tracks.pop(track_id)
            if getattr(config, "LOG_MACHINERY_MOTION", True):
                LOGGER.info(
                    "[MOTION] source=%s track=%s class=%s removed reason=lost",
                    self.source_label,
                    track.track_id,
                    track.class_name,
                )

    def _remove_stale_person_tracks(self, timestamp: float) -> None:
        timeout = float(config.PERSON_TRACK_LOST_SECONDS)
        stale_ids = [
            track_id
            for track_id, track in self._person_tracks.items()
            if timestamp - track.last_seen_at > timeout
        ]
        for track_id in stale_ids:
            self._person_tracks.pop(track_id, None)
            stale_pairs = [key for key in self._rider_pairs if key[0] == track_id]
            for key in stale_pairs:
                self._rider_pairs.pop(key, None)

    def _assessment(self, track: _MachineryTrack) -> MotionAssessment:
        return MotionAssessment(
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            state=self._state_name(track),
            body_moving=track.body_moving,
            arm_working=track.arm_working,
            body_displacement_ratio=track.body_displacement_ratio,
            arm_speed_ratio=track.arm_speed_ratio,
            arm_moving_pixel_ratio=track.arm_moving_pixel_ratio,
            arm_component_ratio=track.arm_component_ratio,
            arm_sample_reliable=track.arm_sample_reliable,
            body_speed_ratio_per_second=track.body_speed_ratio_per_second,
            motion_direction=track.motion_direction,
            body_net_ratio=track.body_net_ratio,
            body_direction_consistency=track.body_direction_consistency,
            body_visual_support=track.body_visual.support,
            body_visual_reliable=track.body_visual.reliable,
            box_stable=track.box_stable,
        )

    @staticmethod
    def _bottom_center_and_diagonal(
        box: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, y2, math.hypot(x2 - x1, y2 - y1)

    def _camera_stabilized_point(self, x: float, y: float) -> Tuple[float, float]:
        if not bool(config.ARM_GLOBAL_MOTION_COMPENSATION):
            return x, y
        inverse = np.linalg.inv(self._camera_transform)
        stabilized = inverse @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(float(stabilized[2])) < 1e-9:
            return x, y
        return float(stabilized[0] / stabilized[2]), float(stabilized[1] / stabilized[2])

    @staticmethod
    def _state_name(track: _MachineryTrack) -> str:
        if track.body_moving and track.arm_working:
            return "BODY_MOVING+ARM_WORKING"
        if track.body_moving:
            return "BODY_MOVING"
        if track.arm_working:
            return "ARM_WORKING"
        return track.body_state

    def _log_transition(self, track: _MachineryTrack, previous_state: str) -> None:
        current_state = self._state_name(track)
        if previous_state == current_state or not getattr(config, "LOG_MACHINERY_MOTION", True):
            return
        LOGGER.info(
            "[MOTION] source=%s track=%s class=%s %s->%s body_displacement=%.4f arm_speed=%.4f arm_ratio=%.4f component_ratio=%.4f",
            self.source_label,
            track.track_id,
            track.class_name,
            previous_state,
            current_state,
            track.body_displacement_ratio,
            track.arm_speed_ratio,
            track.arm_moving_pixel_ratio,
            track.arm_component_ratio,
        )


def validate_config() -> None:
    if not Path(config.MODEL_PATH).exists():
        raise FileNotFoundError(f"未找到模型权重：{config.MODEL_PATH}")
    if config.BUFFER_SHAPE not in {"ellipse", "circle"}:
        raise ValueError('BUFFER_SHAPE 只能设置为 "ellipse" 或 "circle"')
    if config.BUFFER_SCALE <= 0:
        raise ValueError("BUFFER_SCALE 必须大于 0")
    if not 0 <= float(config.CONFIDENCE_THRESHOLD) <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD 必须在 0 到 1 之间")
    if not 0 <= float(getattr(config, "PERSON_CONFIDENCE_THRESHOLD", config.CONFIDENCE_THRESHOLD)) <= 1:
        raise ValueError("PERSON_CONFIDENCE_THRESHOLD 必须在 0 到 1 之间")
    if not 0 <= config.OVERLAP_RATIO_THRESHOLD <= 1:
        raise ValueError("OVERLAP_RATIO_THRESHOLD 必须在 0 到 1 之间")
    if config.RISK_CONSECUTIVE_FRAMES < 1:
        raise ValueError("RISK_CONSECUTIVE_FRAMES 必须大于等于 1")
    if config.DETECTION_INTERVAL_SECONDS < 0:
        raise ValueError("DETECTION_INTERVAL_SECONDS 必须大于等于 0")
    if int(getattr(config, "DETECTION_LOG_EVERY_N_FRAMES", 1)) < 1:
        raise ValueError("DETECTION_LOG_EVERY_N_FRAMES must be greater than or equal to 1")
    if float(getattr(config, "ALARM_SAVE_COOLDOWN_SECONDS", 0.0)) < 0:
        raise ValueError("ALARM_SAVE_COOLDOWN_SECONDS must be greater than or equal to 0")
    artifact_flags = (
        bool(config.ALARM_SAVE_ANNOTATED_IMAGE),
        bool(config.ALARM_SAVE_ORIGINAL_IMAGE),
        bool(config.ALARM_SAVE_RISK_CROP),
        bool(config.ALARM_SAVE_METADATA_JSON),
        bool(config.ALARM_SAVE_SHORT_VIDEO),
    )
    if not any(artifact_flags):
        raise ValueError("至少需要开启一种报警证据保存类型")
    if float(config.ALARM_CROP_PADDING_RATIO) < 0:
        raise ValueError("ALARM_CROP_PADDING_RATIO 必须大于等于 0")
    if float(config.ALARM_CLIP_PRE_SECONDS) < 0 or float(config.ALARM_CLIP_POST_SECONDS) < 0:
        raise ValueError("ALARM_CLIP_PRE_SECONDS 和 ALARM_CLIP_POST_SECONDS 必须大于等于 0")
    if float(config.ALARM_CLIP_MAX_FPS) <= 0:
        raise ValueError("ALARM_CLIP_MAX_FPS 必须大于 0")
    if len(str(config.ALARM_CLIP_CODEC)) != 4:
        raise ValueError("ALARM_CLIP_CODEC 必须是 4 个字符")
    if not 1 <= int(config.ALARM_CLIP_JPEG_QUALITY) <= 100:
        raise ValueError("ALARM_CLIP_JPEG_QUALITY 必须在 1 到 100 之间")
    retention_mode = str(config.ALARM_RETENTION_MODE).strip().lower()
    if retention_mode not in {"days", "count", "both", "none"}:
        raise ValueError('ALARM_RETENTION_MODE 只能是 "days"、"count"、"both" 或 "none"')
    if float(config.ALARM_RETENTION_DAYS) < 0:
        raise ValueError("ALARM_RETENTION_DAYS 必须大于等于 0")
    if int(config.ALARM_RETENTION_MAX_EVENTS) < 0:
        raise ValueError("ALARM_RETENTION_MAX_EVENTS 必须大于等于 0")
    if int(config.ALARM_TITLE_FONT_SIZE) < 1 or int(config.ALARM_INFO_FONT_SIZE) < 1:
        raise ValueError("报警文字字号必须大于 0")
    if int(config.ALARM_INFO_BAR_HEIGHT) < 1:
        raise ValueError("ALARM_INFO_BAR_HEIGHT 必须大于 0")
    if not 0 <= float(config.ALARM_INFO_BAR_ALPHA) <= 1:
        raise ValueError("ALARM_INFO_BAR_ALPHA 必须在 0 到 1 之间")
    if int(config.ALARM_TITLE_TOP_MARGIN) < 0:
        raise ValueError("ALARM_TITLE_TOP_MARGIN 必须大于等于 0")
    if not 0 <= float(config.ALARM_RISK_BOX_ALPHA) <= 1:
        raise ValueError("ALARM_RISK_BOX_ALPHA 必须在 0 到 1 之间")
    if int(config.ALARM_RISK_BOX_BORDER_THICKNESS) < 1:
        raise ValueError("ALARM_RISK_BOX_BORDER_THICKNESS 必须大于等于 1")
    if not 0 <= float(config.ALARM_DRAW_DEDUP_IOU_THRESHOLD) <= 1:
        raise ValueError("ALARM_DRAW_DEDUP_IOU_THRESHOLD 必须在 0 到 1 之间")
    if not 0 <= float(config.ALARM_DRAW_DEDUP_CONTAINMENT_THRESHOLD) <= 1:
        raise ValueError("ALARM_DRAW_DEDUP_CONTAINMENT_THRESHOLD 必须在 0 到 1 之间")
    if int(getattr(config, "LOG_MAX_BYTES", 1)) < 1:
        raise ValueError("LOG_MAX_BYTES must be greater than or equal to 1")
    if int(getattr(config, "LOG_BACKUP_COUNT", 0)) < 0:
        raise ValueError("LOG_BACKUP_COUNT must be greater than or equal to 0")
    if not 0 <= float(config.MACHINERY_TRACK_IOU_THRESHOLD) <= 1:
        raise ValueError("MACHINERY_TRACK_IOU_THRESHOLD 必须在 0 到 1 之间")
    if float(config.MACHINERY_TRACK_MAX_CENTER_DISTANCE_RATIO) <= 0:
        raise ValueError("MACHINERY_TRACK_MAX_CENTER_DISTANCE_RATIO 必须大于 0")
    if float(config.MACHINERY_TRACK_LOST_SECONDS) <= 0:
        raise ValueError("MACHINERY_TRACK_LOST_SECONDS 必须大于 0")
    if not 0 < float(config.MACHINERY_BOX_SMOOTHING_ALPHA) <= 1:
        raise ValueError("MACHINERY_BOX_SMOOTHING_ALPHA 必须在 0 到 1 之间")
    if float(config.MACHINERY_MOTION_WINDOW_SECONDS) <= 0:
        raise ValueError("MACHINERY_MOTION_WINDOW_SECONDS 必须大于 0")
    if int(config.MACHINERY_MOTION_MIN_SAMPLES) < 2:
        raise ValueError("MACHINERY_MOTION_MIN_SAMPLES 必须大于等于 2")
    if float(config.MACHINERY_BODY_STEP_THRESHOLD_RATIO) < 0:
        raise ValueError("MACHINERY_BODY_STEP_THRESHOLD_RATIO 必须大于等于 0")
    if float(config.MACHINERY_MOVING_THRESHOLD_RATIO) < 0:
        raise ValueError("MACHINERY_MOVING_THRESHOLD_RATIO 必须大于等于 0")
    if int(config.MACHINERY_MOVING_CONFIRM_COUNT) < 1:
        raise ValueError("MACHINERY_MOVING_CONFIRM_COUNT 必须大于等于 1")
    if float(config.MACHINERY_STOP_CONFIRM_SECONDS) < 0:
        raise ValueError("MACHINERY_STOP_CONFIRM_SECONDS 必须大于等于 0")
    if float(config.ARM_FLOW_INTERVAL_SECONDS) < 0:
        raise ValueError("ARM_FLOW_INTERVAL_SECONDS 必须大于等于 0")
    if float(config.ARM_ROI_EXPAND_RATIO) < 0:
        raise ValueError("ARM_ROI_EXPAND_RATIO 必须大于等于 0")
    if not 0 <= float(config.ARM_BODY_EXCLUDE_BOTTOM_RATIO) < 1:
        raise ValueError("ARM_BODY_EXCLUDE_BOTTOM_RATIO 必须在 0 到 1 之间")
    if float(config.ARM_PERSON_MASK_EXPAND_RATIO) < 0:
        raise ValueError("ARM_PERSON_MASK_EXPAND_RATIO 必须大于等于 0")
    if int(config.ARM_FLOW_ROI_MAX_SIZE) < 32:
        raise ValueError("ARM_FLOW_ROI_MAX_SIZE 必须大于等于 32")
    if int(config.ARM_FLOW_MIN_VALID_PIXELS) < 1:
        raise ValueError("ARM_FLOW_MIN_VALID_PIXELS 必须大于等于 1")
    for name in (
        "ARM_FLOW_MIN_SPEED_RATIO",
        "ARM_FLOW_MIN_MOVING_PIXEL_RATIO",
        "ARM_FLOW_MIN_COMPONENT_RATIO",
    ):
        if float(getattr(config, name)) < 0:
            raise ValueError(f"{name} 必须大于等于 0")
    if int(config.ARM_FLOW_WINDOW_SIZE) < 1:
        raise ValueError("ARM_FLOW_WINDOW_SIZE 必须大于等于 1")
    if not 1 <= int(config.ARM_FLOW_CONFIRM_COUNT) <= int(config.ARM_FLOW_WINDOW_SIZE):
        raise ValueError("ARM_FLOW_CONFIRM_COUNT 必须在 1 到 ARM_FLOW_WINDOW_SIZE 之间")
    if float(config.ARM_FLOW_STOP_SECONDS) < 0:
        raise ValueError("ARM_FLOW_STOP_SECONDS 必须大于等于 0")
    if float(config.ARM_FLOW_MAX_BRIGHTNESS_DELTA) < 0:
        raise ValueError("ARM_FLOW_MAX_BRIGHTNESS_DELTA 必须大于等于 0")
    if int(config.ARM_GLOBAL_MOTION_MIN_POINTS) < 4:
        raise ValueError("ARM_GLOBAL_MOTION_MIN_POINTS 必须大于等于 4")
    if not 0 <= float(config.RIDER_PERSON_CONTAINMENT_THRESHOLD) <= 1:
        raise ValueError("RIDER_PERSON_CONTAINMENT_THRESHOLD 必须在 0 到 1 之间")
    if not 0 <= float(config.RIDER_PERSON_BOTTOM_MAX_RATIO) <= 1:
        raise ValueError("RIDER_PERSON_BOTTOM_MAX_RATIO 必须在 0 到 1 之间")
    if float(config.RIDER_RELATIVE_POSITION_THRESHOLD) < 0:
        raise ValueError("RIDER_RELATIVE_POSITION_THRESHOLD 必须大于等于 0")
    if int(config.RIDER_CONFIRM_WINDOW_SIZE) < 1:
        raise ValueError("RIDER_CONFIRM_WINDOW_SIZE 必须大于等于 1")
    if not 1 <= int(config.RIDER_CONFIRM_COUNT) <= int(config.RIDER_CONFIRM_WINDOW_SIZE):
        raise ValueError("RIDER_CONFIRM_COUNT 必须在 1 到 RIDER_CONFIRM_WINDOW_SIZE 之间")
    if float(config.RIDER_EXIT_SECONDS) < 0:
        raise ValueError("RIDER_EXIT_SECONDS 必须大于等于 0")
    if not 0 <= float(config.PERSON_TRACK_IOU_THRESHOLD) <= 1:
        raise ValueError("PERSON_TRACK_IOU_THRESHOLD 必须在 0 到 1 之间")
    if float(config.PERSON_TRACK_MAX_CENTER_DISTANCE_RATIO) <= 0:
        raise ValueError("PERSON_TRACK_MAX_CENTER_DISTANCE_RATIO 必须大于 0")
    if float(config.PERSON_TRACK_LOST_SECONDS) <= 0:
        raise ValueError("PERSON_TRACK_LOST_SECONDS 必须大于 0")
    for class_id, zone in config.RIDER_CABIN_ZONES.items():
        if len(zone) != 4 or not all(0 <= float(value) <= 1 for value in zone):
            raise ValueError(f"RIDER_CABIN_ZONES[{class_id}] 必须包含 4 个 0 到 1 之间的值")
        if float(zone[0]) >= float(zone[2]) or float(zone[1]) >= float(zone[3]):
            raise ValueError(f"RIDER_CABIN_ZONES[{class_id}] 坐标顺序无效")
    default_zone = config.RIDER_DEFAULT_CABIN_ZONE
    if len(default_zone) != 4 or not all(0 <= float(value) <= 1 for value in default_zone):
        raise ValueError("RIDER_DEFAULT_CABIN_ZONE 必须包含 4 个 0 到 1 之间的值")
    if float(default_zone[0]) >= float(default_zone[2]) or float(default_zone[1]) >= float(default_zone[3]):
        raise ValueError("RIDER_DEFAULT_CABIN_ZONE 坐标顺序无效")
    for class_id, scales in config.RISK_ZONE_CLASS_SCALES.items():
        if len(scales) != 2 or any(float(value) <= 0 for value in scales):
            raise ValueError(f"RISK_ZONE_CLASS_SCALES[{class_id}] 必须包含 2 个大于 0 的值")
    if not 0 <= float(config.RISK_ZONE_CENTER_Y_RATIO) <= 1:
        raise ValueError("RISK_ZONE_CENTER_Y_RATIO 必须在 0 到 1 之间")
    if float(config.RISK_ZONE_REACTION_TIME_SECONDS) < 0:
        raise ValueError("RISK_ZONE_REACTION_TIME_SECONDS 必须大于等于 0")
    if float(config.RISK_ZONE_MAX_FORWARD_EXTENSION_RATIO) < 0:
        raise ValueError("RISK_ZONE_MAX_FORWARD_EXTENSION_RATIO 必须大于等于 0")
    if not 0 < float(config.PERSON_RISK_LOWER_BODY_RATIO) <= 1:
        raise ValueError("PERSON_RISK_LOWER_BODY_RATIO 必须在 0 到 1 之间")
    if config.STREAM_MAX_RUNTIME_SECONDS < 0:
        raise ValueError("STREAM_MAX_RUNTIME_SECONDS 必须大于等于 0")
    if int(getattr(config, "STREAM_WARMUP_FRAMES", 0)) < 0:
        raise ValueError("STREAM_WARMUP_FRAMES must be greater than or equal to 0")
    if config.VIDEO_STREAM_WORKERS < 0:
        raise ValueError("VIDEO_STREAM_WORKERS 必须大于等于 0")
    if config.VIDEO_RECONNECT_INTERVAL_SECONDS < 0:
        raise ValueError("VIDEO_RECONNECT_INTERVAL_SECONDS 必须大于等于 0")
    if config.VIDEO_RECONNECT_MAX_ATTEMPTS < 0:
        raise ValueError("VIDEO_RECONNECT_MAX_ATTEMPTS 必须大于等于 0")
    if config.VIDEO_READ_FAIL_RECONNECT_FRAMES < 1:
        raise ValueError("VIDEO_READ_FAIL_RECONNECT_FRAMES 必须大于等于 1")
    for time_range in config.DETECTION_TIME_RANGES:
        if len(time_range) != 2:
            raise ValueError("DETECTION_TIME_RANGES 中每个时间段都必须是 (开始时间, 结束时间)")
        parse_time_value(time_range[0])
        parse_time_value(time_range[1])
    if not config.PERSON_CLASS_IDS:
        raise ValueError("PERSON_CLASS_IDS 不能为空")
    if not config.MACHINERY_CLASS_IDS:
        raise ValueError("MACHINERY_CLASS_IDS 不能为空")


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def is_url_source(source: str) -> bool:
    return bool(URL_SCHEME_PATTERN.match(source.strip()))


def is_reconnectable_source(source: str) -> bool:
    return is_camera_source(source) or is_url_source(source)


def normalize_config_source(source: Any) -> str:
    text = str(source).strip().strip('"')
    if not text:
        return ""
    if is_camera_source(text) or is_url_source(text):
        return text

    path = Path(text)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return str(path)


def extract_rtsp_url(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text if text.lower().startswith("rtsp://") else None

    if isinstance(value, dict):
        preferred_keys = (
            "url",
            "rtsp",
            "rtspUrl",
            "rtspURL",
            "streamUrl",
            "streamURL",
            "previewUrl",
            "previewURL",
            "data",
        )
        for key in preferred_keys:
            if key in value:
                found = extract_rtsp_url(value[key])
                if found:
                    return found
        for nested in value.values():
            found = extract_rtsp_url(nested)
            if found:
                return found

    if isinstance(value, list):
        for item in value:
            found = extract_rtsp_url(item)
            if found:
                return found

    return None


def fetch_rtsp_url_by_camera_index_code(
    camera_index_code: str,
    timeout_seconds: Optional[float] = None,
    stream_type: Optional[str] = None,
) -> str:
    api_url = str(getattr(config, "VIDEO_STREAM_API_URL", "")).strip()
    if not api_url:
        raise ValueError("VIDEO_STREAM_API_URL is empty")

    payload = dict(getattr(config, "VIDEO_STREAM_API_PAYLOAD", {}))
    payload["cameraIndexCode"] = camera_index_code
    if stream_type is not None:
        normalized_stream_type = str(stream_type).strip()
        if normalized_stream_type not in {"0", "1"}:
            raise ValueError("streamType must be 0 (main) or 1 (sub)")
        payload["streamType"] = normalized_stream_type
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    configured_timeout = float(getattr(config, "VIDEO_STREAM_API_TIMEOUT_SECONDS", 10.0))
    timeout = configured_timeout
    if timeout_seconds is not None:
        timeout = min(configured_timeout, max(0.1, float(timeout_seconds)))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to resolve cameraIndexCode {camera_index_code}: {exc}") from exc

    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"cameraIndexCode {camera_index_code} response is not JSON: {response_text[:300]}"
        ) from exc

    rtsp_url = extract_rtsp_url(response_json)
    if not rtsp_url:
        raise RuntimeError(
            f"cameraIndexCode {camera_index_code} response does not contain an RTSP URL: {response_text[:500]}"
        )
    return rtsp_url


def make_camera_index_code_source(camera_index_code: str) -> str:
    return f"{CAMERA_INDEX_SOURCE_PREFIX}{camera_index_code}"


def is_camera_index_code_source(source: str) -> bool:
    return str(source).startswith(CAMERA_INDEX_SOURCE_PREFIX)


def camera_index_code_from_source(source: str) -> str:
    return str(source).split(":", 1)[1].split("?", 1)[0].strip()


def camera_stream_type_from_source(source: str) -> Optional[str]:
    text = str(source)
    if "?" not in text:
        return None
    query = text.split("?", 1)[1]
    for item in query.split("&"):
        key, separator, value = item.partition("=")
        if separator and key.strip().lower() == "streamtype":
            normalized = value.strip()
            if normalized not in {"0", "1"}:
                raise ValueError("streamType must be 0 (main) or 1 (sub)")
            return normalized
    return None


def parse_time_value(value: Any) -> datetime_time:
    if isinstance(value, datetime_time):
        return value

    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"检测时间格式错误：{value}，应为 HH:MM 或 HH:MM:SS")


def in_detection_time(now: Optional[datetime] = None) -> bool:
    ranges = getattr(config, "DETECTION_TIME_RANGES", [])
    if not ranges:
        return True

    current = (now or datetime.now()).time()
    for start_value, end_value in ranges:
        start = parse_time_value(start_value)
        end = parse_time_value(end_value)
        if start <= end and start <= current <= end:
            return True
        if start > end and (current >= start or current <= end):
            return True
    return False


def resolve_capture_source(
    source: str,
    timeout_seconds: Optional[float] = None,
) -> Tuple[Any, str]:
    if is_camera_index_code_source(source):
        camera_index_code = camera_index_code_from_source(source)
        stream_type = camera_stream_type_from_source(source)
        rtsp_url = fetch_rtsp_url_by_camera_index_code(
            camera_index_code,
            timeout_seconds=timeout_seconds,
            stream_type=stream_type,
        )
        return rtsp_url, rtsp_url
    if is_camera_source(source):
        return int(source), source
    return source, source


def open_video_capture(source: str) -> Tuple[cv2.VideoCapture, str]:
    capture_source, resolved_source = resolve_capture_source(source)
    cap = cv2.VideoCapture()

    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(config.VIDEO_OPEN_TIMEOUT_MILLISECONDS))
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(config.VIDEO_READ_TIMEOUT_MILLISECONDS))

    cap.open(capture_source)
    return cap, resolved_source


def warmup_video_capture(source_label: str, cap: cv2.VideoCapture) -> None:
    warmup_frames = max(0, int(getattr(config, "STREAM_WARMUP_FRAMES", 0)))
    if warmup_frames <= 0:
        return

    ok_count = 0
    fail_count = 0
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    LOGGER.info(
        "[STREAM] source=%s warmup_frames=%s success=%s failed=%s",
        source_label,
        warmup_frames,
        ok_count,
        fail_count,
    )


def show_frame_or_disable(window_name: str, frame: np.ndarray) -> Tuple[bool, Optional[int]]:
    try:
        cv2.imshow(window_name, frame)
        return True, cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        LOGGER.warning("OpenCV window display is unavailable; continuing without --show. Detail: %s", exc)
        return False, None


def destroy_windows_safely() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def source_type(source: str) -> str:
    if is_camera_index_code_source(source) or is_camera_source(source) or is_url_source(source):
        return "video"

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件：{path}")

    suffix = path.suffix.lower()
    if suffix in config.IMAGE_EXTENSIONS:
        return "image"
    if suffix in config.VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"无法判断输入类型，请检查文件后缀：{suffix or source}")


def read_image_with_unicode_path(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片读取失败：{path}")
    return image


def decode_image_bytes(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片字节解码失败")
    return image


def save_image_with_unicode_path(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or config.ALARM_IMAGE_EXT
    params: List[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)]

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def clip_box(box: Sequence[float], image_shape: Tuple[int, int, int]) -> Optional[Box]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def safe_stem(value: str) -> str:
    stem = f"camera_{value}" if is_camera_source(value) else Path(value).stem
    stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", stem).strip("_")
    return stem or "source"


def camera_display_name(source: str) -> str:
    mapping = dict(getattr(config, "CAMERA_DISPLAY_NAMES", {}))
    source_text = str(source).strip()
    candidates = [source_text]
    if is_camera_index_code_source(source_text):
        candidates.insert(0, camera_index_code_from_source(source_text))
    for candidate in candidates:
        if candidate in mapping:
            return str(mapping[candidate])
    return str(getattr(config, "DEFAULT_CAMERA_DISPLAY_NAME", "Camera"))


@dataclass
class AlarmEventRecord:
    event_id: str
    event_dir: Path
    annotated_path: Optional[Path]
    original_path: Optional[Path]
    crop_path: Optional[Path]
    metadata_path: Optional[Path]
    clip_path: Optional[Path] = None

    @property
    def primary_path(self) -> Path:
        return self.annotated_path or self.original_path or self.crop_path or self.event_dir


class AlarmImageStore:
    """按事件保存报警证据，并按天数或事件数清理。"""

    def __init__(self, output_dir: Path = config.ALARM_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def create_event(
        self,
        annotated_image: np.ndarray,
        original_image: np.ndarray,
        analysis: FrameAnalysis,
        source_name: str,
        camera_name: str,
        captured_at: datetime,
        frame_index: Optional[int] = None,
        allow_short_video: bool = False,
        event_context: Optional[Dict[str, Any]] = None,
    ) -> AlarmEventRecord:
        timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")
        frame_part = f"_frame_{frame_index:06d}" if frame_index is not None else ""
        event_id = f"{EVENT_DIR_PREFIX}_{safe_stem(camera_name)}_{timestamp}{frame_part}"
        event_dir = self.output_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=False)

        annotated_path = event_dir / ALARM_IMAGE_FILENAME if bool(config.ALARM_SAVE_ANNOTATED_IMAGE) else None
        original_path = event_dir / ORIGINAL_IMAGE_FILENAME if bool(config.ALARM_SAVE_ORIGINAL_IMAGE) else None
        crop_path = event_dir / RISK_CROP_FILENAME if bool(config.ALARM_SAVE_RISK_CROP) else None
        metadata_path = event_dir / METADATA_FILENAME if bool(config.ALARM_SAVE_METADATA_JSON) else None

        if annotated_path is not None:
            save_image_with_unicode_path(annotated_path, annotated_image)
        if original_path is not None:
            save_image_with_unicode_path(original_path, original_image)
        if crop_path is not None:
            crop = self._risk_crop(annotated_image, analysis.risks)
            save_image_with_unicode_path(crop_path, crop)

        clip_enabled = allow_short_video and bool(config.ALARM_SAVE_SHORT_VIDEO)
        record = AlarmEventRecord(
            event_id=event_id,
            event_dir=event_dir,
            annotated_path=annotated_path,
            original_path=original_path,
            crop_path=crop_path,
            metadata_path=metadata_path,
        )
        if metadata_path is not None:
            primary_risk = max(analysis.risks, key=lambda item: item.overlap_ratio)
            context = dict(event_context or {})
            report_url = str(context.get("report_url", "") or "").strip()
            metadata = {
                "event_id": event_id,
                "event_type": str(context.get("algorithm_name") or "机械伤害预警"),
                "camera": camera_name,
                "source": source_name,
                "time": captured_at.astimezone().isoformat(timespec="milliseconds"),
                "frame_index": frame_index,
                "risk_count": len(analysis.risks),
                "motion_reasons": sorted({risk.motion_reason or "single_frame" for risk in analysis.risks}),
                "scene_mode": ("crane_boom_person" if any(
                    risk.motion_reason == "crane_boom_area_review" for risk in analysis.risks
                ) else "legacy_machinery"),
                "crane_region_version": "v2-associated-boom-footpoint",
                "motion_filter_version": config.MOTION_FILTER_VERSION,
                "motion_assessments": [item.to_dict() for item in analysis.motion_assessments],
                "operation_evidence": [item.to_dict() for item in analysis.detections
                                       if item.class_id in config.STATIC_OPERATION_EVIDENCE_CLASS_IDS],
                "primary_person_track_id": primary_risk.person.track_id,
                "primary_machinery_track_id": primary_risk.machinery.track_id,
                "primary_person_confidence": round(primary_risk.person.confidence, 5),
                "primary_machinery_class": primary_risk.machinery.class_name,
                "primary_machinery_confidence": round(primary_risk.machinery.confidence, 5),
                "risks": [risk.to_dict() for risk in analysis.risks],
                "artifacts": {
                    "annotated_image": annotated_path.name if annotated_path else None,
                    "original_image": original_path.name if original_path else None,
                    "risk_crop": crop_path.name if crop_path else None,
                    "metadata": metadata_path.name,
                    "short_video": ALARM_CLIP_FILENAME if clip_enabled else None,
                },
                "short_video_status": "pending" if clip_enabled else "disabled",
                "task_id": context.get("task_id"),
                "task_code": str(context.get("task_code", "") or ""),
                "channel_id": context.get("channel_id"),
                "channel_name": str(context.get("channel_name", "") or camera_name),
                "camera_id": str(context.get("camera_id", "") or ""),
                "algorithm_code": str(context.get("algorithm_code", "") or ""),
                "algorithm_name": str(context.get("algorithm_name", "") or "机械伤害预警"),
                "algorithm_config": dict(context.get("algorithm_config") or {}),
                "report_url": report_url,
                "report_status": "pending" if report_url else "local_only",
                "alarm_content": str(
                    ("吊装下禁止站人" if any(risk.motion_reason == "crane_boom_area_review" for risk in analysis.risks)
                     else context.get("alarm_content", "") or "检测到人员进入工程机械危险区域")
                ),
            }
            self._write_metadata(metadata_path, metadata)

        self.enforce_retention()
        return record

    def complete_short_video(
        self,
        record: AlarmEventRecord,
        encoded_frames: Sequence[Tuple[float, bytes]],
    ) -> Optional[Path]:
        if not encoded_frames or not record.event_dir.exists():
            self._update_metadata(record, {"short_video_status": "no_frames"})
            return None

        first_frame = cv2.imdecode(np.frombuffer(encoded_frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if first_frame is None:
            self._update_metadata(record, {"short_video_status": "decode_failed"})
            return None
        height, width = first_frame.shape[:2]
        duration = max(0.0, encoded_frames[-1][0] - encoded_frames[0][0])
        measured_fps = (len(encoded_frames) - 1) / duration if duration > 0 and len(encoded_frames) > 1 else 1.0
        fps = max(1.0, min(float(config.ALARM_CLIP_MAX_FPS), measured_fps))
        clip_path = record.event_dir / ALARM_CLIP_FILENAME
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*str(config.ALARM_CLIP_CODEC)),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            self._update_metadata(record, {"short_video_status": "writer_open_failed"})
            LOGGER.error("[ALARM_CLIP] event=%s writer_open_failed path=%s", record.event_id, clip_path)
            return None
        written = 0
        try:
            for _timestamp, encoded in encoded_frames:
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
        finally:
            writer.release()

        if written == 0:
            clip_path.unlink(missing_ok=True)
            self._update_metadata(record, {"short_video_status": "no_decodable_frames"})
            return None
        record.clip_path = clip_path
        self._update_metadata(
            record,
            {
                "short_video_status": "complete",
                "short_video_frames": written,
                "short_video_fps": round(fps, 3),
                "short_video_duration_seconds": round(duration, 3),
            },
        )
        self.enforce_retention()
        LOGGER.info(
            "[ALARM_CLIP] event=%s frames=%s fps=%.2f path=%s",
            record.event_id,
            written,
            fps,
            clip_path,
        )
        return clip_path

    def enforce_retention(self) -> None:
        mode = str(config.ALARM_RETENTION_MODE).strip().lower()
        if mode == "none" or not self.output_dir.exists():
            return
        event_prefix = f"{EVENT_DIR_PREFIX}_"
        # Other algorithm workers can prune the same output directory. Snapshot
        # timestamps once and tolerate entries removed during enumeration.
        modified_at = {}
        for path in self.output_dir.iterdir():
            if not path.name.startswith(event_prefix) or not path.is_dir():
                continue
            try:
                modified_at[path] = path.stat().st_mtime
            except FileNotFoundError:
                continue
        events = sorted(modified_at, key=lambda item: (modified_at[item], item.name))
        to_delete = set()
        if mode in {"days", "both"}:
            days = float(config.ALARM_RETENTION_DAYS)
            if days > 0:
                cutoff = time.time() - days * 86400.0
                to_delete.update(path for path in events if modified_at[path] < cutoff)
        remaining = [path for path in events if path not in to_delete]
        if mode in {"count", "both"}:
            limit = int(config.ALARM_RETENTION_MAX_EVENTS)
            if limit > 0 and len(remaining) > limit:
                to_delete.update(remaining[:-limit])
        for event_dir in sorted(to_delete, key=lambda item: (modified_at[item], item.name)):
            self._remove_event_dir(event_dir)

    def _remove_event_dir(self, event_dir: Path) -> None:
        output_root = self.output_dir.resolve()
        resolved = event_dir.resolve()
        event_prefix = f"{EVENT_DIR_PREFIX}_"
        if resolved.parent != output_root or not resolved.name.startswith(event_prefix):
            LOGGER.error("[ALARM_RETENTION] refused_to_delete=%s", resolved)
            return
        try:
            shutil.rmtree(resolved)
            LOGGER.info("[ALARM_RETENTION] deleted_event=%s", resolved)
        except OSError:
            LOGGER.exception("[ALARM_RETENTION] delete_failed=%s", resolved)

    @staticmethod
    def _risk_crop(image: np.ndarray, risks: Sequence[RiskMatch]) -> np.ndarray:
        boxes = [box for risk in risks for box in (risk.person.box, risk.machinery.box)]
        if not boxes:
            return image.copy()
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        padding = float(config.ALARM_CROP_PADDING_RATIO)
        pad_x = int(round(max(1, x2 - x1) * padding))
        pad_y = int(round(max(1, y2 - y1) * padding))
        image_height, image_width = image.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _write_metadata(path: Path, data: Dict[str, Any]) -> None:
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _update_metadata(self, record: AlarmEventRecord, updates: Dict[str, Any]) -> None:
        path = record.metadata_path
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(updates)
            self._write_metadata(path, data)
        except Exception:
            LOGGER.exception("[ALARM_METADATA] update_failed event=%s path=%s", record.event_id, path)


@dataclass
class _PendingAlarmClip:
    record: AlarmEventRecord
    end_at: float
    frames: List[Tuple[float, bytes]]


class AlarmClipRecorder:
    """以 JPEG 环形缓存保存报警前帧，并在事件后补齐短视频。"""

    def __init__(self, store: AlarmImageStore) -> None:
        self.store = store
        self._prebuffer: deque[Tuple[float, bytes]] = deque()
        self._pending: List[_PendingAlarmClip] = []
        self._last_sample_at = 0.0

    def observe(self, frame: np.ndarray, timestamp: float) -> None:
        if not bool(config.ALARM_SAVE_SHORT_VIDEO):
            return
        max_fps = max(0.1, float(config.ALARM_CLIP_MAX_FPS))
        if self._last_sample_at > 0 and timestamp - self._last_sample_at + 1e-6 < 1.0 / max_fps:
            self._finalize_due(timestamp)
            return
        encoded = self._encode_frame(frame)
        self._last_sample_at = timestamp
        sample = (timestamp, encoded)
        self._prebuffer.append(sample)
        cutoff = timestamp - max(0.0, float(config.ALARM_CLIP_PRE_SECONDS))
        while self._prebuffer and self._prebuffer[0][0] < cutoff:
            self._prebuffer.popleft()
        for pending in self._pending:
            if not pending.frames or pending.frames[-1][0] < timestamp:
                pending.frames.append(sample)
        self._finalize_due(timestamp)

    def start_event(self, record: AlarmEventRecord, timestamp: float) -> None:
        if not bool(config.ALARM_SAVE_SHORT_VIDEO):
            return
        frames = list(self._prebuffer)
        pending = _PendingAlarmClip(
            record=record,
            end_at=timestamp + max(0.0, float(config.ALARM_CLIP_POST_SECONDS)),
            frames=frames,
        )
        self._pending.append(pending)
        self._finalize_due(timestamp)

    def finalize_all(self) -> None:
        for pending in self._pending:
            self.store.complete_short_video(pending.record, pending.frames)
        self._pending.clear()

    def reset(self, finalize_pending: bool = True) -> None:
        if finalize_pending:
            self.finalize_all()
        else:
            self._pending.clear()
        self._prebuffer.clear()
        self._last_sample_at = 0.0

    def _finalize_due(self, timestamp: float) -> None:
        remaining: List[_PendingAlarmClip] = []
        for pending in self._pending:
            if timestamp >= pending.end_at:
                self.store.complete_short_video(pending.record, pending.frames)
            else:
                remaining.append(pending)
        self._pending = remaining

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        quality = max(1, min(100, int(config.ALARM_CLIP_JPEG_QUALITY)))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("报警短视频帧编码失败")
        return encoded.tobytes()


class MachineryHazardDetector:
    """封装模型推理、风险判断、绘制和报警图保存。"""

    def __init__(self, model_path: Optional[Path] = None, settings: Optional[Any] = None) -> None:
        from algorithms.machinery_hazard.defaults import load_default_settings

        validate_config()
        self.settings = settings or load_default_settings()
        self.settings.validate()
        self.model_path = Path(model_path or self.settings.model_path)
        self.model = YOLO(str(self.model_path))
        self._analysis_lock = threading.RLock()
        self.alarm_store = AlarmImageStore()

    def analyze_frame(
        self,
        frame: np.ndarray,
        motion_analyzer: Optional[MachineryMotionAnalyzer] = None,
        timestamp: Optional[float] = None,
        camera_name: Optional[str] = None,
        captured_at: Optional[datetime] = None,
        supplemental_detections: Sequence[Detection] = (),
        settings: Optional[Any] = None,
    ) -> FrameAnalysis:
        with self._analysis_lock:
            return self._analyze_frame_locked(
                frame=frame,
                motion_analyzer=motion_analyzer,
                timestamp=timestamp,
                camera_name=camera_name,
                captured_at=captured_at,
                supplemental_detections=supplemental_detections,
                settings=settings,
            )

    def _analyze_frame_locked(
        self,
        frame: np.ndarray,
        motion_analyzer: Optional[MachineryMotionAnalyzer] = None,
        timestamp: Optional[float] = None,
        camera_name: Optional[str] = None,
        captured_at: Optional[datetime] = None,
        supplemental_detections: Sequence[Detection] = (),
        settings: Optional[Any] = None,
    ) -> FrameAnalysis:
        active_settings = settings or self.settings
        crane_scene = active_settings.advanced.get("scene_mode") == "crane_boom_person"
        supplemental_detections = tuple(
            item
            for item in supplemental_detections
            if item.confidence >= float(active_settings.confidence_threshold)
            and (not crane_scene or item.class_id == 13)
        )
        detections = self._merge_supplemental_machinery(
            self._predict(frame, settings=active_settings),
            supplemental_detections,
        )
        if crane_scene:
            from algorithms.machinery_hazard.crane_regions import deduplicate
            detections = deduplicate([item for item in detections if item.class_id in {0, 1, 2, 3, 13}])
        persons = [item for item in detections if item.class_id in active_settings.person_class_ids]
        machinery_class_ids = set(active_settings.machinery_class_ids) | set(
            config.SUPPLEMENTAL_MACHINERY_CLASS_IDS
        )
        if crane_scene:
            machinery_class_ids = {1, 3}
        machinery = [item for item in detections if item.class_id in machinery_class_ids]

        motion_assessments: List[MotionAssessment] = []
        rider_assessments: List[RiderAssessment] = []
        if motion_analyzer is not None and not crane_scene:
            analysis_timestamp = timestamp if timestamp is not None else time.monotonic()
            original_persons = persons
            if bool(active_settings.rider_filter_enabled):
                persons = motion_analyzer.update_person_detections(original_persons, analysis_timestamp)
            tracked_persons_by_object_id = {
                id(original): tracked
                for original, tracked in zip(original_persons, persons)
            }
            original_machinery = machinery
            machinery, motion_assessments = motion_analyzer.update_detections(
                original_machinery,
                persons,
                analysis_timestamp,
            )
            tracked_by_object_id = {
                id(original): tracked
                for original, tracked in zip(original_machinery, machinery)
            }
            detections = [
                tracked_by_object_id.get(id(item), tracked_persons_by_object_id.get(id(item), item))
                for item in detections
            ]
            rider_assessments = motion_analyzer.evaluate_riders(
                persons,
                machinery,
                analysis_timestamp,
            )

        assessments_by_track = {
            assessment.track_id: assessment
            for assessment in motion_assessments
        }
        crane_regions = {}
        if crane_scene:
            from algorithms.machinery_hazard.crane_regions import build_regions, foot_in_region
            regions = build_regions(machinery, detections, frame.shape,
                                    margin_ratio=float(config.CRANE_REGION_MARGIN_RATIO),
                                    score_margin=float(config.CRANE_ASSOCIATION_SCORE_MARGIN),
                                    min_score=float(config.CRANE_ASSOCIATION_MIN_SCORE),
                                    min_aux_confidence=float(config.CRANE_AUXILIARY_MIN_CONFIDENCE))
            crane_regions = {id(machine): region for machine, region in zip(machinery, regions)}
        machinery_buffers = [
            (item, self._make_crane_candidate_buffer(item, frame.shape, crane_regions[id(item)]) if crane_scene
             else self._make_buffer(item, assessments_by_track.get(item.track_id)))
            for item in machinery
        ]
        rider_pairs = {
            (assessment.person_track_id, assessment.machinery_track_id)
            for assessment in rider_assessments
            if assessment.is_rider
        }

        risks: List[RiskMatch] = []
        suppressed_risks: List[RiskMatch] = []
        for person in persons:
            for machine, buffer in machinery_buffers:
                overlap_ratio = self._person_buffer_overlap_ratio(person.box, buffer, frame.shape)
                in_region = not crane_scene or foot_in_region(person.box, crane_regions[id(machine)])
                if in_region and overlap_ratio >= active_settings.overlap_ratio_threshold and (not crane_scene or overlap_ratio > 0):
                    assessment = assessments_by_track.get(machine.track_id)
                    risk = RiskMatch(
                        person=person,
                        machinery=machine,
                        buffer=buffer,
                        overlap_ratio=overlap_ratio,
                        candidate_region=crane_regions.get(id(machine)),
                        motion_reason="crane_boom_area_review" if crane_scene else (assessment.reason if assessment is not None else None),
                    )
                    if (person.track_id, machine.track_id) in rider_pairs:
                        suppressed_risks.append(replace(risk, suppression_reason="rider"))
                        continue
                    if not crane_scene and motion_analyzer is not None and bool(active_settings.machinery_motion_enabled):
                        if assessment is None or not assessment.active:
                            if motion_analyzer.static_operation_candidate(machine, person, frame.shape, detections):
                                risk = replace(risk, motion_reason="static_operation_review")
                            else:
                                suppressed_risks.append(replace(risk, suppression_reason="machinery_not_active"))
                                continue
                    if crane_scene and not platform_config.ALARM_REVIEW_ENABLED:
                        suppressed_risks.append(replace(risk, suppression_reason="crane_scene_requires_review"))
                        continue
                    risks.append(risk)

        annotated = self._draw_alarm_overlay(
            frame,
            risks,
            camera_name=camera_name or str(config.DEFAULT_CAMERA_DISPLAY_NAME),
            captured_at=captured_at or datetime.now(),
        )
        return FrameAnalysis(
            detections=detections,
            persons=persons,
            machinery=machinery,
            machinery_buffers=machinery_buffers,
            risks=risks,
            suppressed_risks=suppressed_risks,
            motion_assessments=motion_assessments,
            rider_assessments=rider_assessments,
            annotated=annotated,
        )

    def process_image(self, source: str) -> Dict[str, Any]:
        image = read_image_with_unicode_path(Path(source))
        return self.process_image_array(image, source_name=source)

    def process_image_array(self, image: np.ndarray, source_name: str = "upload") -> Dict[str, Any]:
        if not in_detection_time():
            return {
                "risk": False,
                "alarm_path": None,
                "alarm_original_path": None,
                "alarm_event_dir": None,
                "alarm_crop_path": None,
                "alarm_metadata_path": None,
                "risk_count": 0,
                "suppressed_risk_count": 0,
                "person_count": 0,
                "machinery_count": 0,
                "detections": [],
                "risks": [],
                "suppressed_risks": [],
                "motion_assessments": [],
                "rider_assessments": [],
                "skipped": True,
                "skip_reason": "outside_detection_time",
            }

        captured_at = datetime.now()
        display_name = camera_display_name(source_name)
        analysis = self.analyze_frame(
            image,
            camera_name=display_name,
            captured_at=captured_at,
        )
        event_record: Optional[AlarmEventRecord] = None
        if analysis.has_risk:
            event_record = self.alarm_store.create_event(
                annotated_image=analysis.annotated,
                original_image=image,
                analysis=analysis,
                source_name=source_name,
                camera_name=display_name,
                captured_at=captured_at,
                allow_short_video=False,
            )
        result = analysis.to_dict(
            alarm_path=event_record.annotated_path if event_record else None,
            alarm_original_path=event_record.original_path if event_record else None,
        )
        result["alarm_event_dir"] = str(event_record.event_dir) if event_record else None
        result["alarm_crop_path"] = str(event_record.crop_path) if event_record and event_record.crop_path else None
        result["alarm_metadata_path"] = (
            str(event_record.metadata_path) if event_record and event_record.metadata_path else None
        )
        return result

    def process_video(
        self,
        source: str,
        show: bool = False,
        save_video: bool = False,
        stop_event: Optional[threading.Event] = None,
        event_context: Optional[Dict[str, Any]] = None,
        event_callback: Optional[Callable[[AlarmEventRecord], None]] = None,
        capture_factory: Optional[Callable[[], Tuple[Any, str]]] = None,
        alarm_confirmation: Optional[Callable[..., Dict[str, Any]]] = None,
        algorithm_alarm_intervals: Optional[Dict[str, float]] = None,
        algorithm_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        del alarm_confirmation  # Object-family small-target algorithms use this hook.
        should_reconnect = bool(
            config.VIDEO_RECONNECT_ENABLED
            and (is_camera_index_code_source(source) or is_reconnectable_source(source))
        )
        max_runtime = float(config.STREAM_MAX_RUNTIME_SECONDS)
        max_reconnects = int(config.VIDEO_RECONNECT_MAX_ATTEMPTS)
        reconnect_interval = float(config.VIDEO_RECONNECT_INTERVAL_SECONDS)
        read_fail_limit = int(config.VIDEO_READ_FAIL_RECONNECT_FRAMES)
        raw_parameters = (algorithm_parameters or {}).get(
            machinery_defaults.ALGORITHM_CODE,
            {},
        )
        setting_updates = {
            key: raw_parameters[key]
            for key in (
                "person_confidence_threshold",
                "confidence_threshold",
                "overlap_ratio_threshold",
                "risk_consecutive_frames",
            )
            if key in raw_parameters
        }
        active_settings = replace(self.settings, **setting_updates) if setting_updates else self.settings
        active_settings.validate()
        detect_interval = float(active_settings.detection_interval_seconds)
        alarm_cooldown_seconds = float(
            (algorithm_alarm_intervals or {}).get(
                machinery_defaults.ALGORITHM_CODE,
                active_settings.alarm_cooldown_seconds,
            )
        )
        start_at = time.monotonic()
        writer = None
        video_output_path = None
        frame_index = 0
        detected_frame_count = 0
        skipped_frame_count = 0
        risk_streak = 0
        last_alarm_at = -10**9
        last_static_review_at = -10**9
        last_detection_at = 0.0
        reconnect_attempts = 0
        alarm_paths: List[Path] = []
        alarm_original_paths: List[Path] = []
        alarm_records: List[AlarmEventRecord] = []
        stop_reason = "end_of_stream"
        window_name = f"{config.DISPLAY_WINDOW_NAME}-{safe_stem(source)}"
        source_label = safe_stem(source)
        show_window = show
        motion_analyzer = MachineryMotionAnalyzer(source_label)
        clip_recorder = AlarmClipRecorder(self.alarm_store)
        display_name = camera_display_name(source)
        detection_period_active = False
        machinery_supplement_key = supplement_key(source, event_context)

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    stop_reason = "task_stopped"
                    break
                if max_runtime > 0 and time.monotonic() - start_at >= max_runtime:
                    stop_reason = "max_runtime_reached"
                    break

                try:
                    if capture_factory is not None:
                        cap, resolved_source = capture_factory()
                    else:
                        cap, resolved_source = open_video_capture(source)
                except Exception as exc:
                    stop_reason = "resolve_failed"
                    LOGGER.warning("[STREAM] source=%s resolve_failed=%s", source_label, exc)
                    if not should_reconnect:
                        raise
                    if max_reconnects > 0 and reconnect_attempts >= max_reconnects:
                        stop_reason = "reconnect_limit_reached"
                        break
                    reconnect_attempts += 1
                    if stop_event is not None and stop_event.wait(reconnect_interval):
                        stop_reason = "task_stopped"
                        break
                    if stop_event is None:
                        time.sleep(reconnect_interval)
                    continue

                LOGGER.info("[STREAM] source=%s opening=%s", source_label, resolved_source)
                if not cap.isOpened():
                    cap.release()
                    stop_reason = "open_failed"
                    LOGGER.warning("[STREAM] source=%s open_failed=%s", source_label, resolved_source)
                    if not should_reconnect:
                        raise RuntimeError(f"无法打开视频源：{source}")
                    if max_reconnects > 0 and reconnect_attempts >= max_reconnects:
                        stop_reason = "reconnect_limit_reached"
                        break
                    reconnect_attempts += 1
                    if stop_event is not None and stop_event.wait(reconnect_interval):
                        stop_reason = "task_stopped"
                        break
                    if stop_event is None:
                        time.sleep(reconnect_interval)
                    continue

                LOGGER.info("[STREAM] source=%s opened=%s", source_label, resolved_source)
                if not getattr(cap, "is_shared_stream", False):
                    warmup_video_capture(source_label, cap)
                motion_analyzer.reset()
                clip_recorder.reset(finalize_pending=True)
                detection_period_active = False

                read_failures = 0
                reconnect_needed = False
                stop_current_source = False
                try:
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            stop_reason = "task_stopped"
                            stop_current_source = True
                            break
                        if max_runtime > 0 and time.monotonic() - start_at >= max_runtime:
                            stop_reason = "max_runtime_reached"
                            stop_current_source = True
                            break

                        ok, frame = cap.read()
                        if not ok or frame is None:
                            if not should_reconnect:
                                stop_reason = "end_of_stream"
                                stop_current_source = True
                                break

                            read_failures += 1
                            LOGGER.warning(
                                "[STREAM] source=%s read_failed=%s/%s stream=%s",
                                source_label,
                                read_failures,
                                read_fail_limit,
                                resolved_source,
                            )
                            if read_failures >= read_fail_limit:
                                stop_reason = "read_failed"
                                reconnect_needed = True
                                LOGGER.warning("[STREAM] source=%s read_failure_limit_reached reconnecting", source_label)
                                break
                            if stop_event is not None and stop_event.wait(0.05):
                                stop_reason = "task_stopped"
                                stop_current_source = True
                                break
                            if stop_event is None:
                                time.sleep(0.05)
                            continue

                        read_failures = 0
                        frame_index += 1
                        annotated = frame
                        frame_timestamp = time.monotonic()
                        try:
                            clip_recorder.observe(frame, frame_timestamp)
                        except Exception:
                            LOGGER.exception(
                                "[ALARM_CLIP] source=%s frame=%s buffer_failed",
                                source_label,
                                frame_index,
                            )

                        if not in_detection_time():
                            if detection_period_active:
                                motion_analyzer.reset()
                                detection_period_active = False
                            risk_streak = 0
                            skipped_frame_count += 1
                        else:
                            if not detection_period_active:
                                motion_analyzer.reset()
                                detection_period_active = True
                            now = frame_timestamp
                            active_motion_analyzer = (
                                motion_analyzer
                                if (
                                    bool(active_settings.machinery_motion_enabled)
                                    or bool(active_settings.rider_filter_enabled)
                                )
                                else None
                            )
                            if active_motion_analyzer is not None:
                                try:
                                    with self._analysis_lock:
                                        active_motion_analyzer.observe_frame(frame, now)
                                except Exception:
                                    LOGGER.exception(
                                        "[MOTION] source=%s frame=%s optical_flow_failed",
                                        source_label,
                                        frame_index,
                                    )
                                    active_motion_analyzer.reset()
                            need_detect = detect_interval <= 0 or last_detection_at <= 0 or now - last_detection_at >= detect_interval
                            if need_detect:
                                last_detection_at = now
                                detected_frame_count += 1
                                captured_at = datetime.now()
                                supplemental_detections = [
                                    Detection(
                                        class_id=item.class_id,
                                        class_name=item.class_name,
                                        confidence=item.confidence,
                                        box=item.box,
                                    )
                                    for item in MACHINERY_SUPPLEMENT_BROKER.latest(
                                        machinery_supplement_key,
                                        max_age_seconds=float(
                                            config.MACHINERY_SUPPLEMENT_MAX_AGE_SECONDS
                                        ),
                                        now=now,
                                    )
                                ]
                                try:
                                    analysis = self.analyze_frame(
                                        frame,
                                        motion_analyzer=active_motion_analyzer,
                                        timestamp=now,
                                        camera_name=display_name,
                                        captured_at=captured_at,
                                        supplemental_detections=supplemental_detections,
                                        settings=active_settings,
                                    )
                                except Exception:
                                    LOGGER.exception(
                                        "[DETECT] source=%s frame=%s detect_count=%s inference_failed",
                                        source_label,
                                        frame_index,
                                        detected_frame_count,
                                    )
                                    risk_streak = 0
                                    continue
                                annotated = analysis.annotated
                                risk_streak = risk_streak + 1 if analysis.has_risk else 0
                                log_every = max(1, int(getattr(config, "DETECTION_LOG_EVERY_N_FRAMES", 1)))
                                if getattr(config, "LOG_DETECTION_FRAMES", True) and detected_frame_count % log_every == 0:
                                    LOGGER.info(
                                        "[DETECT] source=%s frame=%s detect_count=%s persons=%s machinery=%s risks=%s suppressed=%s riders=%s risk_streak=%s motion=%s",
                                        source_label,
                                        frame_index,
                                        detected_frame_count,
                                        len(analysis.persons),
                                        len(analysis.machinery),
                                        len(analysis.risks),
                                        len(analysis.suppressed_risks),
                                        sum(assessment.is_rider for assessment in analysis.rider_assessments),
                                        risk_streak,
                                        ",".join(
                                            f"{assessment.track_id}:{assessment.state}"
                                            for assessment in analysis.motion_assessments
                                        ) or "none",
                                    )
                                    if analysis.suppressed_risks:
                                        LOGGER.info(
                                            "[RISK_SUPPRESSED] source=%s frame=%s matches=%s",
                                            source_label,
                                            frame_index,
                                            ",".join(
                                                f"person:{risk.person.track_id}->machinery:{risk.machinery.track_id}:{risk.suppression_reason}"
                                                for risk in analysis.suppressed_risks
                                            ),
                                        )

                                if risk_streak >= active_settings.risk_consecutive_frames:
                                    cooldown_seconds = alarm_cooldown_seconds
                                    static_only = all(r.motion_reason == "static_operation_review" for r in analysis.risks)
                                    last_same_path = last_static_review_at if static_only else last_alarm_at
                                    can_save = cooldown_seconds <= 0 or time.monotonic() - last_same_path >= cooldown_seconds
                                    if can_save:
                                        event_record = self.alarm_store.create_event(
                                            annotated_image=annotated,
                                            original_image=frame,
                                            analysis=analysis,
                                            source_name=source,
                                            camera_name=display_name,
                                            captured_at=captured_at,
                                            frame_index=frame_index,
                                            allow_short_video=True,
                                            event_context=event_context,
                                        )
                                        alarm_records.append(event_record)
                                        if event_record.annotated_path is not None:
                                            alarm_paths.append(event_record.annotated_path)
                                        if event_record.original_path is not None:
                                            alarm_original_paths.append(event_record.original_path)
                                        clip_recorder.start_event(event_record, now)
                                        if event_callback is not None:
                                            try:
                                                event_callback(event_record)
                                            except Exception:
                                                LOGGER.exception(
                                                    "[ALARM] event callback failed event=%s",
                                                    event_record.event_id,
                                                )
                                        if static_only:
                                            last_static_review_at = time.monotonic()
                                        else:
                                            last_alarm_at = time.monotonic()
                                        if getattr(config, "LOG_ALARMS", True):
                                            LOGGER.warning(
                                                "[ALARM] source=%s camera=%s frame=%s risk_count=%s risk_streak=%s motion_reason=%s event=%s path=%s original_path=%s",
                                                source_label,
                                                display_name,
                                                frame_index,
                                                len(analysis.risks),
                                                risk_streak,
                                                ",".join(sorted({
                                                    risk.motion_reason or "single_frame"
                                                    for risk in analysis.risks
                                                })),
                                                event_record.event_dir,
                                                event_record.annotated_path,
                                                event_record.original_path,
                                            )
                            else:
                                skipped_frame_count += 1

                        if save_video:
                            if writer is None:
                                video_output_path, writer = self._create_video_writer(source, annotated, cap)
                            writer.write(annotated)

                        if show_window:
                            show_window, key = show_frame_or_disable(window_name, annotated)
                            if key in (27, ord("q")):
                                stop_reason = "user_stopped"
                                stop_current_source = True
                                break
                finally:
                    cap.release()

                if stop_current_source:
                    break
                if not reconnect_needed:
                    break
                if max_reconnects > 0 and reconnect_attempts >= max_reconnects:
                    stop_reason = "reconnect_limit_reached"
                    break

                reconnect_attempts += 1
                LOGGER.info(
                    "[STREAM] source=%s reconnect_attempt=%s sleep=%s",
                    source_label,
                    reconnect_attempts,
                    reconnect_interval,
                )
                if stop_event is not None and stop_event.wait(reconnect_interval):
                    stop_reason = "task_stopped"
                    break
                if stop_event is None:
                    time.sleep(reconnect_interval)
        finally:
            clip_recorder.finalize_all()
            if writer is not None:
                writer.release()
            if show:
                destroy_windows_safely()

        return {
            "source": source,
            "frames": frame_index,
            "detected_frames": detected_frame_count,
            "skipped_frames": skipped_frame_count,
            "alarm_count": len(alarm_records),
            "alarm_paths": [str(path) for path in alarm_paths],
            "alarm_original_paths": [str(path) for path in alarm_original_paths],
            "alarm_event_dirs": [str(record.event_dir) for record in alarm_records],
            "alarm_crop_paths": [str(record.crop_path) for record in alarm_records if record.crop_path],
            "alarm_metadata_paths": [str(record.metadata_path) for record in alarm_records if record.metadata_path],
            "alarm_clip_paths": [str(record.clip_path) for record in alarm_records if record.clip_path],
            "reconnect_attempts": reconnect_attempts,
            "stop_reason": stop_reason,
            "video_output_path": str(video_output_path) if video_output_path else None,
        }

    def _create_video_writer(
        self,
        source: str,
        frame: np.ndarray,
        cap: cv2.VideoCapture,
    ) -> Tuple[Path, cv2.VideoWriter]:
        config.VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_output_path = config.VIDEO_OUTPUT_DIR / f"{safe_stem(source)}_{timestamp}_hazard_detect.mp4"
        writer = cv2.VideoWriter(
            str(video_output_path),
            cv2.VideoWriter_fourcc(*config.VIDEO_CODEC),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频输出文件：{video_output_path}")
        return video_output_path, writer

    def process_source(self, source: str, show: bool = False, save_video: bool = False) -> Dict[str, Any]:
        kind = source_type(source)
        if kind == "image":
            return self.process_image(source)
        return self.process_video(source, show=show, save_video=save_video)

    def _predict(self, frame: np.ndarray, settings: Optional[Any] = None) -> List[Detection]:
        active_settings = settings or self.settings
        confidence_threshold = float(active_settings.confidence_threshold)
        person_confidence_threshold = float(active_settings.person_confidence_threshold)
        predict_confidence_threshold = min(confidence_threshold, person_confidence_threshold)

        with self._analysis_lock:
            results = self.model.predict(
                source=frame,
                conf=predict_confidence_threshold,
                iou=float(active_settings.iou_threshold),
                imgsz=int(active_settings.image_size),
                device=active_settings.device or None,
                classes=([0, 1, 2, 3] if active_settings.advanced.get("scene_mode") == "crane_boom_person"
                         else sorted(set(active_settings.person_class_ids + active_settings.machinery_class_ids) | {2})),
                max_det=int(active_settings.max_detections),
                verbose=False,
            )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or config.CLASS_NAMES
        detections: List[Detection] = []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        for box_values, class_id, confidence in zip(xyxy, classes, confidences):
            class_id = int(class_id)
            confidence = float(confidence)
            required_confidence = (
                person_confidence_threshold if class_id in active_settings.person_class_ids else confidence_threshold
            )
            if confidence < required_confidence:
                continue

            box = clip_box(box_values, frame.shape)
            if box is None:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, config.CLASS_NAMES.get(class_id, f"class_{class_id}"))),
                    confidence=confidence,
                    box=box,
                )
            )
        return detections

    @staticmethod
    def _merge_supplemental_machinery(
        primary: Sequence[Detection],
        supplemental: Sequence[Detection],
    ) -> List[Detection]:
        """Merge model2 machinery into best.pt results without duplicate boxes."""

        merged = list(primary)
        iou_threshold = float(config.MACHINERY_SUPPLEMENT_DEDUP_IOU_THRESHOLD)
        for extra in supplemental:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if existing.class_id == extra.class_id
                    and box_iou(existing.box, extra.box) >= iou_threshold
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(extra)
                continue
            existing = merged[duplicate_index]
            merged[duplicate_index] = replace(
                existing,
                confidence=max(existing.confidence, extra.confidence),
                box=(
                    min(existing.box[0], extra.box[0]),
                    min(existing.box[1], extra.box[1]),
                    max(existing.box[2], extra.box[2]),
                    max(existing.box[3], extra.box[3]),
                ),
            )
        return merged

    @staticmethod
    def _make_crane_candidate_buffer(machinery: Detection, image_shape, region=None) -> BufferGeometry:
        # Coarse image-space recall region, NOT a calibrated boom danger zone.
        # The dedicated VLM review must verify the actual boom/person relation.
        height, width = image_shape[:2]
        x1, y1, x2, _ = machinery.box
        margin = (x2 - x1) * float(config.CRANE_CANDIDATE_HORIZONTAL_MARGIN)
        left, right = max(0, int(x1 - margin)), min(width, int(x2 + margin))
        top, bottom = max(0, y1), height
        if region is not None:
            left, top, right, bottom = region["box"]
        return BufferGeometry("rectangle", ((left + right) // 2, (top + bottom) // 2),
                              (max(1, (right - left + 1) // 2), max(1, (bottom - top + 1) // 2)))

    def _make_buffer(
        self,
        machinery: Detection,
        motion: Optional[MotionAssessment] = None,
    ) -> BufferGeometry:
        x1, y1, x2, y2 = machinery.box
        width = x2 - x1
        height = y2 - y1
        center = (x1 + width // 2, y1 + height // 2)
        scale = float(config.BUFFER_SCALE)

        if config.BUFFER_SHAPE == "circle":
            radius = max(1, int(round(max(width, height) * scale / 2)))
            return BufferGeometry(shape="circle", center=center, axes=(radius, radius))

        class_scales = config.RISK_ZONE_CLASS_SCALES.get(machinery.class_id, (scale, scale))
        axis_x = max(1.0, width * float(class_scales[0]) / 2.0)
        axis_y = max(1.0, height * float(class_scales[1]) / 2.0)
        center_x = x1 + width / 2.0
        center_y = y1 + height * float(config.RISK_ZONE_CENTER_Y_RATIO)
        angle = 0.0

        if (
            motion is not None
            and motion.body_moving
            and bool(config.RISK_ZONE_USE_MOTION_DIRECTION)
        ):
            direction_x, direction_y = motion.motion_direction
            direction_length = math.hypot(direction_x, direction_y)
            if direction_length > 1e-6:
                direction_x /= direction_length
                direction_y /= direction_length
                major_axis = max(axis_x, axis_y)
                minor_axis = min(axis_x, axis_y)
                diagonal = max(1.0, math.hypot(width, height))
                speed_pixels_per_second = motion.body_speed_ratio_per_second * diagonal
                requested_extension = speed_pixels_per_second * float(config.RISK_ZONE_REACTION_TIME_SECONDS)
                max_extension = major_axis * float(config.RISK_ZONE_MAX_FORWARD_EXTENSION_RATIO)
                extension = max(0.0, min(requested_extension, max_extension))
                center_x += direction_x * extension / 2.0
                center_y += direction_y * extension / 2.0
                axis_x = major_axis + extension / 2.0
                axis_y = minor_axis
                angle = math.degrees(math.atan2(direction_y, direction_x))

        return BufferGeometry(
            shape="ellipse",
            center=(int(round(center_x)), int(round(center_y))),
            axes=(max(1, int(round(axis_x))), max(1, int(round(axis_y)))),
            angle=angle,
        )

    def _person_buffer_overlap_ratio(
        self,
        person_box: Box,
        buffer: BufferGeometry,
        image_shape: Tuple[int, int, int],
    ) -> float:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = person_box
        lower_body_ratio = float(config.PERSON_RISK_LOWER_BODY_RATIO)
        y1 = max(y1, int(round(y2 - max(1, y2 - y1) * lower_body_ratio)))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi_width = x2 - x1
        roi_height = y2 - y1
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        local_center = (buffer.center[0] - x1, buffer.center[1] - y1)

        if buffer.shape == "rectangle":
            left, top = buffer.center[0] - buffer.axes[0], buffer.center[1] - buffer.axes[1]
            right, bottom = buffer.center[0] + buffer.axes[0], buffer.center[1] + buffer.axes[1]
            return (max(0, min(x2, right) - max(x1, left)) *
                    max(0, min(y2, bottom) - max(y1, top))) / (roi_width * roi_height)
        if buffer.shape == "circle":
            cv2.circle(mask, local_center, buffer.axes[0], 255, thickness=-1)
        else:
            cv2.ellipse(mask, local_center, buffer.axes, buffer.angle, 0, 360, 255, thickness=-1)

        overlap_area = int(np.count_nonzero(mask))
        person_area = roi_width * roi_height
        return overlap_area / person_area if person_area > 0 else 0.0

    def _draw_alarm_overlay(
        self,
        frame: np.ndarray,
        risks: Sequence[RiskMatch],
        camera_name: str,
        captured_at: datetime,
    ) -> np.ndarray:
        annotated = frame.copy()
        if not risks:
            return annotated

        person_color = tuple(int(value) for value in config.ALARM_PERSON_COLOR_BGR)
        machinery_color = tuple(int(value) for value in config.ALARM_MACHINERY_COLOR_BGR)
        risk_persons = self._deduplicate_alarm_detections([risk.person for risk in risks])
        risk_machinery = self._deduplicate_alarm_detections([risk.machinery for risk in risks])
        labels: List[Tuple[Detection, Tuple[int, int, int], Optional[str]]] = []
        if bool(config.DRAW_PERSON_BOX):
            for person in risk_persons:
                annotated = self._draw_alarm_box(annotated, person.box, person_color)
                labels.append((person, person_color, None))
        if bool(config.DRAW_MACHINERY_BOX):
            for machine in risk_machinery:
                annotated = self._draw_alarm_box(annotated, machine.box, machinery_color)
                labels.append((machine, machinery_color, None))

        annotated = self._draw_detection_labels(annotated, labels)
        annotated = self._draw_centered_alarm_title(annotated)
        return annotated

    @staticmethod
    def _deduplicate_alarm_detections(detections: Sequence[Detection]) -> List[Detection]:
        selected: List[Detection] = []
        seen_track_ids = set()
        iou_threshold = float(config.ALARM_DRAW_DEDUP_IOU_THRESHOLD)
        containment_threshold = float(config.ALARM_DRAW_DEDUP_CONTAINMENT_THRESHOLD)
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if detection.track_id is not None and detection.track_id in seen_track_ids:
                continue
            duplicate = False
            for existing in selected:
                if box_iou(detection.box, existing.box) >= iou_threshold:
                    duplicate = True
                    break
                intersection_x1 = max(detection.box[0], existing.box[0])
                intersection_y1 = max(detection.box[1], existing.box[1])
                intersection_x2 = min(detection.box[2], existing.box[2])
                intersection_y2 = min(detection.box[3], existing.box[3])
                intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                detection_area = max(1, detection.box[2] - detection.box[0]) * max(
                    1,
                    detection.box[3] - detection.box[1],
                )
                existing_area = max(1, existing.box[2] - existing.box[0]) * max(
                    1,
                    existing.box[3] - existing.box[1],
                )
                containment = intersection / max(1, min(detection_area, existing_area))
                if containment >= containment_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(detection)
            if detection.track_id is not None:
                seen_track_ids.add(detection.track_id)
        return selected

    @staticmethod
    def _draw_alarm_box(
        image: np.ndarray,
        box: Box,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        border = max(1, int(config.ALARM_RISK_BOX_BORDER_THICKNESS))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=border)
        return image

    def _draw_buffer(self, image: np.ndarray, buffer: BufferGeometry, color: Tuple[int, int, int], thickness: int) -> None:
        if buffer.shape == "circle":
            cv2.circle(image, buffer.center, buffer.axes[0], color, thickness=thickness)
        else:
            cv2.ellipse(image, buffer.center, buffer.axes, buffer.angle, 0, 360, color, thickness=thickness)

    def _draw_box(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int],
        thickness: int,
        label_override: Optional[str] = None,
    ) -> None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness)
        if bool(config.DRAW_LABELS):
            image[:] = self._draw_detection_labels(image, [(detection, color, label_override)])

    def _draw_detection_labels(
        self,
        image: np.ndarray,
        labels: Sequence[Tuple[Detection, Tuple[int, int, int], Optional[str]]],
    ) -> np.ndarray:
        """Draw UTF-8 class labels with a Chinese-capable TrueType font."""
        if not bool(config.DRAW_LABELS) or not labels:
            return image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        image_height, image_width = image.shape[:2]
        preferred_size = max(12, int(config.ALARM_INFO_FONT_SIZE))

        for detection, bgr_color, label_override in labels:
            display_name = config.ALARM_CLASS_LABELS.get(detection.class_id, "未知目标")
            label = label_override or f"{display_name} {detection.confidence:.2f}"
            x1, y1, _x2, _y2 = detection.box
            text_x = max(0, min(image_width - 1, int(x1)))
            max_width = max(20, image_width - text_x - 6)
            font = self._fit_alarm_font(draw, label, preferred_size, max_width)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = max(1, bbox[2] - bbox[0])
            text_height = max(1, bbox[3] - bbox[1])
            padding_x = 4
            padding_y = 3
            background_height = text_height + padding_y * 2
            if y1 >= background_height:
                background_y1 = int(y1) - background_height
            else:
                background_y1 = min(image_height - background_height, max(0, int(y1)))
            rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
            draw.rectangle(
                (
                    text_x,
                    background_y1,
                    min(image_width - 1, text_x + text_width + padding_x * 2),
                    min(image_height - 1, background_y1 + background_height),
                ),
                fill=rgb_color,
            )
            draw.text(
                (text_x + padding_x - bbox[0], background_y1 + padding_y - bbox[1]),
                label,
                font=font,
                fill=(255, 255, 255),
            )

        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_centered_alarm_title(self, image: np.ndarray) -> np.ndarray:
        title = str(config.ALARM_RISK_TEXT)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        max_width = max(20, image.shape[1] - 24)
        title_font = self._fit_alarm_font(draw, title, int(config.ALARM_TITLE_FONT_SIZE), max_width)
        bbox = draw.textbbox((0, 0), title, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        left = (image.shape[1] - text_width) // 2
        top = (image.shape[0] - text_height) // 2
        background = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        background_draw = ImageDraw.Draw(background)
        background_draw.rectangle(
            (left - 12, top - 8, left + text_width + 12, top + text_height + 8),
            fill=(0, 0, 0, 150),
        )
        pil_image = Image.alpha_composite(pil_image.convert("RGBA"), background).convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        title_x = left - bbox[0]
        title_y = top - bbox[1]
        draw.text((title_x, title_y), title, font=title_font, fill=(255, 0, 0))
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fit_alarm_font(
        draw: ImageDraw.ImageDraw,
        text: str,
        preferred_size: int,
        max_width: int,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = [
            Path(config.ALARM_FONT_PATH),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        smallest_true_type_font: Optional[ImageFont.FreeTypeFont] = None
        for size in range(max(12, preferred_size), 11, -1):
            for font_path in font_candidates:
                if not font_path.exists():
                    continue
                font = ImageFont.truetype(str(font_path), size=size)
                smallest_true_type_font = font
                bbox = draw.textbbox((0, 0), text, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    return font
        # Long labels may not fit even at the minimum size. Keep a TrueType
        # font in that case: Pillow's bitmap fallback cannot reliably measure
        # or draw Chinese text and can raise "Only supported for TrueType fonts".
        if smallest_true_type_font is not None:
            return smallest_true_type_font
        return ImageFont.load_default()


def iter_config_source_values(values: Any) -> List[Any]:
    if isinstance(values, (str, Path)):
        return [values]
    return list(values)


def configured_image_sources() -> List[str]:
    return [source for source in (normalize_config_source(item) for item in iter_config_source_values(
        config.IMAGE_SOURCES)) if source]


def configured_direct_video_stream_sources() -> List[str]:
    return [
        source
        for source in (normalize_config_source(item) for item in iter_config_source_values(config.VIDEO_STREAM_SOURCES))
        if source
    ]


def configured_camera_index_codes() -> List[str]:
    camera_codes = getattr(config, "VIDEO_STREAM_CAMERA_INDEX_CODES", [])
    return [str(item).strip() for item in iter_config_source_values(camera_codes) if str(item).strip()]


def configured_video_stream_sources_with_errors() -> Tuple[List[str], List[Dict[str, str]]]:
    sources = configured_direct_video_stream_sources()
    errors: List[Dict[str, str]] = []

    for camera_index_code in configured_camera_index_codes():
        sources.append(make_camera_index_code_source(camera_index_code))

    return sources, errors


def configured_video_stream_sources() -> List[str]:
    sources, _ = configured_video_stream_sources_with_errors()
    return sources


def has_configured_sources() -> bool:
    return bool(configured_image_sources() or configured_direct_video_stream_sources() or configured_camera_index_codes())


def process_configured_video_source(source: str, show: bool, save_video: bool) -> Dict[str, Any]:
    detector = MachineryHazardDetector()
    try:
        result = detector.process_video(source, show=show, save_video=save_video)
        result["source"] = source
        return result
    except Exception as exc:
        return {"source": source, "error": str(exc)}


def process_configured_sources(show: bool = False, save_video: bool = False) -> Dict[str, Any]:
    image_sources = configured_image_sources()
    video_sources, video_source_errors = configured_video_stream_sources_with_errors()
    results: Dict[str, Any] = {
        "image_sources": image_sources,
        "video_stream_sources": video_sources,
        "video_stream_source_errors": video_source_errors,
        "images": [],
        "videos": [],
    }

    if image_sources:
        image_detector = MachineryHazardDetector()
        for source in image_sources:
            try:
                result = image_detector.process_image(source)
                result["source"] = source
            except Exception as exc:
                result = {"source": source, "error": str(exc)}
            results["images"].append(result)

    if not video_sources:
        return results

    parallel = bool(config.PROCESS_VIDEO_STREAMS_IN_PARALLEL) and len(video_sources) > 1
    if not parallel:
        for source in video_sources:
            results["videos"].append(process_configured_video_source(source, show=show, save_video=save_video))
        return results

    workers = int(config.VIDEO_STREAM_WORKERS) or len(video_sources)
    workers = max(1, min(workers, len(video_sources)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_source = {
            executor.submit(process_configured_video_source, source, show, save_video): source for source in video_sources
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                results["videos"].append(future.result())
            except Exception as exc:
                results["videos"].append({"source": source, "error": str(exc)})

    return results


def create_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise RuntimeError("HTTP 服务模式需要安装 fastapi 和 uvicorn，或改用命令行 --source 模式。") from exc

    from live_preview import LivePreviewManager, register_live_preview_routes
    from video_channels import VideoChannelStore, register_video_channel_routes
    from video_alarms import VideoAlarmStore, register_video_alarm_routes
    from video_tasks import VideoTaskStore, register_video_task_routes
    from manual_reviews import ManualReviewStore, register_manual_review_routes
    from model_training import ModelTrainingManager, register_model_training_routes
    from task_runtime import TaskExecutionManager
    from live_alarm_review import LiveAlarmReviewManager
    from algorithms import algorithm_registry
    from algorithms.composite import create_task_algorithm_runner
    from algorithms.machinery_hazard.defaults import ALGORITHM_CODE
    from shared_stream import SharedStreamManager

    app = FastAPI(title="Machinery Hazard Detection Service")
    detector_holder: Dict[str, Optional[Any]] = {"detector": None}
    detector_lock = threading.Lock()

    def get_detector() -> Any:
        # Web channel management must not pay the model-loading cost. Detection
        # endpoints initialize the model only when they are actually called.
        detector = detector_holder["detector"]
        if detector is None:
            with detector_lock:
                detector = detector_holder["detector"]
                if detector is None:
                    detector = algorithm_registry.create(
                        ALGORITHM_CODE,
                        core_factory=MachineryHazardDetector,
                    )
                    detector_holder["detector"] = detector
        return detector

    def get_task_detector(algorithm_codes: List[str]) -> Any:
        machinery_runner = get_detector() if ALGORITHM_CODE in algorithm_codes else None
        return create_task_algorithm_runner(
            algorithm_codes=algorithm_codes,
            machinery_runner=machinery_runner,
        )

    channel_store = VideoChannelStore()
    app.state.video_channel_store = channel_store
    task_store = VideoTaskStore()
    app.state.video_task_store = task_store
    shared_stream_manager = SharedStreamManager(
        store=channel_store,
        resolve_capture_source=resolve_capture_source,
        make_camera_source=make_camera_index_code_source,
    )
    app.state.shared_stream_manager = shared_stream_manager
    register_video_channel_routes(
        app,
        store=channel_store,
        resolve_capture_source=resolve_capture_source,
        make_camera_source=make_camera_index_code_source,
        delete_guard=task_store.channel_delete_block_reason,
        shared_stream_manager=shared_stream_manager,
    )
    preview_manager = LivePreviewManager(
        store=channel_store,
        resolve_capture_source=resolve_capture_source,
        make_camera_source=make_camera_index_code_source,
        stream_manager=shared_stream_manager,
    )
    app.state.live_preview_manager = preview_manager
    register_live_preview_routes(app, preview_manager, channel_store)
    app.add_event_handler("shutdown", preview_manager.stop_all)
    alarm_index_store = VideoAlarmStore(max_records=int(config.ALARM_RETENTION_MAX_EVENTS))
    app.state.video_alarm_store = alarm_index_store
    alarm_review_manager = LiveAlarmReviewManager(alarm_index_store)
    app.state.alarm_review_manager = alarm_review_manager
    task_runtime = TaskExecutionManager(
        task_store=task_store,
        channel_store=channel_store,
        detector_factory=get_task_detector,
        make_camera_source=make_camera_index_code_source,
        alarm_store=alarm_index_store,
        stream_manager=shared_stream_manager,
        review_manager=alarm_review_manager,
    )
    app.state.task_execution_manager = task_runtime
    register_video_task_routes(app, task_store, runtime=task_runtime)
    register_video_alarm_routes(app, alarm_index_store)
    manual_review_store = ManualReviewStore(alarm_index_store)
    app.state.manual_review_store = manual_review_store
    register_manual_review_routes(app, manual_review_store)
    model_training_manager = ModelTrainingManager(active_task_provider=task_runtime.list_tasks)
    app.state.model_training_manager = model_training_manager
    register_model_training_routes(app, model_training_manager)

    def start_task_services() -> None:
        alarm_index_store.sync_from_output()
        alarm_review_manager.start()
        alarm_index_store.report_pending()
        task_runtime.start_saved_tasks()

    app.add_event_handler("startup", start_task_services)
    app.add_event_handler("shutdown", task_runtime.stop_all)
    app.add_event_handler("shutdown", alarm_review_manager.stop)
    app.add_event_handler("shutdown", shared_stream_manager.stop_all)
    app.add_event_handler("shutdown", model_training_manager.stop_all)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "motion_filter_version": config.MOTION_FILTER_VERSION,
            "model_path": str(config.MODEL_PATH),
            "model_exists": Path(config.MODEL_PATH).exists(),
            "model_loaded": detector_holder["detector"] is not None,
            "video_channel_count": len(channel_store.list_channels()),
            "video_task_count": len(task_store.list_tasks()),
            "alarm_review": alarm_review_manager.status(),
        }

    @app.post("/detect/configured")
    def detect_configured() -> Dict[str, Any]:
        try:
            return process_configured_sources()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/detect/path")
    def detect_path(payload: Dict[str, Any]) -> Dict[str, Any]:
        source = str(payload.get("source", "")).strip().strip('"')
        if not source:
            raise HTTPException(status_code=400, detail='请求体需要提供 "source"')
        try:
            return get_detector().process_source(source)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/detect/image")
    async def detect_image(request: Request, filename: str = "upload.jpg") -> Dict[str, Any]:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="请求体为空，请用二进制方式上传图片内容")
        try:
            image = decode_image_bytes(data)
            return get_detector().process_image_array(image, source_name=filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def run_http_service(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("HTTP 服务模式需要安装 uvicorn，或改用命令行 --source 模式。") from exc

    uvicorn.run(create_app(), host=host, port=port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工程机械伤害检测服务")
    parser.add_argument("--source", "-s", default=config.DEFAULT_SOURCE, help="图片/视频路径，或摄像头编号，如 0")
    parser.add_argument("--config-sources", action="store_true", help="按 config.py 中的图片地址和视频流地址运行")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--host", default=config.SERVER_HOST, help="HTTP 服务监听地址")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="HTTP 服务监听端口")
    parser.add_argument("--show", action="store_true", default=config.SHOW_WINDOW, help="视频检测时显示实时窗口")
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=config.SAVE_ANNOTATED_VIDEO,
        help="视频检测时保存带报警标注的视频",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        run_http_service(args.host, args.port)
        return

    source = str(args.source).strip().strip('"')
    if not source:
        if args.config_sources or has_configured_sources():
            result = process_configured_sources(show=args.show, save_video=args.save_video)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        raise SystemExit("请通过 --source 指定图片、视频或摄像头编号，或在 config.py 中配置图片/视频流地址。")

    if args.config_sources:
        result = process_configured_sources(show=args.show, save_video=args.save_video)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    detector = MachineryHazardDetector()
    result = detector.process_source(source, show=args.show, save_video=args.save_video)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
