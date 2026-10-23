"""Model ownership, class mappings, and defaults for imported YOLOv5 weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from algorithms.base import AlgorithmDescriptor
from algorithms.machinery_hazard.defaults import SCENE_MODE
from algorithms.yolov5_objects.settings import ObjectDetectionSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_VERSION = 3
DEVICE = "0"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECTION_INTERVAL_SECONDS = 0.5
ALARM_COOLDOWN_SECONDS = 30.0
PERSON_ASSOCIATION_EXPAND_RATIO = 0.15
PERSON_ASSOCIATION_UPPER_BODY_RATIO = 0.70


@dataclass(frozen=True)
class ObjectAlgorithmDefinition:
    code: str
    name: str
    description: str
    model_key: str
    model_path: Path
    class_ids: Tuple[int, ...]
    class_labels: Mapping[int, str]
    expected_model_names: Tuple[str, ...]
    alarm_rule: str = "detected"
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    alarm_cooldown_seconds: float = ALARM_COOLDOWN_SECONDS
    person_association_expand_ratio: float = PERSON_ASSOCIATION_EXPAND_RATIO
    person_association_upper_body_ratio: float = PERSON_ASSOCIATION_UPPER_BODY_RATIO
    alarm_window_size: int = 1
    alarm_min_hits: int = 1
    alarm_temporal_iou_threshold: float = 0.0
    public: bool = True


HELMET_MODEL_NAMES = ("person", "head", "helmet")
MARITIME_MODEL_NAMES = (
    "fire",
    "smoke",
    "cigarette",
    "life_jacket_m",
    "life_jacket_s",
    "ship",
    "cellphone",
    "Excavator",
    "Hanging_Basket",
    "CraneArm",
    "Rebar",
    "fainting",
)
ROAD_MODEL_NAMES = ("car", "person", "zhui")

HELMET_MODEL_PATH = PROJECT_ROOT / "models" / "safety_helmet" / "model1_best.pt"
MARITIME_MODEL_PATH = PROJECT_ROOT / "models" / "maritime_safety" / "model2_best.pt"
ROAD_MODEL_PATH = PROJECT_ROOT / "models" / "road_safety" / "car-3cls_best.pt"


def _definition(
    code: str,
    name: str,
    description: str,
    model_key: str,
    model_path: Path,
    class_ids: Tuple[int, ...],
    class_labels: Mapping[int, str],
    expected_model_names: Tuple[str, ...],
    alarm_rule: str = "detected",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    alarm_cooldown_seconds: float = ALARM_COOLDOWN_SECONDS,
    person_association_expand_ratio: float = PERSON_ASSOCIATION_EXPAND_RATIO,
    person_association_upper_body_ratio: float = PERSON_ASSOCIATION_UPPER_BODY_RATIO,
    alarm_window_size: int = 1,
    alarm_min_hits: int = 1,
    alarm_temporal_iou_threshold: float = 0.0,
    public: bool = True,
) -> ObjectAlgorithmDefinition:
    return ObjectAlgorithmDefinition(
        code=code,
        name=name,
        description=description,
        model_key=model_key,
        model_path=model_path,
        class_ids=class_ids,
        class_labels=dict(class_labels),
        expected_model_names=expected_model_names,
        alarm_rule=alarm_rule,
        confidence_threshold=confidence_threshold,
        alarm_cooldown_seconds=alarm_cooldown_seconds,
        person_association_expand_ratio=person_association_expand_ratio,
        person_association_upper_body_ratio=person_association_upper_body_ratio,
        alarm_window_size=alarm_window_size,
        alarm_min_hits=alarm_min_hits,
        alarm_temporal_iou_threshold=alarm_temporal_iou_threshold,
        public=public,
    )


_DEFINITIONS = (
    _definition(
        "safety_helmet",
        "安全帽检测",
        "人员头部未匹配到安全帽时告警",
        "helmet",
        HELMET_MODEL_PATH,
        (0, 1, 2),
        {0: "未佩戴安全帽", 1: "未佩戴安全帽", 2: "安全帽"},
        HELMET_MODEL_NAMES,
        alarm_rule="missing_helmet",
    ),
    _definition("ship_detection", "船舶检测", "识别水域或作业画面中的船舶", "maritime", MARITIME_MODEL_PATH, (5,), {5: "船舶"}, MARITIME_MODEL_NAMES),
    _definition("hanging_basket", "吊篮检测", "识别施工吊篮", "maritime", MARITIME_MODEL_PATH, (8,), {8: "吊篮"}, MARITIME_MODEL_NAMES),
    _definition("life_jacket", "救生衣检测", "识别两类救生衣目标", "maritime", MARITIME_MODEL_PATH, (3, 4), {3: "救生衣", 4: "救生衣"}, MARITIME_MODEL_NAMES),
    _definition("fire_detection", "明火检测", "识别画面中的明火", "maritime", MARITIME_MODEL_PATH, (0,), {0: "明火"}, MARITIME_MODEL_NAMES),
    _definition(
        "cigarette_detection",
        "香烟检测",
        "同一画面同时检测到人员和香烟时告警",
        "maritime",
        MARITIME_MODEL_PATH,
        (2,),
        {0: "人员", 2: "香烟"},
        MARITIME_MODEL_NAMES,
        alarm_rule="person_with_cigarette",
    ),
    _definition("smoke_detection", "烟雾检测", "识别画面中的烟雾", "maritime", MARITIME_MODEL_PATH, (1,), {1: "烟雾"}, MARITIME_MODEL_NAMES),
    _definition(
        "cellphone_usage",
        "玩手机检测",
        "人员上半身关联区域内检测到手机并持续出现时告警",
        "maritime",
        MARITIME_MODEL_PATH,
        (6,),
        {0: "人员", 6: "手机"},
        MARITIME_MODEL_NAMES,
        alarm_rule="person_with_cellphone",
        confidence_threshold=0.30,
        person_association_expand_ratio=0.20,
        person_association_upper_body_ratio=0.85,
        alarm_window_size=3,
        alarm_min_hits=2,
    ),
    _definition(
        "rebar_detection",
        "钢筋检测",
        "钢筋目标持续出现时告警",
        "maritime",
        MARITIME_MODEL_PATH,
        (10,),
        {10: "钢筋"},
        MARITIME_MODEL_NAMES,
        confidence_threshold=0.35,
        alarm_window_size=3,
        alarm_min_hits=2,
        alarm_temporal_iou_threshold=0.20,
    ),
    _definition(
        "fainting_detection",
        "晕倒检测",
        "疑似晕倒目标持续出现时告警",
        "maritime",
        MARITIME_MODEL_PATH,
        (11,),
        {11: "疑似晕倒"},
        MARITIME_MODEL_NAMES,
        confidence_threshold=0.35,
        alarm_cooldown_seconds=60.0,
        alarm_window_size=3,
        alarm_min_hits=2,
        alarm_temporal_iou_threshold=0.20,
    ),
    _definition(
        "machinery_hazard_supplement",
        "机械伤害补充检测",
        ("向起重机场景提供吊臂辅助证据" if SCENE_MODE == "crane_boom_person"
         else "向机械伤害算法提供挖掘机和吊臂补充检测结果"),
        "maritime",
        MARITIME_MODEL_PATH,
        (9,) if SCENE_MODE == "crane_boom_person" else (7, 9),
        {7: "挖掘机", 9: "吊臂"},
        MARITIME_MODEL_NAMES,
        alarm_rule="publish_machinery_supplement",
        public=False,
    ),
    _definition("vehicle_detection", "车辆检测", "识别道路或作业区域中的车辆", "road", ROAD_MODEL_PATH, (0,), {0: "车辆"}, ROAD_MODEL_NAMES),
    _definition("traffic_cone", "交通锥检测", "识别道路交通锥", "road", ROAD_MODEL_PATH, (2,), {2: "交通锥"}, ROAD_MODEL_NAMES),
)

OBJECT_ALGORITHM_DEFINITIONS: Dict[str, ObjectAlgorithmDefinition] = {
    item.code: item for item in _DEFINITIONS
}
OBJECT_ALGORITHM_DESCRIPTORS = tuple(
    AlgorithmDescriptor(
        code=item.code,
        name=item.name,
        config_version=CONFIG_VERSION,
        description=item.description,
    )
    for item in _DEFINITIONS
    if item.public
)


def get_object_algorithm_definition(code: str) -> ObjectAlgorithmDefinition:
    normalized = str(code or "").strip()
    try:
        return OBJECT_ALGORITHM_DEFINITIONS[normalized]
    except KeyError as exc:
        raise KeyError(f"未注册的 YOLOv5 目标检测算法：{normalized}") from exc


def load_object_settings(code: str) -> ObjectDetectionSettings:
    item = get_object_algorithm_definition(code)
    settings = ObjectDetectionSettings(
        algorithm_code=item.code,
        algorithm_name=item.name,
        description=item.description,
        model_key=item.model_key,
        model_path=item.model_path,
        class_ids=item.class_ids,
        class_labels=item.class_labels,
        expected_model_names=item.expected_model_names,
        alarm_rule=item.alarm_rule,
        confidence_threshold=item.confidence_threshold,
        person_confidence_threshold=item.confidence_threshold,
        bare_head_confidence_threshold=item.confidence_threshold,
        helmet_confidence_threshold=item.confidence_threshold,
        iou_threshold=IOU_THRESHOLD,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        detection_interval_seconds=DETECTION_INTERVAL_SECONDS,
        alarm_cooldown_seconds=item.alarm_cooldown_seconds,
        person_association_expand_ratio=item.person_association_expand_ratio,
        person_association_upper_body_ratio=item.person_association_upper_body_ratio,
        alarm_window_size=item.alarm_window_size,
        alarm_min_hits=item.alarm_min_hits,
        alarm_temporal_iou_threshold=item.alarm_temporal_iou_threshold,
        config_version=CONFIG_VERSION,
    )
    settings.validate()
    return settings
