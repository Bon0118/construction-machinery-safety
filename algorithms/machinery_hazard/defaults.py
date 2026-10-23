"""Default values owned exclusively by machinery-hazard detection."""

from __future__ import annotations

from pathlib import Path

from algorithms.machinery_hazard.settings import MachineryHazardSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_CODE = "machinery_hazard"
ALGORITHM_NAME = "机械伤害预警"
CONFIG_VERSION = 5

# Model and inference.
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
CONFIDENCE_THRESHOLD = 0.25
PERSON_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.70
IMAGE_SIZE = 640
# Use the first NVIDIA GPU. The yolov5 environment contains the matching
# CUDA-enabled PyTorch build and this host exposes one RTX 4060 as device 0.
DEVICE = "0"
MAX_DETECTIONS = 300
CLASS_NAMES = {
    0: "person",
    1: "tower_crane",
    2: "hook",
    3: "mobile_crane",
    4: "road_roller",
    5: "bulldozer",
    6: "excavator",
    7: "dump_truck",
    8: "loader",
    9: "concrete_pump_truck",
    10: "concrete_mixer_truck",
    11: "pile_driver",
    12: "other_construction_vehicle",
    13: "crane_arm",
}
PERSON_CLASS_IDS = {0}
# dump_truck (7) is excluded from machinery-hazard inference.
# Switch back to legacy_machinery to restore the preserved broad hazard path.
SCENE_MODE = "crane_boom_person"
LEGACY_MACHINERY_CLASS_IDS = {1, 3, 4, 5, 6, 8, 9, 10, 11}
MACHINERY_CLASS_IDS = {1, 3} if SCENE_MODE == "crane_boom_person" else LEGACY_MACHINERY_CLASS_IDS
CRANE_CANDIDATE_HORIZONTAL_MARGIN = 0.15
CRANE_REGION_MARGIN_RATIO = 0.12
CRANE_ASSOCIATION_SCORE_MARGIN = 0.15
CRANE_ASSOCIATION_MIN_SCORE = 0.45
CRANE_AUXILIARY_MIN_CONFIDENCE = 0.40
# Hook is evidence for static-operation review, not a standalone risk machine.
PREDICT_CLASS_IDS = sorted(PERSON_CLASS_IDS | MACHINERY_CLASS_IDS | {2})
# These canonical IDs come from the shared model2 object inference. They are
# deliberately excluded from PREDICT_CLASS_IDS because best.pt has no class 13.
SUPPLEMENTAL_MACHINERY_CLASS_IDS = {6, 13}
MACHINERY_SUPPLEMENT_MAX_AGE_SECONDS = 1.5
MACHINERY_SUPPLEMENT_DEDUP_IOU_THRESHOLD = 0.50
DETECTION_INTERVAL_SECONDS = 0.5

# Machinery tracking and motion state.
MACHINERY_MOTION_ENABLED = True
MACHINERY_TRACK_IOU_THRESHOLD = 0.15
MACHINERY_TRACK_MAX_CENTER_DISTANCE_RATIO = 0.60
MACHINERY_TRACK_LOST_SECONDS = 2.0
MACHINERY_BOX_SMOOTHING_ALPHA = 0.25
MACHINERY_MOTION_WINDOW_SECONDS = 3.0
MACHINERY_MOTION_MIN_SAMPLES = 4
MACHINERY_BODY_STEP_THRESHOLD_RATIO = 0.004
MACHINERY_MOVING_THRESHOLD_RATIO = 0.025
MACHINERY_MOVING_CONFIRM_COUNT = 3
MACHINERY_STOP_CONFIRM_SECONDS = 3.0
LOG_MACHINERY_MOTION = True
MOTION_FILTER_VERSION = "v2-visual-body-and-static-review"
MACHINERY_BOX_MAX_SCALE_CHANGE = 1.25
MACHINERY_BOX_MAX_CENTER_STEP = 0.15
MACHINERY_NET_MOVING_THRESHOLD_RATIO = 0.020
MACHINERY_DIRECTION_CONSISTENCY = 0.65
BODY_VISUAL_MIN_POINTS = 8
BODY_VISUAL_MIN_SUPPORT = 0.60
BODY_VISUAL_MIN_CELLS = 3
BODY_VISUAL_MIN_STEP_PIXELS = 0.20
BODY_VISUAL_MIN_NET_RATIO = 0.008
BODY_VISUAL_MIN_VOTES = 3
BODY_VISUAL_MAX_AGE_SECONDS = 0.8
CAMERA_MIN_INLIER_RATIO = 0.65
CAMERA_MAX_RESIDUAL_PIXELS = 1.5
STATIC_OPERATION_REVIEW_ENABLED = True
STATIC_OPERATION_REVIEW_CLASS_IDS = {1, 3, 6, 8, 9, 11, 13}
STATIC_OPERATION_EVIDENCE_CLASS_IDS = {2, 13}
STATIC_OPERATION_EVIDENCE_CONFIDENCE = 0.40
STATIC_OPERATION_MIN_TRACK_SECONDS = 2.0
STATIC_OPERATION_MAX_FRAME_AREA = 0.45

# Excavator arm optical-flow assessment.
ARM_MOTION_CLASS_IDS = {6}
ARM_FLOW_INTERVAL_SECONDS = 0.2
ARM_ROI_EXPAND_RATIO = 0.08
ARM_BODY_EXCLUDE_BOTTOM_RATIO = 0.38
ARM_PERSON_MASK_EXPAND_RATIO = 0.10
ARM_FLOW_ROI_MAX_SIZE = 320
ARM_FLOW_MIN_VALID_PIXELS = 80
ARM_FLOW_MIN_SPEED_RATIO = 0.025
ARM_FLOW_MIN_MOVING_PIXEL_RATIO = 0.03
ARM_FLOW_MIN_COMPONENT_RATIO = 0.01
ARM_FLOW_WINDOW_SIZE = 5
ARM_FLOW_CONFIRM_COUNT = 3
ARM_FLOW_STOP_SECONDS = 2.0
ARM_FLOW_MAX_BRIGHTNESS_DELTA = 35.0
ARM_GLOBAL_MOTION_COMPENSATION = True
ARM_GLOBAL_MOTION_MIN_POINTS = 12

# Rider suppression.
RIDER_FILTER_ENABLED = True
RIDER_FILTER_CLASS_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
RIDER_CABIN_ZONES = {
    3: (0.05, 0.00, 0.95, 0.72),
    4: (0.10, 0.00, 0.90, 0.72),
    5: (0.05, 0.00, 0.95, 0.72),
    6: (0.05, 0.00, 0.95, 0.72),
    7: (0.00, 0.00, 1.00, 0.72),
    8: (0.05, 0.00, 0.95, 0.72),
    9: (0.00, 0.00, 1.00, 0.72),
    10: (0.00, 0.00, 1.00, 0.72),
    11: (0.05, 0.00, 0.95, 0.72),
    12: (0.05, 0.00, 0.95, 0.72),
}
RIDER_DEFAULT_CABIN_ZONE = (0.05, 0.00, 0.95, 0.72)
RIDER_PERSON_CONTAINMENT_THRESHOLD = 0.45
RIDER_PERSON_BOTTOM_MAX_RATIO = 0.88
RIDER_RELATIVE_POSITION_THRESHOLD = 0.10
RIDER_CONFIRM_WINDOW_SIZE = 4
RIDER_CONFIRM_COUNT = 3
RIDER_EXIT_SECONDS = 1.0
PERSON_TRACK_IOU_THRESHOLD = 0.20
PERSON_TRACK_MAX_CENTER_DISTANCE_RATIO = 0.80
PERSON_TRACK_LOST_SECONDS = 1.5

# Risk-zone and alarm trigger rules.
RISK_ZONE_CLASS_SCALES = {
    1: (1.10, 1.05),
    3: (1.25, 1.05),
    4: (1.15, 0.95),
    5: (1.20, 1.00),
    6: (1.40, 1.20),
    7: (1.20, 1.00),
    8: (1.25, 1.05),
    9: (1.20, 1.00),
    10: (1.20, 1.00),
    11: (1.25, 1.10),
    12: (1.15, 1.00),
    13: (1.35, 1.20),
}
RISK_ZONE_CENTER_Y_RATIO = 0.68
RISK_ZONE_USE_MOTION_DIRECTION = True
RISK_ZONE_REACTION_TIME_SECONDS = 1.2
RISK_ZONE_MAX_FORWARD_EXTENSION_RATIO = 0.75
PERSON_RISK_LOWER_BODY_RATIO = 0.40
BUFFER_SHAPE = "ellipse"
BUFFER_SCALE = 1.10
OVERLAP_RATIO_THRESHOLD = 0.01
RISK_CONSECUTIVE_FRAMES = 1
ALARM_SAVE_COOLDOWN_SECONDS = 30.0

# Machinery-hazard alarm rendering.
ALARM_CLASS_LABELS = {
    0: "人员",
    1: "塔式起重机",
    2: "吊钩",
    3: "移动式起重机",
    4: "压路机",
    5: "推土机",
    6: "挖掘机",
    7: "自卸车",
    8: "铲车",
    9: "混凝土泵车",
    10: "混凝土搅拌车",
    11: "打桩机",
    12: "其他工程车辆",
    13: "起重机吊臂",
}
ALARM_RISK_TEXT = "吊装下禁止站人" if SCENE_MODE == "crane_boom_person" else "注意机械伤害风险"
ALARM_TITLE_FONT_SIZE = 30
ALARM_INFO_BAR_HEIGHT = 96
ALARM_INFO_BAR_ALPHA = 0.72
ALARM_TITLE_TOP_MARGIN = 12
ALARM_PERSON_COLOR_BGR = (0, 0, 255)
ALARM_MACHINERY_COLOR_BGR = (0, 0, 255)
ALARM_RISK_BOX_ALPHA = 0.30
ALARM_RISK_BOX_BORDER_THICKNESS = 2
ALARM_DRAW_DEDUP_IOU_THRESHOLD = 0.65
ALARM_DRAW_DEDUP_CONTAINMENT_THRESHOLD = 0.85
ALARM_COLOR_BGR = (0, 0, 255)
DRAW_LINE_THICKNESS = 5
DRAW_PERSON_BOX = True
DRAW_BUFFER = False
DRAW_ALL_MACHINERY_BUFFERS = True
DRAW_MACHINERY_BOX = True
DRAW_LABELS = True
DRAW_ALARM_TEXT = False
DISPLAY_WINDOW_NAME = "Machinery Hazard Detection"


def _advanced_defaults():
    core_names = {
        "MODEL_PATH",
        "CONFIDENCE_THRESHOLD",
        "PERSON_CONFIDENCE_THRESHOLD",
        "IOU_THRESHOLD",
        "IMAGE_SIZE",
        "DEVICE",
        "MAX_DETECTIONS",
        "PERSON_CLASS_IDS",
        "MACHINERY_CLASS_IDS",
        "DETECTION_INTERVAL_SECONDS",
        "OVERLAP_RATIO_THRESHOLD",
        "RISK_CONSECUTIVE_FRAMES",
        "ALARM_SAVE_COOLDOWN_SECONDS",
        "MACHINERY_MOTION_ENABLED",
        "RIDER_FILTER_ENABLED",
    }
    excluded = core_names | {
        "PROJECT_ROOT",
        "ALGORITHM_CODE",
        "ALGORITHM_NAME",
        "CONFIG_VERSION",
        "PREDICT_CLASS_IDS",
    }
    return {
        name.lower(): value
        for name, value in globals().items()
        if name.isupper() and name not in excluded
    }


def load_default_settings() -> MachineryHazardSettings:
    settings = MachineryHazardSettings(
        model_path=MODEL_PATH,
        confidence_threshold=float(CONFIDENCE_THRESHOLD),
        person_confidence_threshold=float(PERSON_CONFIDENCE_THRESHOLD),
        iou_threshold=float(IOU_THRESHOLD),
        image_size=int(IMAGE_SIZE),
        device=str(DEVICE or ""),
        max_detections=int(MAX_DETECTIONS),
        person_class_ids=tuple(sorted(int(value) for value in PERSON_CLASS_IDS)),
        machinery_class_ids=tuple(sorted(int(value) for value in MACHINERY_CLASS_IDS)),
        detection_interval_seconds=float(DETECTION_INTERVAL_SECONDS),
        overlap_ratio_threshold=float(OVERLAP_RATIO_THRESHOLD),
        risk_consecutive_frames=int(RISK_CONSECUTIVE_FRAMES),
        alarm_cooldown_seconds=float(ALARM_SAVE_COOLDOWN_SECONDS),
        machinery_motion_enabled=bool(MACHINERY_MOTION_ENABLED) and SCENE_MODE != "crane_boom_person",
        rider_filter_enabled=bool(RIDER_FILTER_ENABLED) and SCENE_MODE != "crane_boom_person",
        advanced=_advanced_defaults(),
        config_version=CONFIG_VERSION,
    )
    settings.validate()
    return settings
