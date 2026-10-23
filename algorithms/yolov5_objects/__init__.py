"""YOLOv5 object-detection algorithms backed by the three imported models."""

from algorithms.yolov5_objects.defaults import (
    OBJECT_ALGORITHM_DEFINITIONS,
    OBJECT_ALGORITHM_DESCRIPTORS,
    get_object_algorithm_definition,
    load_object_settings,
)
from algorithms.yolov5_objects.detector import (
    YoloV5ObjectAlgorithm,
    YoloV5ObjectTaskRunner,
    create_object_algorithm,
)
from algorithms.yolov5_objects.settings import ObjectDetectionSettings


__all__ = [
    "OBJECT_ALGORITHM_DEFINITIONS",
    "OBJECT_ALGORITHM_DESCRIPTORS",
    "ObjectDetectionSettings",
    "YoloV5ObjectAlgorithm",
    "YoloV5ObjectTaskRunner",
    "create_object_algorithm",
    "get_object_algorithm_definition",
    "load_object_settings",
]
