"""Algorithm registry and built-in detector packages."""

from algorithms.registry import AlgorithmRegistry
from algorithms.machinery_hazard.detector import (
    MACHINERY_HAZARD_DESCRIPTOR,
    create_machinery_hazard_algorithm,
)
from algorithms.yolov5_objects.defaults import OBJECT_ALGORITHM_DESCRIPTORS
from algorithms.yolov5_objects.detector import create_object_algorithm


algorithm_registry = AlgorithmRegistry()
algorithm_registry.register(MACHINERY_HAZARD_DESCRIPTOR, create_machinery_hazard_algorithm)
for _descriptor in OBJECT_ALGORITHM_DESCRIPTORS:
    algorithm_registry.register(
        _descriptor,
        lambda _code=_descriptor.code, **kwargs: create_object_algorithm(_code, **kwargs),
    )


__all__ = ["AlgorithmRegistry", "algorithm_registry"]
