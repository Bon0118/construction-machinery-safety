"""Thread-safe algorithm metadata and factory registry."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Tuple

from algorithms.base import AlgorithmDescriptor, DetectionAlgorithm


AlgorithmFactory = Callable[..., DetectionAlgorithm]


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, Tuple[AlgorithmDescriptor, AlgorithmFactory]] = {}

    def register(
        self,
        descriptor: AlgorithmDescriptor,
        factory: AlgorithmFactory,
        replace: bool = False,
    ) -> None:
        code = str(descriptor.code or "").strip()
        if not code:
            raise ValueError("算法编号不能为空")
        with self._lock:
            if code in self._entries and not replace:
                raise ValueError(f"算法已经注册：{code}")
            self._entries[code] = (descriptor, factory)

    def descriptor(self, code: str) -> AlgorithmDescriptor:
        normalized = str(code or "").strip()
        with self._lock:
            entry = self._entries.get(normalized)
        if entry is None:
            raise KeyError(f"未注册的算法：{normalized}")
        return entry[0]

    def create(self, code: str, **kwargs: Any) -> DetectionAlgorithm:
        normalized = str(code or "").strip()
        with self._lock:
            entry = self._entries.get(normalized)
        if entry is None:
            raise KeyError(f"未注册的算法：{normalized}")
        return entry[1](**kwargs)

    def list_descriptors(self) -> List[AlgorithmDescriptor]:
        with self._lock:
            return [entry[0] for entry in self._entries.values()]

    def list_public(self) -> List[Dict[str, Any]]:
        return [descriptor.to_dict() for descriptor in self.list_descriptors()]
