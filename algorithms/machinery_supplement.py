"""Thread-safe bridge from shared object inference to machinery-risk analysis."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


Box = Tuple[int, int, int, int]
SUPPLEMENT_CLASS_MAP = {
    7: (6, "excavator"),
    9: (13, "crane_arm"),
}


@dataclass(frozen=True)
class SupplementalMachineryDetection:
    class_id: int
    class_name: str
    confidence: float
    box: Box
    source_class_id: int


def supplement_key(source: str, event_context: Optional[Mapping[str, Any]]) -> str:
    context = dict(event_context or {})
    task_id = context.get("task_id")
    if task_id not in (None, ""):
        return f"task:{task_id}"
    return f"source:{str(source or '').strip()}"


def map_supplement_detection(item: Any) -> Optional[SupplementalMachineryDetection]:
    source_class_id = int(getattr(item, "class_id"))
    mapped = SUPPLEMENT_CLASS_MAP.get(source_class_id)
    if mapped is None:
        return None
    class_id, class_name = mapped
    return SupplementalMachineryDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=float(getattr(item, "confidence")),
        box=tuple(int(value) for value in getattr(item, "box")),
        source_class_id=source_class_id,
    )


class MachinerySupplementBroker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, Tuple[float, Tuple[SupplementalMachineryDetection, ...]]] = {}

    def publish(
        self,
        key: str,
        detections: Iterable[SupplementalMachineryDetection],
        observed_at: Optional[float] = None,
    ) -> None:
        timestamp = time.monotonic() if observed_at is None else float(observed_at)
        with self._lock:
            self._entries[str(key)] = (timestamp, tuple(detections))

    def latest(
        self,
        key: str,
        max_age_seconds: float,
        now: Optional[float] = None,
    ) -> Tuple[SupplementalMachineryDetection, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            entry = self._entries.get(str(key))
        if entry is None or timestamp - entry[0] > max(0.0, float(max_age_seconds)):
            return ()
        return entry[1]

    def clear(self, key: str) -> None:
        with self._lock:
            self._entries.pop(str(key), None)


MACHINERY_SUPPLEMENT_BROKER = MachinerySupplementBroker()
