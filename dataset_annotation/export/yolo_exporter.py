"""Write YOLO labels, rich JSON annotations, and preview images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import cv2
import numpy as np

from dataset_annotation.annotation.rules import PrelabelBox


COLORS = {
    "person": (255, 160, 0),
    "head": (0, 0, 255),
    "helmet": (0, 200, 0),
}


def _yolo_line(item: PrelabelBox, width: int, height: int) -> str:
    x1, y1, x2, y2 = item.box
    center_x = (x1 + x2) / 2.0 / width
    center_y = (y1 + y2) / 2.0 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{item.class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def write_yolo(path: Path, labels: Iterable[PrelabelBox], width: int, height: int) -> None:
    content = "\n".join(_yolo_line(item, width, height) for item in labels)
    path.write_text(content + ("\n" if content else ""), encoding="ascii")


def write_annotation_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_preview(path: Path, image: np.ndarray, labels: Sequence[PrelabelBox], title: str) -> None:
    preview = image.copy()
    for item in labels:
        color = COLORS[item.class_name]
        x1, y1, x2, y2 = item.box
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        text = f"{item.class_name} {item.confidence:.2f}"
        cv2.putText(preview, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(preview, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError(f"无法编码预览图：{path}")
    encoded.tofile(str(path))

