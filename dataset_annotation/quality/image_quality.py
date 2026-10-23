"""Deterministic image-quality screening for alarm originals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityResult:
    accepted: bool
    rejection_reason: str
    width: int
    height: int
    low_saturation_ratio: float
    mean_channel_spread: float
    white_ratio: float
    black_ratio: float
    brightness_mean: float
    brightness_std: float
    blur_variance: float
    perceptual_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def difference_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def hash_distance(first: str, second: str) -> int:
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def load_image(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def evaluate_image(path: Path, quality_config: Dict[str, Any]) -> Tuple[ImageQualityResult, Optional[np.ndarray]]:
    image = load_image(path)
    if image is None or image.ndim != 3:
        return (
            ImageQualityResult(
                accepted=False,
                rejection_reason="decode_failed",
                width=0,
                height=0,
                low_saturation_ratio=1.0,
                mean_channel_spread=0.0,
                white_ratio=0.0,
                black_ratio=0.0,
                brightness_mean=0.0,
                brightness_std=0.0,
                blur_variance=0.0,
                perceptual_hash="",
            ),
            None,
        )

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    low_saturation_ratio = float(np.mean(saturation <= 8))
    channel_spread = image.max(axis=2).astype(np.float32) - image.min(axis=2).astype(np.float32)
    mean_channel_spread = float(np.mean(channel_spread))
    white_threshold = int(quality_config["white_pixel_threshold"])
    black_threshold = int(quality_config["black_pixel_threshold"])
    white_ratio = float(np.mean(np.all(image >= white_threshold, axis=2)))
    black_ratio = float(np.mean(np.all(image <= black_threshold, axis=2)))
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_grey = (
        low_saturation_ratio >= float(quality_config["grey_low_saturation_ratio"])
        and mean_channel_spread <= float(quality_config["grey_mean_channel_spread"])
        and brightness_std <= float(quality_config["grey_max_brightness_std"])
    )
    is_overexposed = white_ratio >= float(quality_config["maximum_white_ratio"])
    reason = "grey_frame" if is_grey else "overexposed_white" if is_overexposed else ""
    result = ImageQualityResult(
        accepted=not bool(reason),
        rejection_reason=reason,
        width=int(width),
        height=int(height),
        low_saturation_ratio=round(low_saturation_ratio, 6),
        mean_channel_spread=round(mean_channel_spread, 3),
        white_ratio=round(white_ratio, 6),
        black_ratio=round(black_ratio, 6),
        brightness_mean=round(brightness_mean, 3),
        brightness_std=round(brightness_std, 3),
        blur_variance=round(blur_variance, 3),
        perceptual_hash=difference_hash(image),
    )
    return result, image
