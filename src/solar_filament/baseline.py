"""Fast full-resolution classical vision baseline for a first valid submission."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineConfig:
    contrast_threshold: float = 14.0
    background_sigma: float = 24.0
    min_area: int = 250
    max_area: int = 40000
    limb_margin: int = 65
    close_size: int = 3


def solar_disk_mask(image: np.ndarray, margin: int) -> np.ndarray:
    """Find the bright solar disk and remove the high-contrast limb."""
    blurred = cv2.GaussianBlur(image, (0, 0), 12)
    threshold = max(35, int(np.percentile(blurred, 50) * 0.45))
    foreground = (blurred > threshold).astype(np.uint8)
    count, component, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count < 2:
        raise ValueError("Solar disk could not be located")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (component == main).astype(np.uint8)
    distance = cv2.distanceTransform(disk, cv2.DIST_L2, 5)
    return distance > margin


def predict_labels(image: np.ndarray, config: BaselineConfig) -> np.ndarray:
    """Detect dark local structures, join tiny gaps, and label components."""
    disk = solar_disk_mask(image, config.limb_margin)
    background = cv2.GaussianBlur(image, (0, 0), config.background_sigma)
    contrast = background.astype(np.float32) - image.astype(np.float32)
    binary = ((contrast >= config.contrast_threshold) & disk).astype(np.uint8)
    if config.close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_size, config.close_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return components_to_labels(binary, config.min_area, config.max_area)


def components_to_labels(binary: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Filter instance components and assign contiguous positive IDs."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    valid = np.flatnonzero(
        (stats[:, cv2.CC_STAT_AREA] >= min_area)
        & (stats[:, cv2.CC_STAT_AREA] <= max_area)
    )
    valid = valid[valid != 0]
    lookup = np.zeros(count, dtype=np.int32)
    lookup[valid] = np.arange(1, len(valid) + 1)
    return lookup[components]
