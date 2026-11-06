"""Polar range-scan representation shared by the sensor and VFH+."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import Pose2D, wrap_angle


def sector_angles(n_sectors: int) -> np.ndarray:
    """Centres of ``n_sectors`` equal angular bins covering [-pi, pi)."""

    if n_sectors < 8:
        raise ValueError("n_sectors must be at least 8")
    width = 2.0 * math.pi / n_sectors
    return -math.pi + (np.arange(n_sectors, dtype=float) + 0.5) * width


def angle_to_sector(angle: float, n_sectors: int) -> int:
    width = 2.0 * math.pi / n_sectors
    return int(math.floor((wrap_angle(angle) + math.pi) / width)) % n_sectors


def sector_center(index: int, n_sectors: int) -> float:
    width = 2.0 * math.pi / n_sectors
    return wrap_angle(-math.pi + (index + 0.5) * width)


@dataclass(frozen=True, slots=True)
class Scan:
    ranges: np.ndarray
    angles: np.ndarray
    max_range: float
    pose: Pose2D | None = None

    def __post_init__(self) -> None:
        if self.ranges.ndim != 1 or self.angles.ndim != 1:
            raise ValueError("ranges and angles must be 1-D")
        if self.ranges.shape != self.angles.shape:
            raise ValueError("ranges and angles must have the same shape")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")

    @property
    def n_sectors(self) -> int:
        return int(self.ranges.size)

    def normalized(self) -> np.ndarray:
        return np.clip(self.ranges / self.max_range, 0.0, 1.0)
