"""VFH+ local guidance layer.

The global planner still decides *where* to go.  VFH+ only adjusts the preferred
heading produced by the path tracker, using a polar obstacle histogram.  This is
the intended role of VFH+ in the V3.0 ablation: global RRT* + pure pursuit, with
and without local obstacle avoidance.

References
----------
Borenstein, J. & Koren, Y. (1991). The Vector Field Histogram -- fast obstacle
avoidance for mobile robots. *IEEE Transactions on Robotics and Automation*,
7(3), 278-288.

Ulrich, I. & Borenstein, J. (1998). VFH+: reliable obstacle avoidance for fast
mobile robots. *Proceedings of the IEEE ICRA*, 1042-1047.

Implementation choices
----------------------
* The polar histogram is ``H(k) = 1 - d_k / d_max`` clipped to [0, 1].
* A two-threshold hysteresis rule produces the binary histogram.
* Obstacle sectors are widened by ``atan((r + margin) / d)`` to account for the
  vessel radius; this is a conservative circular-footprint version of the
  VFH+ masking step.
* The speed factor is a linear function of the nearest return inside the
  angular cone subtended by the hull at ``speed_cone_distance``.  Tying the
  cone to the braking distance (a wider cone) slowed the vessel for obstacles
  it would pass beside; the narrow cone removes that artificial penalty while
  keeping the distance-based braking law.
* Every free sector is a candidate.  The cost combines alignment with the
  preferred heading, heading continuity, obstacle density and distance to the
  centre of the free valley.  This avoids the abrupt direction changes of a
  pure "nearest valley" rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.scan import Scan
from ..core.types import wrap_angle

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class VFHConfig:
    n_sectors: int = 72
    max_range: float = 3.0
    threshold_high: float = 0.45
    threshold_low: float = 0.20
    robot_radius: float = 0.09
    safety_margin: float = 0.02
    w_target: float = 2.0
    w_smooth: float = 0.5
    w_density: float = 1.0
    w_valley: float = 0.30
    slow_distance: float = 1.00
    stop_distance: float = 0.25
    speed_cone_distance: float = 0.80
    slow_cone_half_angle: float = 1.05
    min_speed_factor: float = 0.25

    def __post_init__(self) -> None:
        if self.n_sectors < 8:
            raise ValueError("n_sectors must be at least 8")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if not 0.0 <= self.threshold_low < self.threshold_high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        if self.robot_radius < 0.0 or self.safety_margin < 0.0:
            raise ValueError("radii must be non-negative")
        if self.slow_distance <= self.stop_distance:
            raise ValueError("slow_distance must exceed stop_distance")
        if self.speed_cone_distance is not None and self.speed_cone_distance <= 0.0:
            raise ValueError("speed_cone_distance must be positive")
        if not 0.0 < self.slow_cone_half_angle <= math.pi:
            raise ValueError("slow_cone_half_angle must be in (0, pi]")

    @property
    def sector_width(self) -> float:
        return 2.0 * math.pi / self.n_sectors


@dataclass(slots=True)
class VFHResult:
    heading: float
    speed_factor: float
    blocked: bool
    sector: int
    valley_center: float | None
    density: float
    histogram: np.ndarray
    binary: np.ndarray
    masked: np.ndarray
    candidate_costs: np.ndarray = field(repr=False)


class VFHPlus:
    def __init__(self, config: VFHConfig | None = None) -> None:
        self.config = config or VFHConfig()
        self._prev_binary: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_binary = None

    def _binary_histogram(self, histogram: np.ndarray) -> np.ndarray:
        c = self.config
        if self._prev_binary is None or self._prev_binary.shape != histogram.shape:
            binary = (histogram > c.threshold_high).astype(np.int8)
        else:
            binary = self._prev_binary.copy()
            binary[histogram > c.threshold_high] = 1
            binary[histogram < c.threshold_low] = 0
        self._prev_binary = binary.copy()
        return binary

    def _mask(self, binary: np.ndarray, scan: Scan) -> np.ndarray:
        c = self.config
        n = c.n_sectors
        masked = binary.astype(bool).copy()
        half_width = c.robot_radius + c.safety_margin
        for k in range(n):
            if not binary[k]:
                continue
            d = float(scan.ranges[k])
            if not math.isfinite(d) or d <= _EPS:
                d = _EPS
            gamma = math.atan2(half_width, max(d, _EPS))
            span = int(math.ceil(gamma / c.sector_width))
            for offset in range(-span, span + 1):
                masked[(k + offset) % n] = True
        return masked

    @staticmethod
    def _valleys(free: np.ndarray) -> list[list[int]]:
        n = int(free.size)
        if free.all():
            return [list(range(n))]
        if not free.any():
            return []
        # Find starts of circular free runs.
        starts = [i for i in range(n) if free[i] and not free[(i - 1) % n]]
        valleys: list[list[int]] = []
        for start in starts:
            valley: list[int] = []
            i = start
            while free[i] and len(valley) < n:
                valley.append(i)
                i = (i + 1) % n
            if valley:
                valleys.append(valley)
        return valleys

    def _valley_centers(
        self,
        free: np.ndarray,
        preferred_heading: float,
        scan: Scan,
    ) -> np.ndarray:
        c = self.config
        n = c.n_sectors
        centers = np.full(n, np.nan, dtype=float)
        valleys = self._valleys(free)
        if len(valleys) == 1 and len(valleys[0]) == n:
            centers[:] = preferred_heading
            return centers
        for valley in valleys:
            middle = valley[len(valley) // 2]
            centre = float(scan.angles[middle])
            for idx in valley:
                centers[idx] = centre
        return centers

    def guidance(
        self,
        preferred_heading: float,
        scan: Scan,
        previous_heading: float | None = None,
    ) -> VFHResult:
        c = self.config
        if scan.n_sectors != c.n_sectors:
            raise ValueError(
                f"scan has {scan.n_sectors} sectors, VFH+ expects {c.n_sectors}"
            )
        if abs(scan.max_range - c.max_range) > 1e-9:
            raise ValueError("scan max_range does not match VFH+ configuration")

        preferred_heading = wrap_angle(preferred_heading)
        histogram = np.clip(1.0 - scan.ranges / c.max_range, 0.0, 1.0)
        binary = self._binary_histogram(histogram)
        masked = self._mask(binary, scan)
        free = ~masked
        centers = self._valley_centers(free, preferred_heading, scan)

        if not free.any():
            heading = previous_heading if previous_heading is not None else preferred_heading
            return VFHResult(
                heading=wrap_angle(heading),
                speed_factor=0.0,
                blocked=True,
                sector=-1,
                valley_center=None,
                density=float(histogram.max()),
                histogram=histogram,
                binary=binary,
                masked=masked,
                candidate_costs=np.full(c.n_sectors, np.inf),
            )

        costs = np.full(c.n_sectors, np.inf, dtype=float)
        for k in range(c.n_sectors):
            if not free[k]:
                continue
            angle = float(scan.angles[k])
            cost = c.w_target * abs(wrap_angle(angle - preferred_heading))
            if previous_heading is not None:
                cost += c.w_smooth * abs(wrap_angle(angle - previous_heading))
            cost += c.w_density * float(histogram[k])
            centre = centers[k]
            if math.isfinite(centre):
                cost += c.w_valley * abs(wrap_angle(angle - centre))
            costs[k] = cost

        best = int(np.argmin(costs))
        if not math.isfinite(costs[best]):  # pragma: no cover - free.any() guarantees
            best = int(np.argmin(histogram))
            blocked = True
            speed_factor = 0.0
        else:
            blocked = False
            # Reduce speed according to the closest return inside a forward
            # cone around the *current vessel heading*.  The chosen VFH sector
            # may already point away from the obstacle, but the hull still has
            # to rotate and carry its momentum; looking only around the chosen
            # sector reacted too late in the unmapped-block stress case.  The
            # cone is deliberately wider than the hull footprint and is
            # independent of the braking distance.
            heading_now = (
                float(scan.pose.theta)
                if scan.pose is not None
                else float(preferred_heading)
            )
            angular_window = np.abs(
                np.array(
                    [
                        wrap_angle(float(angle) - heading_now)
                        for angle in scan.angles
                    ],
                    dtype=float,
                )
            ) <= c.slow_cone_half_angle
            if np.any(angular_window):
                d_front = float(np.min(scan.ranges[angular_window]))
            else:  # pragma: no cover - 72 sectors always cover the cone
                d_front = c.max_range
            if not math.isfinite(d_front):
                d_front = c.max_range
            distance_factor = (d_front - c.stop_distance) / (
                c.slow_distance - c.stop_distance
            )
            density_factor = 1.0 - float(histogram[best])
            speed_factor = min(1.0, max(c.min_speed_factor, distance_factor))
            speed_factor = min(
                speed_factor, max(c.min_speed_factor, density_factor)
            )

        heading = float(scan.angles[best])
        return VFHResult(
            heading=heading,
            speed_factor=float(speed_factor),
            blocked=blocked,
            sector=best,
            valley_center=float(centers[best]) if math.isfinite(centers[best]) else None,
            density=float(histogram[best]),
            histogram=histogram,
            binary=binary,
            masked=masked,
            candidate_costs=costs,
        )
