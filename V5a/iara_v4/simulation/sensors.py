"""Ideal 2-D range scanner used by the VFH+ layer.

The scan angles are expressed in the **world frame** and rotate with the vessel.
This keeps VFH+ indexing consistent: sector ``k`` always corresponds to
``scan.angles[k]``, regardless of the current heading.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..core.geometry import CircleObstacle, Obstacle, RectangleObstacle
from ..core.scan import Scan, sector_angles
from ..core.types import Pose2D

_EPS = 1e-12


class RangeScanner:
    def __init__(self, n_sectors: int = 72, max_range: float = 3.0) -> None:
        if max_range <= 0.0:
            raise ValueError("max_range must be positive")
        self.n_sectors = int(n_sectors)
        self.max_range = float(max_range)
        self.base_angles = sector_angles(self.n_sectors)

    def _ray_rectangle(
        self, origin: np.ndarray, directions: np.ndarray, ob: RectangleObstacle
    ) -> np.ndarray:
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = directions[:, 0], directions[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1 = (ob.min_x - ox) / dx
            tx2 = (ob.max_x - ox) / dx
            ty1 = (ob.min_y - oy) / dy
            ty2 = (ob.max_y - oy) / dy
        tx_min = np.minimum(tx1, tx2)
        tx_max = np.maximum(tx1, tx2)
        ty_min = np.minimum(ty1, ty2)
        ty_max = np.maximum(ty1, ty2)

        parallel_x = np.abs(dx) < _EPS
        inside_x = (ob.min_x <= ox <= ob.max_x)
        tx_min = np.where(parallel_x, np.where(inside_x, -np.inf, np.inf), tx_min)
        tx_max = np.where(parallel_x, np.where(inside_x, np.inf, -np.inf), tx_max)

        parallel_y = np.abs(dy) < _EPS
        inside_y = (ob.min_y <= oy <= ob.max_y)
        ty_min = np.where(parallel_y, np.where(inside_y, -np.inf, np.inf), ty_min)
        ty_max = np.where(parallel_y, np.where(inside_y, np.inf, -np.inf), ty_max)

        t_enter = np.maximum(tx_min, ty_min)
        t_exit = np.minimum(tx_max, ty_max)
        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit >= 0.0) & (
            t_enter <= self.max_range
        )
        t = np.where(hit, np.maximum(t_enter, 0.0), np.inf)
        return np.minimum(t, self.max_range)

    def _ray_circle(
        self, origin: np.ndarray, directions: np.ndarray, ob: CircleObstacle
    ) -> np.ndarray:
        oc = origin - np.array([ob.cx, ob.cy])
        b = 2.0 * (directions @ oc)
        c = float(oc @ oc) - ob.radius * ob.radius
        disc = b * b - 4.0 * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sqrt_disc) / 2.0
        t1 = (-b + sqrt_disc) / 2.0
        inside = c < 0.0
        t = np.where(inside, 0.0, np.where(t0 >= 0.0, t0, t1))
        hit = (disc >= 0.0) & (t >= 0.0) & (t <= self.max_range)
        return np.where(hit, t, np.inf)

    def scan(self, pose: Pose2D, obstacles: Sequence[Obstacle]) -> Scan:
        theta = pose.theta
        world_angles = theta + self.base_angles
        world_angles = (world_angles + math.pi) % (2.0 * math.pi) - math.pi
        directions = np.column_stack(
            (np.cos(theta + self.base_angles), np.sin(theta + self.base_angles))
        )
        origin = np.array([pose.x, pose.y], dtype=float)
        ranges = np.full(self.n_sectors, self.max_range, dtype=float)
        for ob in obstacles:
            if isinstance(ob, RectangleObstacle):
                t = self._ray_rectangle(origin, directions, ob)
            elif isinstance(ob, CircleObstacle):
                t = self._ray_circle(origin, directions, ob)
            else:  # pragma: no cover - extension point
                continue
            ranges = np.minimum(ranges, t)
        return Scan(
            ranges=np.clip(ranges, 0.0, self.max_range),
            angles=world_angles,
            max_range=self.max_range,
            pose=pose,
        )

    def scan_points(self, scan: Scan) -> np.ndarray:
        if scan.pose is None:
            raise ValueError("scan has no pose")
        dirs = np.column_stack((np.cos(scan.angles), np.sin(scan.angles)))
        origin = np.array([scan.pose.x, scan.pose.y], dtype=float)
        return origin + dirs * scan.ranges[:, None]
