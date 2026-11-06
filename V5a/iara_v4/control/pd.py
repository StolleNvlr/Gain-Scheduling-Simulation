"""Corrected path-tracking PD baseline.

This is *not* the main controller.  It exists so the paper can compare pure
pursuit against a conventional cross-track/heading PD law on exactly the same
paths and seeds.  The V2.0 implementation applied coefficients discretised for
``Ts = 1 s`` at ``dt = 0.05 s`` and reset its filter at every waypoint; neither
is done here.  The law is continuous-time and the simulator integrates it with
the exact constant-twist map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.kinematics import DifferentialDrive
from ..core.path import Path, PathProjection
from ..core.types import ControlCommand, Pose2D, Vec2, wrap_angle


@dataclass(frozen=True, slots=True)
class PDConfig:
    v_max: float = 0.45
    v_min: float = 0.05
    a_lat_max: float = 0.80
    a_dec_max: float = 0.50
    goal_tolerance: float = 0.08
    k_y: float = 2.0
    k_theta: float = 2.5
    lookahead_speed: float = 0.30

    def __post_init__(self) -> None:
        if self.v_max <= 0.0 or self.v_min < 0.0:
            raise ValueError("speed bounds are invalid")
        if self.a_lat_max <= 0.0 or self.a_dec_max <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be non-negative")


class PathPD:
    def __init__(
        self,
        config: PDConfig | None = None,
        drive: DifferentialDrive | None = None,
    ) -> None:
        self.config = config or PDConfig()
        self.drive = drive or DifferentialDrive()

    def speed_profile(self, pose: Pose2D, path: Path, projection: PathProjection) -> float:
        c = self.config
        distance_to_goal = pose.position.distance_to(path.end)
        if distance_to_goal <= c.goal_tolerance:
            return 0.0
        s0 = projection.s
        s1 = min(path.length, s0 + c.lookahead_speed)
        samples = np.linspace(s0, s1, 6)
        kappa = max(abs(path.curvature_at(float(s))) for s in samples)
        kappa = max(kappa, 1e-6)
        v_curve = math.sqrt(c.a_lat_max / kappa)
        v_goal = math.sqrt(
            2.0 * c.a_dec_max * max(0.0, distance_to_goal - c.goal_tolerance)
        )
        v = min(c.v_max, v_curve, v_goal, self.drive.max_speed_for_curvature(kappa))
        if v < c.v_min and distance_to_goal > c.goal_tolerance:
            v = min(c.v_min, self.drive.max_speed_for_curvature(kappa))
        return max(0.0, v)

    def command(
        self,
        pose: Pose2D,
        path: Path,
        projection: PathProjection | None = None,
    ) -> ControlCommand:
        projection = projection or path.project(pose.position)
        v = self.speed_profile(pose, path, projection)
        if v <= 0.0:
            return self.drive.command(0.0, 0.0, target=path.end)

        tangent = path.tangent_at(projection.s)
        normal = Vec2(-tangent.y, tangent.x)
        e_y = (pose.position - projection.point).dot(normal)
        e_theta = wrap_angle(pose.theta - path.heading_at(projection.s))
        omega = -self.config.k_y * e_y - self.config.k_theta * e_theta
        return self.drive.command(v, omega, target=path.point_at(
            min(path.length, projection.s + 0.20)
        ))
