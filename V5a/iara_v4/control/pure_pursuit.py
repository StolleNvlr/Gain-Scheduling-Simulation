"""Geometric pure pursuit with curvature feed-forward and a speed profile.

The steering law is the classical pure-pursuit curvature

    kappa = 2 sin(alpha) / L_d,        omega = V kappa

where ``alpha`` is the angle between the vessel heading and the look-ahead
point (Coulter, 1992; Snider, 2009).  The speed profile limits lateral
acceleration on curves and deceleration near the goal.  The inverse
differential-drive map and the actuator saturation are applied by
``DifferentialDrive``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.kinematics import DifferentialDrive
from ..core.path import Path, PathProjection
from ..core.types import ControlCommand, Pose2D, Vec2, wrap_angle


@dataclass(frozen=True, slots=True)
class PurePursuitConfig:
    lookahead_min: float = 0.15
    lookahead_max: float = 0.60
    lookahead_gain: float = 1.00
    v_max: float = 0.45
    v_min: float = 0.05
    a_lat_max: float = 0.80
    a_dec_max: float = 0.50
    goal_tolerance: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 < self.lookahead_min <= self.lookahead_max:
            raise ValueError("lookahead bounds are invalid")
        if self.lookahead_gain < 0.0:
            raise ValueError("lookahead_gain must be non-negative")
        if self.v_max <= 0.0 or self.v_min < 0.0:
            raise ValueError("speed bounds are invalid")
        if self.a_lat_max <= 0.0 or self.a_dec_max <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be non-negative")


class PurePursuit:
    def __init__(
        self,
        config: PurePursuitConfig | None = None,
        drive: DifferentialDrive | None = None,
    ) -> None:
        self.config = config or PurePursuitConfig()
        self.drive = drive or DifferentialDrive()
        self.reset()

    def reset(self) -> None:
        self._last_v = 0.0

    def lookahead_distance(self, remaining: float) -> float:
        c = self.config
        # A conservative speed estimate for look-ahead scheduling.  Using the
        # remaining distance avoids a hidden dependence on the previous command.
        v_ref = min(c.v_max, math.sqrt(2.0 * c.a_dec_max * max(0.0, remaining)))
        return min(c.lookahead_max, max(c.lookahead_min, c.lookahead_min + c.lookahead_gain * v_ref))

    def speed_profile(
        self, pose: Pose2D, path: Path, projection: PathProjection
    ) -> tuple[float, float]:
        """Return ``(v, kappa_used)`` for the current state."""

        c = self.config
        remaining = max(0.0, path.length - projection.s)
        distance_to_goal = pose.position.distance_to(path.end)
        if distance_to_goal <= c.goal_tolerance:
            return 0.0, 0.0

        lookahead = self.lookahead_distance(remaining)
        s0 = projection.s
        s1 = min(path.length, s0 + max(lookahead, 0.10))
        samples = np.linspace(s0, s1, 8)
        kappa = max(abs(path.curvature_at(float(s))) for s in samples)
        kappa = max(kappa, 1e-6)

        v_curve = math.sqrt(c.a_lat_max / kappa)
        # Decelerate on the true Euclidean distance to the goal: the success
        # criterion is a disc around the goal, not the remaining arc length.
        v_goal = math.sqrt(
            2.0 * c.a_dec_max * max(0.0, distance_to_goal - c.goal_tolerance)
        )
        v = min(c.v_max, v_curve, v_goal, self.drive.max_speed_for_curvature(kappa))
        if v < c.v_min and distance_to_goal > c.goal_tolerance:
            v = min(c.v_min, self.drive.max_speed_for_curvature(kappa))
        return max(0.0, v), kappa

    def lookahead_point(
        self, pose: Pose2D, path: Path, projection: PathProjection | None = None
    ) -> tuple[Vec2, PathProjection, float]:
        projection = projection or path.project(pose.position)
        remaining = max(0.0, path.length - projection.s)
        ld = self.lookahead_distance(remaining)
        s = min(path.length, projection.s + ld)
        return path.point_at(s), projection, ld

    def desired_heading(
        self, pose: Pose2D, path: Path, projection: PathProjection | None = None
    ) -> float:
        target, _, _ = self.lookahead_point(pose, path, projection)
        return math.atan2(target.y - pose.y, target.x - pose.x)

    def command_to_point(
        self,
        pose: Pose2D,
        target: Vec2,
        speed: float,
    ) -> ControlCommand:
        dx = target.x - pose.x
        dy = target.y - pose.y
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            return self.drive.command(0.0, 0.0, target=target)
        alpha = wrap_angle(math.atan2(dy, dx) - pose.theta)
        ld = max(distance, self.config.lookahead_min)
        kappa = 2.0 * math.sin(alpha) / ld
        omega = speed * kappa
        self._last_v = speed
        return self.drive.command(speed, omega, target=target)

    def command(
        self,
        pose: Pose2D,
        path: Path,
        projection: PathProjection | None = None,
    ) -> ControlCommand:
        projection = projection or path.project(pose.position)
        target, projection, _ = self.lookahead_point(pose, path, projection)
        v, _ = self.speed_profile(pose, path, projection)
        if v <= 0.0:
            return self.drive.command(0.0, 0.0, target=path.end)
        return self.command_to_point(pose, target, v)
