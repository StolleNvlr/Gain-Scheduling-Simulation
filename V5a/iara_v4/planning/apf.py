"""Artificial Potential Field baseline.

This is intentionally a *baseline*: the classical attractive/repulsive APF can
trap the vessel in local minima.  The experiment reports those failures rather
than hiding them with hand-tuned escape rules.

Reference: Khatib, O. (1986). Real-time obstacle avoidance for manipulators and
mobile robots. *International Journal of Robotics Research*, 5(1), 90-98.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.geometry import (
    CircleObstacle,
    Obstacle,
    RectangleObstacle,
    line_of_sight,
    point_clearance,
)
from ..core.path import Path
from ..core.types import Vec2
from .result import PlanningResult

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class APFConfig:
    step_size: float = 0.06
    max_iterations: int = 3000
    goal_tolerance: float = 0.08
    clearance: float = 0.09
    k_att: float = 1.0
    k_rep: float = 0.20
    influence_distance: float = 0.80
    bounds_margin: float = 0.5

    def __post_init__(self) -> None:
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.clearance < 0.0:
            raise ValueError("clearance must be non-negative")
        if self.influence_distance <= 0.0:
            raise ValueError("influence_distance must be positive")


def _nearest_point(ob: Obstacle, p: Vec2) -> Vec2:
    if isinstance(ob, RectangleObstacle):
        return Vec2(
            min(max(p.x, ob.min_x), ob.max_x),
            min(max(p.y, ob.min_y), ob.max_y),
        )
    if isinstance(ob, CircleObstacle):
        d = p - ob.center
        n = d.norm()
        if n < _EPS:
            return ob.center + Vec2(ob.radius, 0.0)
        return ob.center + d * (ob.radius / n)
    # Generic fallback: the closest point is not available; use the centre.
    return Vec2(ob.cx, ob.cy)  # type: ignore[attr-defined]


class ArtificialPotentialField:
    def __init__(self, config: APFConfig | None = None) -> None:
        self.config = config or APFConfig()

    def _force(self, p: Vec2, goal: Vec2, obstacles: Sequence[Obstacle]) -> Vec2:
        c = self.config
        attractive = (goal - p) * c.k_att
        repulsive = Vec2(0.0, 0.0)
        for ob in obstacles:
            nearest = _nearest_point(ob, p)
            delta = p - nearest
            d = delta.norm()
            if d < _EPS:
                # Inside an obstacle: push away from its centre deterministically.
                centre = Vec2(getattr(ob, "cx", 0.0), getattr(ob, "cy", 0.0))
                delta = p - centre
                d = max(delta.norm(), _EPS)
            if d < c.influence_distance:
                magnitude = c.k_rep * (1.0 / d - 1.0 / c.influence_distance) / (d * d)
                repulsive = repulsive + delta.normalized() * magnitude
        return attractive + repulsive

    def _bounds(
        self, start: Vec2, goal: Vec2, obstacles: Sequence[Obstacle]
    ) -> tuple[float, float, float, float]:
        xs = [start.x, goal.x]
        ys = [start.y, goal.y]
        for ob in obstacles:
            min_x, min_y, max_x, max_y = ob.bounding_box()
            xs.extend((min_x, max_x))
            ys.extend((min_y, max_y))
        m = self.config.bounds_margin
        return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

    def plan(
        self,
        start: Vec2,
        goal: Vec2,
        obstacles: Sequence[Obstacle],
        bounds: tuple[float, float, float, float] | None = None,
        rng: np.random.Generator | None = None,  # interface compatibility
    ) -> PlanningResult:
        del rng
        c = self.config
        t0 = time.perf_counter()
        obstacles = list(obstacles)
        if start.distance_to(goal) <= c.goal_tolerance:
            return PlanningResult(
                path=None, success=True, cost=0.0, reason="start within goal tolerance"
            )
        if point_clearance(start, obstacles) < c.clearance:
            return PlanningResult(
                path=None, success=False, reason="start is in collision",
                planning_time=time.perf_counter() - t0,
            )
        if point_clearance(goal, obstacles) < c.clearance:
            return PlanningResult(
                path=None, success=False, reason="goal is in collision",
                planning_time=time.perf_counter() - t0,
            )
        if bounds is None:
            bounds = self._bounds(start, goal, obstacles)
        min_x, min_y, max_x, max_y = bounds

        points = [start]
        p = start
        reason = "iteration limit"
        success = False
        for it in range(1, c.max_iterations + 1):
            if p.distance_to(goal) <= c.goal_tolerance:
                success = True
                reason = "goal reached"
                break
            force = self._force(p, goal, obstacles)
            if force.norm() < 1e-9:
                reason = "local minimum (zero force)"
                break
            direction = force.normalized()
            moved = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = p + direction * (c.step_size * scale)
                if not (min_x <= candidate.x <= max_x and min_y <= candidate.y <= max_y):
                    continue
                if line_of_sight(p, candidate, obstacles, c.clearance):
                    p = candidate
                    points.append(p)
                    moved = True
                    break
            if not moved:
                reason = "blocked by obstacle or bounds"
                break
        else:
            it = c.max_iterations

        if success:
            if points[-1].distance_to(goal) > 1e-9:
                points.append(goal)
            path = Path(points, smoothed=False, name="apf")
            return PlanningResult(
                path=path,
                success=True,
                cost=path.length,
                iterations=it,
                planning_time=time.perf_counter() - t0,
                nodes=len(points),
                reason=reason,
            )
        return PlanningResult(
            path=None,
            success=False,
            cost=math.inf,
            iterations=it,
            planning_time=time.perf_counter() - t0,
            nodes=len(points),
            reason=reason,
        )
