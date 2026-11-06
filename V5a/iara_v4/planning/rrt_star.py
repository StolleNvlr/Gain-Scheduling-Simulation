"""Asymptotically optimal RRT* with exact segment clearance checks.

References
----------
Karaman, S. & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion
planning. *International Journal of Robotics Research*, 30(7), 846-894.

LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.

Implementation notes
--------------------
* The random generator is injected; no global state is used.
* Nearest-neighbour queries use an incremental uniform grid.
* Rewiring updates the affected subtree costs, keeping the cost-to-come
  consistent.  Ancestors of the new node are never rewired to it, which would
  create a cycle.
* Collision checks use the exact distance between a segment and each obstacle,
  not point sampling.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..core.geometry import (
    Obstacle,
    point_clearance,
    segment_clearance,
)
from ..core.path import Path
from ..core.types import Vec2
from .result import PlanningResult
from .spatial import SpatialIndex

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RRTStarConfig:
    step_size: float = 0.15
    goal_bias: float = 0.10
    max_iterations: int = 3000
    post_goal_iterations: int = 300
    goal_tolerance: float = 0.08
    goal_connect_radius: float = 0.25
    clearance: float = 0.09
    rewire_gamma: float = 2.5
    rewire_radius_min: float = 0.08
    rewire_radius_max: float = 0.45
    bounds_margin: float = 0.5

    def __post_init__(self) -> None:
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not 0.0 <= self.goal_bias <= 1.0:
            raise ValueError("goal_bias must be in [0, 1]")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.clearance < 0.0:
            raise ValueError("clearance must be non-negative")


class RRTStar:
    def __init__(self, config: RRTStarConfig | None = None) -> None:
        self.config = config or RRTStarConfig()

    def _bounds(
        self,
        start: Vec2,
        goal: Vec2,
        obstacles: Sequence[Obstacle],
    ) -> tuple[float, float, float, float]:
        xs = [start.x, goal.x]
        ys = [start.y, goal.y]
        for ob in obstacles:
            min_x, min_y, max_x, max_y = ob.bounding_box()
            xs.extend((min_x, max_x))
            ys.extend((min_y, max_y))
        m = self.config.bounds_margin
        return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

    def _free_point(self, p: Vec2, obstacles: Sequence[Obstacle]) -> bool:
        return point_clearance(p, obstacles) >= self.config.clearance

    def _free_segment(
        self, a: Vec2, b: Vec2, obstacles: Sequence[Obstacle]
    ) -> bool:
        return segment_clearance(a, b, obstacles) >= self.config.clearance

    def _rewire_radius(self, n: int) -> float:
        c = self.config
        r = c.rewire_gamma * math.sqrt(math.log(n + 1.0) / (n + 1.0))
        return min(c.rewire_radius_max, max(c.rewire_radius_min, r))

    @staticmethod
    def _reparent(
        index: int,
        new_parent: int,
        new_cost: float,
        parents: list[int],
        costs: list[float],
        children: list[list[int]],
    ) -> None:
        """Change a node's parent and propagate the cost delta to its subtree."""

        old_parent = parents[index]
        if old_parent >= 0:
            children[old_parent].remove(index)
        parents[index] = new_parent
        children[new_parent].append(index)
        delta = new_cost - costs[index]
        costs[index] = new_cost
        if abs(delta) < _EPS:
            return
        stack = [index]
        while stack:
            node = stack.pop()
            for child in children[node]:
                costs[child] += delta
                stack.append(child)

    def plan(
        self,
        start: Vec2,
        goal: Vec2,
        obstacles: Sequence[Obstacle],
        bounds: tuple[float, float, float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> PlanningResult:
        c = self.config
        t0 = time.perf_counter()
        rng = rng or np.random.default_rng(0)
        obstacles = list(obstacles)

        if start.distance_to(goal) <= c.goal_tolerance:
            return PlanningResult(
                path=None,
                success=True,
                cost=0.0,
                reason="start within goal tolerance",
            )
        if not self._free_point(start, obstacles):
            return PlanningResult(
                path=None,
                success=False,
                reason="start is in collision",
                planning_time=time.perf_counter() - t0,
            )
        if not self._free_point(goal, obstacles):
            return PlanningResult(
                path=None,
                success=False,
                reason="goal is in collision",
                planning_time=time.perf_counter() - t0,
            )

        if bounds is None:
            bounds = self._bounds(start, goal, obstacles)
        min_x, min_y, max_x, max_y = bounds
        if not (min_x <= start.x <= max_x and min_y <= start.y <= max_y):
            return PlanningResult(
                path=None,
                success=False,
                reason="start outside bounds",
                planning_time=time.perf_counter() - t0,
            )
        if not (min_x <= goal.x <= max_x and min_y <= goal.y <= max_y):
            return PlanningResult(
                path=None,
                success=False,
                reason="goal outside bounds",
                planning_time=time.perf_counter() - t0,
            )

        xs = [start.x]
        ys = [start.y]
        parents = [-1]
        costs = [0.0]
        children: list[list[int]] = [[]]
        index = SpatialIndex(max(c.step_size, c.rewire_radius_min))
        index.add(0, start)

        best_goal_parent = -1
        best_goal_cost = math.inf
        first_goal_iteration = -1
        iterations = 0

        for it in range(c.max_iterations):
            iterations = it + 1
            if rng.random() < c.goal_bias:
                sample = goal
            else:
                sample = Vec2(
                    float(rng.uniform(min_x, max_x)),
                    float(rng.uniform(min_y, max_y)),
                )

            nearest = index.nearest(sample)
            if nearest is None:  # pragma: no cover - start always exists
                continue
            nearest_idx, nearest_dist = nearest
            if nearest_dist < _EPS:
                continue
            nearest_point = Vec2(xs[nearest_idx], ys[nearest_idx])
            direction = (sample - nearest_point).normalized()
            step = min(c.step_size, nearest_dist)
            new_point = nearest_point + direction * step
            if not self._free_point(new_point, obstacles):
                continue
            if not self._free_segment(nearest_point, new_point, obstacles):
                continue

            n = len(xs)
            radius = self._rewire_radius(n)
            near = index.within_radius(new_point, radius)

            # Choose the cheapest collision-free parent in the neighbourhood.
            parent_idx = nearest_idx
            parent_cost = costs[nearest_idx] + nearest_point.distance_to(new_point)
            for idx, dist in near:
                candidate = costs[idx] + dist
                if candidate + _EPS < parent_cost:
                    p = Vec2(xs[idx], ys[idx])
                    if self._free_segment(p, new_point, obstacles):
                        parent_idx = idx
                        parent_cost = candidate

            new_idx = len(xs)
            xs.append(new_point.x)
            ys.append(new_point.y)
            parents.append(parent_idx)
            costs.append(parent_cost)
            children.append([])
            children[parent_idx].append(new_idx)
            index.add(new_idx, new_point)

            # Ancestors of the new node must not be rewired to it (cycle guard).
            ancestors: set[int] = set()
            cur = parent_idx
            while cur >= 0:
                ancestors.add(cur)
                cur = parents[cur]

            for idx, dist in near:
                if idx == parent_idx or idx in ancestors:
                    continue
                candidate = parent_cost + dist
                if candidate + _EPS < costs[idx]:
                    p = Vec2(xs[idx], ys[idx])
                    if self._free_segment(new_point, p, obstacles):
                        self._reparent(idx, new_idx, candidate, parents, costs, children)

            # Try to connect the new node to the goal.
            d_goal = new_point.distance_to(goal)
            if d_goal <= c.goal_connect_radius:
                if self._free_segment(new_point, goal, obstacles):
                    candidate = parent_cost + d_goal
                    if candidate + _EPS < best_goal_cost:
                        best_goal_cost = candidate
                        best_goal_parent = new_idx
                        if first_goal_iteration < 0:
                            first_goal_iteration = it

            if (
                best_goal_parent >= 0
                and first_goal_iteration >= 0
                and it - first_goal_iteration >= c.post_goal_iterations
            ):
                break

        elapsed = time.perf_counter() - t0
        if best_goal_parent < 0:
            return PlanningResult(
                path=None,
                success=False,
                cost=math.inf,
                iterations=iterations,
                planning_time=elapsed,
                nodes=len(xs),
                reason="no collision-free path found",
            )

        # Rewiring may have lowered the cost of nodes that were not the best
        # goal connection when they were created.  Re-evaluate every node
        # within the connection radius against the final tree costs.
        best_goal_parent = -1
        best_goal_cost = math.inf
        for i in range(len(xs)):
            p = Vec2(xs[i], ys[i])
            d_goal = p.distance_to(goal)
            if d_goal > c.goal_connect_radius:
                continue
            candidate = costs[i] + d_goal
            if candidate + _EPS < best_goal_cost and self._free_segment(
                p, goal, obstacles
            ):
                best_goal_cost = candidate
                best_goal_parent = i
        if best_goal_parent < 0:  # pragma: no cover - original connection exists
            return PlanningResult(
                path=None,
                success=False,
                cost=math.inf,
                iterations=iterations,
                planning_time=elapsed,
                nodes=len(xs),
                reason="goal connection lost during rewiring",
            )

        chain: list[Vec2] = []
        cur = best_goal_parent
        while cur >= 0:
            chain.append(Vec2(xs[cur], ys[cur]))
            cur = parents[cur]
        chain.reverse()
        chain.append(goal)
        path = Path(chain, smoothed=False, name="rrt-star")
        # Rewiring can change the cost of the node used for the goal connection
        # after it was recorded.  The parent chain is authoritative, so report
        # its exact length instead of the stale search-time estimate.
        final_cost = path.length
        return PlanningResult(
            path=path,
            success=True,
            cost=final_cost,
            iterations=iterations,
            planning_time=elapsed,
            nodes=len(xs),
            reason="goal reached",
            metadata={
                "goal_parent": best_goal_parent,
                "first_goal_iteration": first_goal_iteration,
            },
        )
