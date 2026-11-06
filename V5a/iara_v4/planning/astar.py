"""Deterministic 8-connected A* baseline on an inflated occupancy grid."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.geometry import Obstacle, line_of_sight, point_clearance
from ..core.path import Path
from ..core.types import Vec2
from .result import PlanningResult


@dataclass(frozen=True, slots=True)
class AStarConfig:
    resolution: float = 0.04
    clearance: float = 0.09
    bounds_margin: float = 0.5
    diagonal: bool = True
    simplify: bool = True
    max_expansions: int = 200_000

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.clearance < 0.0:
            raise ValueError("clearance must be non-negative")


class AStar:
    def __init__(self, config: AStarConfig | None = None) -> None:
        self.config = config or AStarConfig()

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

        if start.distance_to(goal) <= c.resolution:
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
        if not (min_x <= start.x <= max_x and min_y <= start.y <= max_y):
            return PlanningResult(
                path=None, success=False, reason="start outside bounds",
                planning_time=time.perf_counter() - t0,
            )
        if not (min_x <= goal.x <= max_x and min_y <= goal.y <= max_y):
            return PlanningResult(
                path=None, success=False, reason="goal outside bounds",
                planning_time=time.perf_counter() - t0,
            )

        nx = int(math.ceil((max_x - min_x) / c.resolution)) + 1
        ny = int(math.ceil((max_y - min_y) / c.resolution)) + 1
        if nx <= 1 or ny <= 1:
            return PlanningResult(
                path=None, success=False, reason="degenerate bounds",
                planning_time=time.perf_counter() - t0,
            )

        free = np.zeros((nx, ny), dtype=bool)
        for ix in range(nx):
            x = min_x + ix * c.resolution
            for iy in range(ny):
                y = min_y + iy * c.resolution
                free[ix, iy] = point_clearance(Vec2(x, y), obstacles) >= c.clearance

        def to_cell(p: Vec2) -> tuple[int, int]:
            return (
                min(nx - 1, max(0, int(round((p.x - min_x) / c.resolution)))),
                min(ny - 1, max(0, int(round((p.y - min_y) / c.resolution)))),
            )

        def cell_center(cell: tuple[int, int]) -> Vec2:
            return Vec2(min_x + cell[0] * c.resolution, min_y + cell[1] * c.resolution)

        def nearest_free(cell: tuple[int, int], max_ring: int = 6) -> tuple[int, int] | None:
            if free[cell]:
                return cell
            for r in range(1, max_ring + 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue
                        ix, iy = cell[0] + dx, cell[1] + dy
                        if 0 <= ix < nx and 0 <= iy < ny and free[ix, iy]:
                            return (ix, iy)
            return None

        start_cell = nearest_free(to_cell(start))
        goal_cell = nearest_free(to_cell(goal))
        if start_cell is None or goal_cell is None:
            return PlanningResult(
                path=None, success=False, reason="no free start/goal cell",
                planning_time=time.perf_counter() - t0,
            )

        # The exact start/goal must be connectable to their representative cells.
        if not line_of_sight(start, cell_center(start_cell), obstacles, c.clearance):
            return PlanningResult(
                path=None, success=False, reason="start cannot reach grid",
                planning_time=time.perf_counter() - t0,
            )
        if not line_of_sight(goal, cell_center(goal_cell), obstacles, c.clearance):
            return PlanningResult(
                path=None, success=False, reason="goal cannot reach grid",
                planning_time=time.perf_counter() - t0,
            )

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if c.diagonal:
            moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        g = np.full((nx, ny), np.inf, dtype=float)
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        g[start_cell] = 0.0
        heap: list[tuple[float, float, tuple[int, int]]] = [
            (math.dist(start_cell, goal_cell), 0.0, start_cell)
        ]
        closed = np.zeros((nx, ny), dtype=bool)
        expanded = 0
        found = False

        while heap and expanded < c.max_expansions:
            _, g_cur, cell = heapq.heappop(heap)
            if closed[cell]:
                continue
            closed[cell] = True
            expanded += 1
            if cell == goal_cell:
                found = True
                break
            ix, iy = cell
            for dx, dy in moves:
                nx_i, ny_i = ix + dx, iy + dy
                if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                    continue
                if not free[nx_i, ny_i] or closed[nx_i, ny_i]:
                    continue
                if dx != 0 and dy != 0:
                    # Do not cut through a blocked cardinal neighbour.
                    if not free[ix + dx, iy] or not free[ix, iy + dy]:
                        continue
                    step_cost = c.resolution * math.sqrt(2.0)
                else:
                    step_cost = c.resolution
                tentative = g_cur + step_cost
                if tentative + 1e-12 < g[nx_i, ny_i]:
                    g[nx_i, ny_i] = tentative
                    parent[(nx_i, ny_i)] = cell
                    h = math.hypot(nx_i - goal_cell[0], ny_i - goal_cell[1]) * c.resolution
                    heapq.heappush(heap, (tentative + h, tentative, (nx_i, ny_i)))

        if not found:
            return PlanningResult(
                path=None,
                success=False,
                cost=math.inf,
                iterations=expanded,
                planning_time=time.perf_counter() - t0,
                nodes=int(free.sum()),
                reason="A* exhausted the grid" if not heap else "A* expansion limit",
            )

        cells = [goal_cell]
        cur = goal_cell
        while cur != start_cell:
            cur = parent[cur]
            cells.append(cur)
        cells.reverse()
        pts = [start] + [cell_center(cell) for cell in cells] + [goal]
        # Remove duplicates introduced by exact start/goal alignment.
        clean: list[Vec2] = []
        for p in pts:
            if not clean or p.distance_to(clean[-1]) > 1e-9:
                clean.append(p)

        if c.simplify and len(clean) > 2:
            simplified: list[Vec2] = [clean[0]]
            i = 0
            while i < len(clean) - 1:
                chosen = i + 1
                for j in range(len(clean) - 1, i, -1):
                    if line_of_sight(clean[i], clean[j], obstacles, c.clearance):
                        chosen = j
                        break
                simplified.append(clean[chosen])
                i = chosen
            clean = simplified

        path = Path(clean, smoothed=False, name="astar")
        return PlanningResult(
            path=path,
            success=True,
            cost=path.length,
            iterations=expanded,
            planning_time=time.perf_counter() - t0,
            nodes=int(free.sum()),
            reason="goal reached",
            metadata={"grid": f"{nx}x{ny}", "resolution": c.resolution},
        )
