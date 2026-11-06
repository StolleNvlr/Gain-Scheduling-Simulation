"""Uniform-grid spatial index used by RRT*.

A grid avoids rebuilding a KD-tree at every iteration (a known inefficiency of
the V2.0 implementation) while remaining exact: the nearest-neighbour query
expands rings until the current best distance is provably closer than any point
in the next ring.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from ..core.types import Vec2


class SpatialIndex:
    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._points: list[Vec2] = []
        self._min_cell: tuple[int, int] | None = None
        self._max_cell: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self._points)

    @property
    def points(self) -> list[Vec2]:
        return self._points

    def _cell(self, p: Vec2) -> tuple[int, int]:
        return (math.floor(p.x / self.cell_size), math.floor(p.y / self.cell_size))

    def add(self, index: int, point: Vec2) -> None:
        if index != len(self._points):
            raise ValueError("indices must be added in order")
        cell = self._cell(point)
        self._cells[cell].append(index)
        self._points.append(point)
        if self._min_cell is None:
            self._min_cell = self._max_cell = cell
        else:
            self._min_cell = (
                min(self._min_cell[0], cell[0]),
                min(self._min_cell[1], cell[1]),
            )
            self._max_cell = (
                max(self._max_cell[0], cell[0]),
                max(self._max_cell[1], cell[1]),
            )

    def nearest(self, query: Vec2) -> tuple[int, float] | None:
        if not self._points:
            return None
        qcell = self._cell(query)
        best_idx: int | None = None
        best_d = math.inf
        # A point in a cell at Chebyshev ring r+1 is at least r*cell_size away
        # from any point in the query cell.  This gives a conservative stop.
        max_ring = 1
        if self._min_cell is not None and self._max_cell is not None:
            max_ring = max(
                abs(qcell[0] - self._min_cell[0]),
                abs(qcell[0] - self._max_cell[0]),
                abs(qcell[1] - self._min_cell[1]),
                abs(qcell[1] - self._max_cell[1]),
            ) + 2
        for r in range(0, max_ring + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    for idx in self._cells.get((qcell[0] + dx, qcell[1] + dy), ()):
                        d = query.distance_to(self._points[idx])
                        if d < best_d:
                            best_d = d
                            best_idx = idx
            if best_idx is not None and best_d <= r * self.cell_size + 1e-12:
                break
        if best_idx is None:  # pragma: no cover - only if the index is empty
            return None
        return best_idx, best_d

    def within_radius(self, query: Vec2, radius: float) -> list[tuple[int, float]]:
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        qcell = self._cell(query)
        span = int(math.ceil(radius / self.cell_size)) + 1
        found: list[tuple[int, float]] = []
        for cx in range(qcell[0] - span, qcell[0] + span + 1):
            for cy in range(qcell[1] - span, qcell[1] + span + 1):
                for idx in self._cells.get((cx, cy), ()):
                    d = query.distance_to(self._points[idx])
                    if d <= radius + 1e-12:
                        found.append((idx, d))
        return found

    def extend(self, points: Iterable[Vec2]) -> None:
        for p in points:
            self.add(len(self._points), p)
