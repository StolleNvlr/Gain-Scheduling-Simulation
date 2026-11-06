"""Common result type returned by every global planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.path import Path


@dataclass(slots=True)
class PlanningResult:
    path: Path | None
    success: bool
    cost: float = 0.0
    iterations: int = 0
    planning_time: float = 0.0
    nodes: int = 0
    reason: str = ""
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)
