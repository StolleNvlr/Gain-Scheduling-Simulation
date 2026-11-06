"""Shared controller interface and guidance record.

Every V4 controller implements the same three operations required by the
specification:

    reset(measurement, reference, seed)
    update(context) -> ControlCommand
    telemetry() -> dict

The guidance record is computed by a single, controller-independent layer so
that Pure Pursuit, PD, gain scheduling and MRAC all see the same look-ahead
point, speed profile and path projection.  This is the fairness condition for
the comparative experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.path import Path, PathProjection
from ..core.types import ControlCommand, Measurement, Vec2


@dataclass(frozen=True, slots=True)
class Guidance:
    speed: float
    heading_ref: float
    target: Vec2
    lookahead: float
    path_heading: float
    cross_track: float
    speed_factor: float = 1.0
    blocked: bool = False


@dataclass(frozen=True, slots=True)
class ControllerContext:
    measurement: Measurement
    path: Path
    projection: PathProjection
    guidance: Guidance
    dt: float


class Controller(Protocol):
    name: str

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None: ...

    def update(self, context: ControllerContext) -> ControlCommand: ...

    def telemetry(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class TelemetryBuffer:
    """Small, structured record written by every controller update."""

    last: dict[str, Any] = field(default_factory=dict)

    def set(self, **values: Any) -> None:
        self.last = dict(values)

    def get(self) -> dict[str, Any]:
        return dict(self.last)
