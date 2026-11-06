"""Common state records for kinematic and dynamic plants."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import ControlCommand, Pose2D


@dataclass(frozen=True, slots=True)
class PlantState:
    time: float
    pose: Pose2D
    u: float
    v: float
    r: float
    tau_u: float = 0.0
    tau_r: float = 0.0
    delta: float = 0.0
    delta_cmd: float = 0.0
    disturbance_u: float = 0.0
    disturbance_v: float = 0.0
    disturbance_r: float = 0.0
    saturated: bool = False
    deadband_active: bool = False

    def as_nu(self) -> tuple[float, float, float]:
        return (self.u, self.v, self.r)


@dataclass(frozen=True, slots=True)
class PlantStep:
    state: PlantState
    command: ControlCommand
    dt: float
    tau_u_cmd: float = 0.0
    tau_r_cmd: float = 0.0
    delta_cmd: float = 0.0
    delta_applied: float = 0.0
    saturated: bool = False
    deadband_active: bool = False
