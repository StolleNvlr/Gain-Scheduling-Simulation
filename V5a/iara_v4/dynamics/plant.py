"""Dynamic 3-DOF plant and the V3-compatible kinematic baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.geometry import (
    Obstacle,
    footprint_collision,
    oriented_rectangle_clearance,
)
from ..core.kinematics import DifferentialDrive, DriveLimits
from ..core.types import ControlCommand, Pose2D, Twist
from .actuator import ActuatorConfig, ActuatorModel
from .disturbance import DisturbanceConfig, DisturbanceModel
from .model import VehicleParameters
from .state import PlantState, PlantStep


class PlantNumericalError(RuntimeError):
    """Raised when a plant state becomes non-finite; never silently zeroed."""


@dataclass(frozen=True, slots=True)
class DynamicPlantConfig:
    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    dt: float = 0.02
    substeps: int = 4
    boat_length: float = 0.191
    boat_width: float = 0.132
    boat_height: float = 0.078
    safety_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")
        if self.boat_length <= 0.0 or self.boat_width <= 0.0:
            raise ValueError("boat dimensions must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")

    @property
    def half_diagonal(self) -> float:
        return 0.5 * math.hypot(self.boat_length, self.boat_width)

    @property
    def planning_radius(self) -> float:
        return self.half_diagonal + self.safety_margin


class DynamicUSVPlant:
    """Minimal 3-DOF manoeuvring plant with a first-order actuator layer."""

    def __init__(
        self,
        config: DynamicPlantConfig | None = None,
        initial_pose: Pose2D | None = None,
        *,
        seed: int = 0,
        scenario: str = "",
    ) -> None:
        self.config = config or DynamicPlantConfig()
        self.vehicle = self.config.vehicle
        self.actuator = ActuatorModel(self.config.actuator, self.vehicle)
        self.disturbance = DisturbanceModel(
            self.config.disturbance, seed=seed, scenario=scenario
        )
        pose = initial_pose or Pose2D(0.0, 0.0, 0.0)
        self._state = PlantState(0.0, pose, 0.0, 0.0, 0.0)

    @property
    def state(self) -> PlantState:
        return self._state

    @property
    def pose(self) -> Pose2D:
        return self._state.pose

    @property
    def time(self) -> float:
        return self._state.time

    def reset(
        self,
        pose: Pose2D,
        *,
        u: float = 0.0,
        v: float = 0.0,
        r: float = 0.0,
    ) -> None:
        self.actuator.reset()
        self.disturbance.reset()
        self._state = PlantState(0.0, pose, float(u), float(v), float(r))

    def step(self, command: ControlCommand, dt: float | None = None) -> PlantStep:
        dt = self.config.dt if dt is None else float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.disturbance.update(dt)
        act = self.actuator.step(
            v_ref=command.v,
            omega_ref=command.omega,
            u=self._state.u,
            r=self._state.r,
            delta_cmd=command.delta_rudder,
            tau_u_cmd=command.tau_u,
            tau_r_cmd=command.tau_r,
            dt=dt,
        )
        tau_ext = self.disturbance.force(self._state.pose.theta)
        x0 = np.array(
            [
                self._state.pose.x,
                self._state.pose.y,
                self._state.pose.theta,
                self._state.u,
                self._state.v,
                self._state.r,
            ],
            dtype=float,
        )
        x1 = self._integrate_rk4(x0, dt, act.tau_u_applied, act.tau_r_applied, tau_ext)
        if not np.all(np.isfinite(x1)):
            raise PlantNumericalError(
                f"non-finite dynamic state at t={self._state.time + dt:.6f}: {x1!r}"
            )
        pose = Pose2D(float(x1[0]), float(x1[1]), float(x1[2]))
        self._state = PlantState(
            time=self._state.time + dt,
            pose=pose,
            u=float(x1[3]),
            v=float(x1[4]),
            r=float(x1[5]),
            tau_u=act.tau_u_applied,
            tau_r=act.tau_r_applied,
            delta=act.delta_applied,
            delta_cmd=act.delta_cmd,
            disturbance_u=float(tau_ext[0]),
            disturbance_v=float(tau_ext[1]),
            disturbance_r=float(tau_ext[2]),
            saturated=act.saturated,
            deadband_active=act.deadband_active,
        )
        return PlantStep(
            state=self._state,
            command=command,
            dt=dt,
            tau_u_cmd=act.tau_u_cmd,
            tau_r_cmd=act.tau_r_cmd,
            delta_cmd=act.delta_cmd,
            delta_applied=act.delta_applied,
            saturated=act.saturated,
            deadband_active=act.deadband_active,
        )

    def _integrate_rk4(
        self,
        x0: np.ndarray,
        dt: float,
        tau_u: float,
        tau_r: float,
        tau_ext: np.ndarray,
    ) -> np.ndarray:
        n = max(1, int(self.config.substeps))
        h = dt / n
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(n):
            k1 = self._rhs(x, tau_u, tau_r, tau_ext)
            k2 = self._rhs(x + 0.5 * h * k1, tau_u, tau_r, tau_ext)
            k3 = self._rhs(x + 0.5 * h * k2, tau_u, tau_r, tau_ext)
            k4 = self._rhs(x + h * k3, tau_u, tau_r, tau_ext)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def _rhs(
        self,
        x: np.ndarray,
        tau_u: float,
        tau_r: float,
        tau_ext: np.ndarray,
    ) -> np.ndarray:
        _, _, psi, u, v, r = (float(value) for value in x)
        nu = np.array([u, v, r], dtype=float)
        M = self.vehicle.mass_matrix
        C = self.vehicle.coriolis(nu)
        D = self.vehicle.damping(nu)
        generalized = np.array(
            [tau_u + tau_ext[0], tau_ext[1], tau_r + tau_ext[2]], dtype=float
        )
        nu_dot = np.linalg.solve(M, generalized - C @ nu - D @ nu)
        cpsi, spsi = math.cos(psi), math.sin(psi)
        return np.array(
            [
                cpsi * u - spsi * v,
                spsi * u + cpsi * v,
                r,
                nu_dot[0],
                nu_dot[1],
                nu_dot[2],
            ],
            dtype=float,
        )

    def collision(self, obstacles: Sequence[Obstacle]) -> bool:
        return footprint_collision(
            self._state.pose,
            self.config.boat_length,
            self.config.boat_width,
            obstacles,
        )

    def clearance(self, obstacles: Sequence[Obstacle]) -> float:
        return oriented_rectangle_clearance(
            self._state.pose,
            self.config.boat_length,
            self.config.boat_width,
            obstacles,
        )


# ---------------------------------------------------------------------------
# Kinematic baseline (V3 behaviour, preserved for plant-isolation experiments)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KinematicPlantConfig:
    limits: DriveLimits = field(
        default_factory=lambda: DriveLimits(
            track_width=0.132,
            max_wheel_speed=0.50,
            max_linear_speed=0.50,
            max_angular_speed=2.00,
        )
    )
    dt: float = 0.02
    boat_length: float = 0.191
    boat_width: float = 0.132
    boat_height: float = 0.078
    safety_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.boat_length <= 0.0 or self.boat_width <= 0.0:
            raise ValueError("boat dimensions must be positive")

    @property
    def half_diagonal(self) -> float:
        return 0.5 * math.hypot(self.boat_length, self.boat_width)

    @property
    def planning_radius(self) -> float:
        return self.half_diagonal + self.safety_margin


class KinematicUSVPlant:
    """Exact ideal differential-drive plant, unchanged from V3."""

    def __init__(
        self,
        config: KinematicPlantConfig | None = None,
        initial_pose: Pose2D | None = None,
    ) -> None:
        self.config = config or KinematicPlantConfig()
        self.drive = DifferentialDrive(self.config.limits)
        pose = initial_pose or Pose2D(0.0, 0.0, 0.0)
        self._state = PlantState(0.0, pose, 0.0, 0.0, 0.0)

    @property
    def state(self) -> PlantState:
        return self._state

    @property
    def pose(self) -> Pose2D:
        return self._state.pose

    @property
    def time(self) -> float:
        return self._state.time

    def reset(self, pose: Pose2D) -> None:
        self._state = PlantState(0.0, pose, 0.0, 0.0, 0.0)

    def step(self, command: ControlCommand, dt: float | None = None) -> PlantStep:
        dt = self.config.dt if dt is None else float(dt)
        pose = self.drive.integrate(
            self._state.pose, Twist(command.v, command.omega), dt
        )
        if not (
            math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.theta)
        ):
            raise PlantNumericalError(f"non-finite kinematic state at t={self._state.time + dt:.6f}")
        self._state = PlantState(
            time=self._state.time + dt,
            pose=pose,
            u=command.v,
            v=0.0,
            r=command.omega,
            saturated=command.saturated,
        )
        return PlantStep(
            state=self._state,
            command=command,
            dt=dt,
            saturated=command.saturated,
        )

    def collision(self, obstacles: Sequence[Obstacle]) -> bool:
        return footprint_collision(
            self._state.pose,
            self.config.boat_length,
            self.config.boat_width,
            obstacles,
        )

    def clearance(self, obstacles: Sequence[Obstacle]) -> float:
        return oriented_rectangle_clearance(
            self._state.pose,
            self.config.boat_length,
            self.config.boat_width,
            obstacles,
        )
