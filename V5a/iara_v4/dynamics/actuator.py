"""Actuator models with first-order lag, deadband, rate limit and saturation.

Order of operations (declared explicitly because the two possible orders are
not equivalent):

    command -> first-order lag -> deadband -> saturation -> applied

The solved actuator states, not the commanded values, enter the 3-DOF plant.
Every intermediate signal is returned so the logger can distinguish a
controller-induced command from an actuator-induced limitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import VehicleParameters


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    tau_u_max: float = 12.0
    tau_u_time_constant: float = 0.12
    tau_u_deadband: float = 0.02
    delta_max: float = 0.65
    delta_rate_max: float = 3.0
    delta_time_constant: float = 0.08
    delta_deadband: float = 0.005
    surge_kp: float = 18.0
    yaw_rate_kp: float = 1.2

    def __post_init__(self) -> None:
        if self.tau_u_max <= 0.0 or self.tau_u_time_constant <= 0.0:
            raise ValueError("surge actuator parameters must be positive")
        if self.delta_max <= 0.0 or self.delta_time_constant <= 0.0:
            raise ValueError("rudder actuator parameters must be positive")
        if self.delta_rate_max <= 0.0:
            raise ValueError("rudder rate limit must be positive")
        if min(self.tau_u_deadband, self.delta_deadband) < 0.0:
            raise ValueError("deadbands must be non-negative")
        if self.surge_kp <= 0.0 or self.yaw_rate_kp < 0.0:
            raise ValueError("servo gains are invalid")


@dataclass(frozen=True, slots=True)
class ActuatorOutput:
    tau_u_cmd: float
    tau_u_applied: float
    delta_cmd: float
    delta_applied: float
    tau_r_cmd: float
    tau_r_applied: float
    saturated: bool
    deadband_active: bool


@dataclass(slots=True)
class ActuatorState:
    tau_u_real: float = 0.0
    delta_real: float = 0.0


def _deadband(value: float, width: float) -> tuple[float, bool]:
    if width <= 0.0:
        return value, False
    if abs(value) <= width:
        return 0.0, abs(value) > 0.0
    return math.copysign(abs(value) - width, value), False


def _sat(value: float, limit: float) -> tuple[float, bool]:
    if abs(value) <= limit:
        return value, False
    return math.copysign(limit, value), True


class ActuatorModel:
    def __init__(
        self,
        config: ActuatorConfig | None = None,
        vehicle: VehicleParameters | None = None,
    ) -> None:
        self.config = config or ActuatorConfig()
        self.vehicle = vehicle or VehicleParameters()
        self.state = ActuatorState()

    def reset(self) -> None:
        self.state = ActuatorState()

    # -- surge --------------------------------------------------------------

    def surge_force(self, v_ref: float, u: float, dt: float) -> tuple[float, float, bool, bool]:
        """Track a surge-speed reference with feed-forward drag compensation."""

        c = self.config
        v_ref = float(v_ref)
        drag_ff = self.vehicle.d_u_linear * v_ref + self.vehicle.d_u_quadratic * v_ref * abs(v_ref)
        tau_cmd = drag_ff + c.surge_kp * (v_ref - u)
        tau_cmd, sat_cmd = _sat(tau_cmd, c.tau_u_max)
        alpha = 1.0 - math.exp(-max(dt, 0.0) / c.tau_u_time_constant)
        real = self.state.tau_u_real + alpha * (tau_cmd - self.state.tau_u_real)
        real, dead = _deadband(real, c.tau_u_deadband)
        applied, sat = _sat(real, c.tau_u_max)
        self.state.tau_u_real = applied
        return applied, tau_cmd, bool(sat or sat_cmd), dead

    # -- rudder -------------------------------------------------------------

    def rudder_angle(self, delta_cmd: float, dt: float) -> tuple[float, float, bool, bool]:
        c = self.config
        delta_cmd, sat_cmd = _sat(float(delta_cmd), c.delta_max)
        # First-order approach plus explicit rate limit.
        rate = (delta_cmd - self.state.delta_real) / c.delta_time_constant
        rate = max(-c.delta_rate_max, min(c.delta_rate_max, rate))
        real = self.state.delta_real + rate * max(dt, 0.0)
        real, dead = _deadband(real, c.delta_deadband)
        applied, sat = _sat(real, c.delta_max)
        self.state.delta_real = applied
        return applied, delta_cmd, bool(sat or sat_cmd), dead

    def yaw_rate_to_delta(
        self, r_ref: float, u: float, r: float, dt: float
    ) -> tuple[float, float, bool, bool]:
        """Nominal inverse map plus proportional yaw-rate feedback.

        The feed-forward term is the steady-state inverse of the yaw model,
        ``delta_ff = a(q)/b(q) r_ref``.  The feedback term rejects the
        unmodelled sway coupling, actuator lag and disturbances.  It is part of
        the shared low-level layer used by the non-adaptive baselines.
        """

        a = self.vehicle.yaw_a(u)
        b = max(self.vehicle.yaw_b(u), 1e-12)
        delta_ff = (a / b) * float(r_ref)
        delta_cmd = delta_ff + self.config.yaw_rate_kp * (float(r_ref) - float(r))
        return self.rudder_angle(delta_cmd, dt)

    def step(
        self,
        *,
        v_ref: float,
        omega_ref: float,
        u: float,
        r: float,
        delta_cmd: float | None,
        tau_u_cmd: float | None,
        tau_r_cmd: float | None,
        dt: float,
    ) -> ActuatorOutput:
        """Advance the actuator layer for one control interval."""

        if tau_u_cmd is None:
            tau_u_applied, tau_u_cmd_used, sat_u, dead_u = self.surge_force(v_ref, u, dt)
        else:
            tau_u_cmd_used, sat_u = _sat(float(tau_u_cmd), self.config.tau_u_max)
            alpha = 1.0 - math.exp(-max(dt, 0.0) / self.config.tau_u_time_constant)
            real = self.state.tau_u_real + alpha * (tau_u_cmd_used - self.state.tau_u_real)
            real, dead_u = _deadband(real, self.config.tau_u_deadband)
            tau_u_applied, sat_u2 = _sat(real, self.config.tau_u_max)
            sat_u = bool(sat_u or sat_u2)
            self.state.tau_u_real = tau_u_applied

        if tau_r_cmd is not None:
            gain = max(self.vehicle.rudder_gain(u), 1e-12)
            delta_cmd = float(tau_r_cmd) / gain

        if delta_cmd is None:
            delta_command = self.yaw_rate_to_delta_command(omega_ref, u, r)
        else:
            # Direct-rudder controllers use the same yaw-rate rejection
            # layer as twist controllers so benchmark conditions are fair.
            delta_command = float(delta_cmd) + self.config.yaw_rate_kp * (
                float(omega_ref) - float(r)
            )
        delta_applied, delta_cmd_used, sat_r, dead_r = self.rudder_angle(
            delta_command, dt
        )

        gain = max(self.vehicle.rudder_gain(u), 1e-12)
        tau_r_cmd_out = gain * delta_command
        tau_r_applied = self.vehicle.rudder_gain(u) * delta_applied
        return ActuatorOutput(
            tau_u_cmd=float(tau_u_cmd_used),
            tau_u_applied=float(tau_u_applied),
            delta_cmd=float(delta_cmd_used),
            delta_applied=float(delta_applied),
            tau_r_cmd=float(tau_r_cmd_out),
            tau_r_applied=float(tau_r_applied),
            saturated=bool(sat_u or sat_r),
            deadband_active=bool(dead_u or dead_r),
        )

    def yaw_rate_to_delta_command(self, r_ref: float, u: float, r: float) -> float:
        """Return the common yaw-rate feed-forward and rejection command."""

        a = self.vehicle.yaw_a(u)
        b = max(self.vehicle.yaw_b(u), 1e-12)
        delta_ff = (a / b) * float(r_ref)
        return delta_ff + self.config.yaw_rate_kp * (float(r_ref) - float(r))
