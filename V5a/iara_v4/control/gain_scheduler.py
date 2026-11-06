"""Corrected gain-scheduled heading controller.

The scheduling variable is the measured surge speed ``alpha = u`` in m/s.
Gains are interpolated with non-negative convex weights; outside the ordering
envelope the value is clamped (never extrapolated silently).  The controller
can operate in two transparent modes:

* ``corrected``: gains derived from the V4 nominal yaw model with a constant
  damping ratio.  The physical coefficients are interpolated and the gains
  are recomputed at runtime, preserving the designed continuous closed-loop
  yaw matrix between design points.
* ``matlab_absolute``: the absolute gains printed in ``projeto_barco.m``.  They
  are kept for the audit and for an explicit transfer-failure ablation; they
  are *not* the operational defaults because the MATLAB input is not the V4
  rudder angle.

The legacy 1 Hz ARX implementation is not placed here.  It is reconstructed in
``experiments/reference.py`` because it is a plant/closed-loop simulation, not
a controller in the V4 sense.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..core.kinematics import DifferentialDrive
from ..core.types import ControlCommand, Measurement, Vec2, wrap_angle
from ..dynamics.model import VehicleParameters
from .base import ControllerContext, Guidance, TelemetryBuffer


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    speed: float
    kp: float
    kd: float
    label: str = ""
    source: str = "corrected"

    def __post_init__(self) -> None:
        if self.speed < 0.0:
            raise ValueError("operating-point speed must be non-negative")
        if not (math.isfinite(self.kp) and math.isfinite(self.kd)):
            raise ValueError("gains must be finite")


@dataclass(frozen=True, slots=True)
class ScheduledGains:
    speed: float
    speed_used: float
    kp: float
    kd: float
    interpolation_weights: tuple[tuple[float, float], ...]
    regime: str
    clamped: bool
    applied_directly: bool = False


@dataclass(frozen=True, slots=True)
class GainSchedulerConfig:
    points: tuple[OperatingPoint, ...]
    extrapolation: str = "clamp"
    gamma_max: float = 2.0

    def __post_init__(self) -> None:
        if len(self.points) < 1:
            raise ValueError("at least one operating point is required")
        speeds = [p.speed for p in self.points]
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("operating points must be strictly increasing in speed")
        if self.extrapolation not in {"clamp", "linear"}:
            raise ValueError("extrapolation must be 'clamp' or 'linear'")
        if self.gamma_max <= 0.0:
            raise ValueError("gamma_max must be positive")

    @property
    def speed_min(self) -> float:
        return self.points[0].speed

    @property
    def speed_max(self) -> float:
        return self.points[-1].speed


class GainScheduler:
    def __init__(
        self,
        config: GainSchedulerConfig,
        *,
        corrected_model: tuple[VehicleParameters, float, float] | None = None,
    ) -> None:
        self.config = config
        self._speeds = np.array([p.speed for p in config.points], dtype=float)
        self._corrected_model = corrected_model

    def weights(self, speed: float) -> tuple[tuple[tuple[float, float], ...], str, bool]:
        """Return ``(weights, regime, clamped)`` with convex weights."""

        alpha = float(speed)
        clamped = False
        if math.isnan(alpha) or not math.isfinite(alpha):
            alpha = self.config.speed_min
            clamped = True
        if alpha <= self.config.speed_min:
            clamped = alpha < self.config.speed_min
            return (((self.config.points[0].speed, 1.0),), "below_min", clamped)
        if alpha >= self.config.speed_max:
            clamped = alpha > self.config.speed_max
            return (((self.config.points[-1].speed, 1.0),), "above_max", clamped)

        idx = int(np.searchsorted(self._speeds, alpha, side="right") - 1)
        idx = min(max(idx, 0), len(self.config.points) - 2)
        lo = self.config.points[idx]
        hi = self.config.points[idx + 1]
        span = hi.speed - lo.speed
        lam = 0.0 if span <= 0.0 else (alpha - lo.speed) / span
        if abs(lam) < 1e-12:
            return (((lo.speed, 1.0),), f"point_{idx}", False)
        if abs(lam - 1.0) < 1e-12:
            return (((hi.speed, 1.0),), f"point_{idx + 1}", False)
        return (
            ((lo.speed, 1.0 - lam), (hi.speed, lam)),
            f"segment_{idx}_{idx + 1}",
            False,
        )

    def gains(self, speed: float) -> ScheduledGains:
        weights, regime, clamped = self.weights(speed)
        by_speed = {p.speed: p for p in self.config.points}
        applied_directly = self._corrected_model is not None
        if not applied_directly:
            kp = sum(weight * by_speed[point_speed].kp for point_speed, weight in weights)
            kd = sum(weight * by_speed[point_speed].kd for point_speed, weight in weights)
        else:
            vehicle, omega_n, zeta = self._corrected_model
            # Evaluate the nominal coefficients at the measured speed.  Linear
            # interpolation of gains would not preserve the closed-loop
            # matrix when the plant coefficients are nonlinear in speed.
            speed_used = min(
                max(float(speed), self.config.speed_min), self.config.speed_max
            )
            a = vehicle.yaw_a(speed_used)
            b = vehicle.yaw_b(speed_used)
            kp = omega_n * omega_n / max(b, 1e-12)
            kd = (2.0 * zeta * omega_n - a) / max(b, 1e-12)
        return ScheduledGains(
            speed=float(speed),
            speed_used=min(max(float(speed), self.config.speed_min), self.config.speed_max),
            kp=float(kp),
            kd=float(kd),
            interpolation_weights=weights,
            regime=regime,
            clamped=clamped,
            applied_directly=applied_directly,
        )

    def alpha_dot(self, previous_speed: float, speed: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        return (float(speed) - float(previous_speed)) / float(dt)

    def rate_violation(self, alpha_dot: float) -> bool:
        return abs(alpha_dot) > self.config.gamma_max

    # -- construction helpers ----------------------------------------------

    @classmethod
    def corrected(
        cls,
        vehicle: VehicleParameters | None = None,
        *,
        omega_n: float = 2.0,
        zeta: float = 1.0,
        speeds: Sequence[float] = (0.1, 0.8, 1.0),
        gamma_max: float = 2.0,
    ) -> "GainScheduler":
        """Constant-damping-ratio redesign for the V4 nominal yaw model.

        For ``b(q)=b_0+b_n q^2`` and ``a(q)=a_n q``, choosing
        ``kp = omega_n^2 / b(q)`` and ``kd = (2 zeta omega_n - a(q)) / b(q)``
        gives the same continuous closed-loop matrix at every design point.
        The legacy MATLAB plant has ``b_0=0``; the V4 nominal plant includes the
        low-speed differential-thrust term documented in ``dynamics/model.py``.
        """

        p = vehicle or VehicleParameters()
        points: list[OperatingPoint] = []
        for speed in speeds:
            b = p.yaw_b(speed)
            a = p.yaw_a(speed)
            kp = omega_n * omega_n / b
            kd = (2.0 * zeta * omega_n - a) / b
            points.append(
                OperatingPoint(
                    speed=float(speed),
                    kp=float(kp),
                    kd=float(kd),
                    label=f"u={speed:g}",
                    source=f"corrected(omega_n={omega_n:g},zeta={zeta:g})",
                )
            )
        return cls(
            GainSchedulerConfig(tuple(points), gamma_max=gamma_max),
            corrected_model=(p, float(omega_n), float(zeta)),
        )

    @classmethod
    def matlab_absolute(
        cls,
        *,
        speeds: Sequence[float] = (0.1, 0.8, 1.0),
        gamma_max: float = 2.0,
    ) -> "GainScheduler":
        """Absolute PD gains reconstructed from ``projeto_barco.m``.

        The MATLAB design uses ``K_i = K/q_i^2`` and
        ``Td_i = (2.4/q_i sqrt(b_i K_i) - a_i)/(b_i K_i)``.  The PD law is
        ``K_i (1 + s Td_i)``, hence ``kp=K_i`` and ``kd=K_i Td_i``.
        """

        an, bn, un = 1.6, 0.2, 1.0
        K = 30.0
        points: list[OperatingPoint] = []
        for speed in speeds:
            q = float(speed) / un
            a = an * q
            b = bn * q * q
            kp = K / (q * q)
            td = (2.4 / q * math.sqrt(b * kp) - a) / (b * kp)
            points.append(
                OperatingPoint(
                    speed=float(speed),
                    kp=float(kp),
                    kd=float(kp * td),
                    label=f"MATLAB u={speed:g}",
                    source="matlab_absolute",
                )
            )
        return cls(GainSchedulerConfig(tuple(points), gamma_max=gamma_max))

    @classmethod
    def matlab_constant_zeta(
        cls,
        *,
        speeds: Sequence[float] = (0.1, 0.8, 1.0),
        omega_n: float = math.sqrt(6.0),
        zeta: float = 1.2,
        gamma_max: float = 2.0,
    ) -> "GainScheduler":
        """MATLAB gain *structure* with the ``2.4/q`` inconsistency removed.

        The script keeps ``K_i b_i = K b_n = 6`` but uses ``2.4/q`` in the
        derivative time, so the actual damping ratio is ``1.2/q``.  This
        constructor exposes the corrected constant-zeta version for the audit.
        """

        an, bn, K = 1.6, 0.2, 30.0
        points: list[OperatingPoint] = []
        for speed in speeds:
            q = float(speed)
            a = an * q
            b = bn * q * q
            kp = K / (q * q)
            kd = (2.0 * zeta * omega_n - a) / b
            points.append(
                OperatingPoint(
                    speed=float(speed),
                    kp=float(kp),
                    kd=float(kd),
                    label=f"MATLAB constant-zeta u={speed:g}",
                    source="matlab_constant_zeta",
                )
            )
        return cls(GainSchedulerConfig(tuple(points), gamma_max=gamma_max))

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [
                {
                    "speed": p.speed,
                    "kp": p.kp,
                    "kd": p.kd,
                    "label": p.label,
                    "source": p.source,
                }
                for p in self.config.points
            ],
            "extrapolation": self.config.extrapolation,
            "gamma_max": self.config.gamma_max,
        }


class GainScheduledController:
    """Continuous gain-scheduled heading autopilot."""

    name = "gain_scheduled"

    def __init__(
        self,
        scheduler: GainScheduler | None = None,
        *,
        drive: DifferentialDrive | None = None,
        vehicle: VehicleParameters | None = None,
        delta_max: float = 0.65,
        omega_max: float = 2.0,
        derivative_filter_tau: float = 0.05,
        use_delta: bool = True,
        fixed_gains: tuple[float, float] | None = None,
    ) -> None:
        self.scheduler = scheduler or GainScheduler.corrected()
        self.drive = drive or DifferentialDrive()
        self.vehicle = vehicle or VehicleParameters()
        self.delta_max = float(delta_max)
        self.omega_max = float(omega_max)
        self.derivative_filter_tau = float(derivative_filter_tau)
        self.use_delta = bool(use_delta)
        self.fixed_gains = fixed_gains
        self._telemetry = TelemetryBuffer()
        self.reset()

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None:
        self._previous_error = 0.0
        self._filtered_error_rate = 0.0
        self._previous_speed = measurement.u if measurement is not None else 0.0
        self._initialised = measurement is not None
        self._telemetry.set(
            time=0.0,
            speed=0.0,
            kp=0.0,
            kd=0.0,
            error=0.0,
            error_rate=0.0,
            delta_cmd=0.0,
            regime="reset",
            alpha_dot=0.0,
            rate_violation=False,
            interpolation_weights=[],
            applied_directly=False,
        )

    def update(self, context: ControllerContext) -> ControlCommand:
        m = context.measurement
        g = context.guidance
        dt = max(context.dt, 1e-9)
        error = wrap_angle(g.heading_ref - m.pose.theta)
        if not self._initialised:
            self._previous_error = error
            self._previous_speed = m.u
            self._initialised = True
        raw_rate = wrap_angle(error - self._previous_error) / dt
        alpha = 1.0 - math.exp(-dt / max(self.derivative_filter_tau, 1e-9))
        self._filtered_error_rate += alpha * (raw_rate - self._filtered_error_rate)
        alpha_dot = self.scheduler.alpha_dot(self._previous_speed, m.u, dt)

        if self.fixed_gains is None:
            sched = self.scheduler.gains(m.u)
            kp, kd = sched.kp, sched.kd
            interpolation_weights = list(sched.interpolation_weights)
            applied_directly = sched.applied_directly
            regime = sched.regime
            clamped = sched.clamped
        else:
            kp, kd = self.fixed_gains
            interpolation_weights = [(self.scheduler.config.points[-1].speed, 1.0)]
            applied_directly = False
            regime = "fixed"
            clamped = False
            sched = self.scheduler.gains(m.u)

        delta_cmd = kp * error + kd * self._filtered_error_rate
        delta_saturated = abs(delta_cmd) > self.delta_max
        if delta_saturated:
            delta_cmd = math.copysign(self.delta_max, delta_cmd)

        # Kinematic fallback: steady-state inverse of the yaw model,
        # r_ss = (b(q)/a(q)) delta.  The dynamic plant uses delta_rudder
        # directly, so this value is only used by the kinematic baseline.
        b = self.vehicle.yaw_b(m.u)
        a = max(self.vehicle.yaw_a(m.u), 1e-9)
        omega_cmd = delta_cmd * b / a
        omega_cmd = max(-self.omega_max, min(self.omega_max, omega_cmd))
        wheels = self.drive.inverse(g.speed, omega_cmd)

        self._previous_error = error
        self._previous_speed = m.u
        self._telemetry.set(
            time=m.time,
            speed=m.u,
            kp=kp,
            kd=kd,
            error=error,
            error_rate=self._filtered_error_rate,
            delta_cmd=delta_cmd,
            omega_cmd=omega_cmd,
            regime=regime,
            alpha_dot=alpha_dot,
            rate_violation=self.scheduler.rate_violation(alpha_dot),
            interpolation_weights=interpolation_weights,
            applied_directly=applied_directly,
            clamped=clamped,
            saturated=bool(delta_saturated),
        )
        return ControlCommand(
            v=g.speed,
            omega=omega_cmd,
            wheels=wheels,
            saturated=bool(delta_saturated),
            delta_rudder=delta_cmd if self.use_delta else None,
            target=g.target,
            metadata={
                "kp": kp,
                "kd": kd,
                "alpha": m.u,
                "alpha_dot": alpha_dot,
                "delta_cmd": delta_cmd,
                "regime": regime,
            },
        )

    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.get()
