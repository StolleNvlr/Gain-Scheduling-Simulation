"""Pure Pursuit and conventional PD heading baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.kinematics import DifferentialDrive
from ..core.types import ControlCommand, Measurement, Vec2, wrap_angle
from .base import ControllerContext, TelemetryBuffer
from .pd import PDConfig, PathPD
from .pure_pursuit import PurePursuit, PurePursuitConfig


class PurePursuitController:
    """Pure Pursuit guidance expressed through the common controller API."""

    name = "pure_pursuit"

    def __init__(
        self,
        config: PurePursuitConfig | None = None,
        drive: DifferentialDrive | None = None,
    ) -> None:
        self.inner = PurePursuit(config, drive)
        self.drive = self.inner.drive
        self._telemetry = TelemetryBuffer()
        self.reset()

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None:
        self.inner.reset()
        self._telemetry.set(time=0.0, alpha=0.0, kappa=0.0, lookahead=0.0)

    def update(self, context: ControllerContext) -> ControlCommand:
        m = context.measurement
        g = context.guidance
        if g.speed <= 0.0:
            command = self.drive.command(0.0, 0.0, target=g.target)
        else:
            command = self.inner.command_to_point(m.pose, g.target, g.speed)
        dx = g.target.x - m.pose.x
        dy = g.target.y - m.pose.y
        alpha = wrap_angle(math.atan2(dy, dx) - m.pose.theta)
        distance = max(math.hypot(dx, dy), 1e-9)
        kappa = 2.0 * math.sin(alpha) / max(distance, self.inner.config.lookahead_min)
        self._telemetry.set(
            time=m.time,
            alpha=alpha,
            kappa=kappa,
            lookahead=g.lookahead,
            speed=g.speed,
            delta_cmd=None,
        )
        return command

    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.get()


class PathPDController:
    """Cross-track/heading PD adapter over the shared guidance layer.

    The V3 ``PathPD`` law is reused, but the speed reference comes from the
    controller-independent guidance layer instead of being recomputed here.
    That removes a hidden difference between the PD and Pure Pursuit baselines
    in the V4 comparison.  The output is a body twist; the dynamic plant maps it
    to a rudder angle through the shared low-level yaw-rate servo.
    """

    name = "path_pd"

    def __init__(
        self,
        config: PDConfig | None = None,
        drive: DifferentialDrive | None = None,
    ) -> None:
        self.inner = PathPD(config, drive)
        self.config = self.inner.config
        self.drive = self.inner.drive
        self._telemetry = TelemetryBuffer()
        self.reset()

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None:
        self._telemetry.set(
            time=0.0,
            k_y=self.config.k_y,
            k_theta=self.config.k_theta,
            cross_track=0.0,
            heading_error=0.0,
            omega_cmd=0.0,
        )

    def update(self, context: ControllerContext) -> ControlCommand:
        m = context.measurement
        g = context.guidance
        projection = context.projection
        tangent = context.path.tangent_at(projection.s)
        normal = Vec2(-tangent.y, tangent.x)
        cross_track = (m.pose.position - projection.point).dot(normal)
        heading_error = wrap_angle(m.pose.theta - g.heading_ref)
        omega = -self.config.k_y * cross_track - self.config.k_theta * heading_error
        omega = max(-2.0, min(2.0, omega))
        saturated = abs(omega) >= 2.0 - 1e-12
        wheels = self.drive.inverse(g.speed, omega)
        self._telemetry.set(
            time=m.time,
            k_y=self.config.k_y,
            k_theta=self.config.k_theta,
            cross_track=cross_track,
            heading_error=heading_error,
            omega_cmd=omega,
            saturated=saturated,
        )
        return ControlCommand(
            v=g.speed,
            omega=omega,
            wheels=wheels,
            saturated=saturated,
            target=g.target,
            metadata={
                "k_y": self.config.k_y,
                "k_theta": self.config.k_theta,
                "cross_track": cross_track,
                "heading_error": heading_error,
            },
        )

    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.get()


@dataclass(frozen=True, slots=True)
class PDHeadingConfig:
    kp: float = 1.2
    kd: float = 0.35
    omega_max: float = 1.5
    derivative_filter_tau: float = 0.05

    def __post_init__(self) -> None:
        if self.kp < 0.0 or self.kd < 0.0:
            raise ValueError("PD gains must be non-negative")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        if self.derivative_filter_tau < 0.0:
            raise ValueError("derivative filter time must be non-negative")


class PDHeadingController:
    """PD on heading error with a filtered derivative and yaw-rate damping."""

    name = "pd_heading"

    def __init__(
        self,
        config: PDHeadingConfig | None = None,
        drive: DifferentialDrive | None = None,
    ) -> None:
        self.config = config or PDHeadingConfig()
        self.drive = drive or DifferentialDrive()
        self._telemetry = TelemetryBuffer()
        self.reset()

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None:
        self._previous_error = 0.0
        self._filtered_rate = 0.0
        self._initialised = measurement is not None
        self._telemetry.set(time=0.0, kp=self.config.kp, kd=self.config.kd, error=0.0)

    def update(self, context: ControllerContext) -> ControlCommand:
        m = context.measurement
        g = context.guidance
        dt = max(context.dt, 1e-9)
        error = wrap_angle(g.heading_ref - m.pose.theta)
        if not self._initialised:
            self._previous_error = error
            self._initialised = True
        raw_rate = wrap_angle(error - self._previous_error) / dt
        alpha = 1.0 - math.exp(-dt / max(self.config.derivative_filter_tau, 1e-9))
        self._filtered_rate += alpha * (raw_rate - self._filtered_rate)
        omega = self.config.kp * error + self.config.kd * self._filtered_rate
        saturated = abs(omega) > self.config.omega_max
        omega = max(-self.config.omega_max, min(self.config.omega_max, omega))
        wheels = self.drive.inverse(g.speed, omega)
        self._previous_error = error
        self._telemetry.set(
            time=m.time,
            kp=self.config.kp,
            kd=self.config.kd,
            error=error,
            error_rate=self._filtered_rate,
            omega_cmd=omega,
            saturated=saturated,
        )
        return ControlCommand(
            v=g.speed,
            omega=omega,
            wheels=wheels,
            saturated=saturated,
            target=g.target,
            metadata={"kp": self.config.kp, "kd": self.config.kd, "error": error},
        )

    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.get()
