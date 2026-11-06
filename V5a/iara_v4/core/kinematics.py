"""Ideal differential-drive kinematics for the IARA USV.

The motor is deliberately *not* modelled.  The scientific plant receives ideal
left/right thruster speeds and applies the exact differential-drive map

    v_left  = V - omega * L / 2
    v_right = V + omega * L / 2

with a common-mode saturation that preserves the commanded curvature.  This is
the standard unicycle model used for marine craft at low speed (Fossen, 2011,
ch. 2; Antonelli, 2014, ch. 2) and is the model requested for V3.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import ControlCommand, Pose2D, Twist, Vec2, WheelSpeeds, wrap_angle

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class DriveLimits:
    """Actuator and workspace limits in SI units."""

    track_width: float = 0.132
    max_wheel_speed: float = 0.50
    max_linear_speed: float = 0.50
    max_angular_speed: float = 2.00

    def __post_init__(self) -> None:
        if self.track_width <= 0.0:
            raise ValueError("track_width must be positive")
        if min(self.max_wheel_speed, self.max_linear_speed, self.max_angular_speed) <= 0.0:
            raise ValueError("speed limits must be positive")


class DifferentialDrive:
    """Exact ideal differential drive with curvature-preserving saturation."""

    def __init__(self, limits: DriveLimits | None = None) -> None:
        self.limits = limits or DriveLimits()

    # -- kinematic maps -----------------------------------------------------

    def inverse(self, v: float, omega: float) -> WheelSpeeds:
        """Map body twist to ideal wheel speeds (no saturation)."""

        half = 0.5 * self.limits.track_width
        return WheelSpeeds(left=v - omega * half, right=v + omega * half)

    def forward(self, wheels: WheelSpeeds) -> Twist:
        """Map ideal wheel speeds to body twist."""

        v = 0.5 * (wheels.left + wheels.right)
        omega = (wheels.right - wheels.left) / self.limits.track_width
        return Twist(v=v, omega=omega)

    def saturate(self, v: float, omega: float) -> tuple[Twist, WheelSpeeds, bool]:
        """Clamp the twist and, if needed, scale both wheels equally.

        Scaling both wheels by the same factor preserves ``omega / v`` and hence
        the instantaneous curvature.  Clamping each wheel independently would
        silently change the path curvature and is therefore not used.
        """

        v = max(-self.limits.max_linear_speed, min(self.limits.max_linear_speed, v))
        omega = max(-self.limits.max_angular_speed, min(self.limits.max_angular_speed, omega))
        wheels = self.inverse(v, omega)
        peak = wheels.max_abs
        saturated = peak > self.limits.max_wheel_speed + _EPS
        if saturated:
            scale = self.limits.max_wheel_speed / peak
            v *= scale
            omega *= scale
            wheels = WheelSpeeds(wheels.left * scale, wheels.right * scale)
        return Twist(v=v, omega=omega), wheels, saturated

    def command(
        self, v: float, omega: float, target: Vec2 | None = None
    ) -> ControlCommand:
        twist, wheels, saturated = self.saturate(v, omega)
        return ControlCommand(
            v=twist.v,
            omega=twist.omega,
            wheels=wheels,
            saturated=saturated,
            target=target,
        )

    # -- exact integration --------------------------------------------------

    def integrate(self, pose: Pose2D, twist: Twist, dt: float) -> Pose2D:
        """Exact constant-twist integration over ``dt``.

        For ``omega != 0`` the unicycle follows a circular arc of radius
        ``R = v / omega``:

            x(t) = x0 + R [sin(theta0 + omega t) - sin(theta0)]
            y(t) = y0 - R [cos(theta0 + omega t) - cos(theta0)]
            theta(t) = theta0 + omega t

        For ``omega = 0`` this reduces to a straight line.  Using the closed form
        removes the O(dt^2) drift of explicit Euler integration and makes the
        simulator exactly reproducible for a given command sequence.
        """

        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if dt == 0.0:
            return pose

        theta0 = pose.theta
        dtheta = twist.omega * dt
        if abs(twist.omega) < 1e-10:
            x = pose.x + twist.v * math.cos(theta0) * dt
            y = pose.y + twist.v * math.sin(theta0) * dt
        else:
            radius = twist.v / twist.omega
            x = pose.x + radius * (math.sin(theta0 + dtheta) - math.sin(theta0))
            y = pose.y - radius * (math.cos(theta0 + dtheta) - math.cos(theta0))
        return Pose2D(x, y, wrap_angle(theta0 + dtheta))

    def integrate_wheels(self, pose: Pose2D, wheels: WheelSpeeds, dt: float) -> Pose2D:
        return self.integrate(pose, self.forward(wheels), dt)

    # -- feasibility --------------------------------------------------------

    @staticmethod
    def curvature(v: float, omega: float) -> float:
        if abs(v) < _EPS:
            return math.copysign(math.inf, omega) if abs(omega) > _EPS else 0.0
        return omega / v

    def max_angular_rate(self, v: float) -> float:
        """Largest feasible ``|omega|`` at forward speed ``v``."""

        v = abs(v)
        if v > self.limits.max_wheel_speed:
            return 0.0
        wheel_limited = 2.0 * (self.limits.max_wheel_speed - v) / self.limits.track_width
        return max(0.0, min(self.limits.max_angular_speed, wheel_limited))

    def max_speed_for_curvature(self, curvature: float) -> float:
        """Largest feasible forward speed for a given path curvature."""

        k = abs(curvature)
        if not math.isfinite(k):
            return 0.0
        candidates = [self.limits.max_linear_speed]
        if k > _EPS:
            candidates.append(self.limits.max_angular_speed / k)
            candidates.append(
                self.limits.max_wheel_speed / (1.0 + 0.5 * k * self.limits.track_width)
            )
        return max(0.0, min(candidates))

    def is_feasible(self, v: float, omega: float) -> bool:
        if abs(v) > self.limits.max_linear_speed + _EPS:
            return False
        if abs(omega) > self.limits.max_angular_speed + _EPS:
            return False
        return self.inverse(v, omega).max_abs <= self.limits.max_wheel_speed + _EPS
