"""Small, dependency-free value objects used across the stack."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterator

_TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi).

    The half-open convention avoids the common ``+pi``/``-pi`` ambiguity when
    comparing headings and when integrating a constant angular rate.
    """

    return (float(angle) + math.pi) % _TWO_PI - math.pi


@dataclass(frozen=True, slots=True)
class Vec2:
    """Two-dimensional Euclidean vector."""

    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        if scalar == 0.0:
            raise ZeroDivisionError("Vec2 division by zero")
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self, eps: float = 1e-12) -> "Vec2":
        n = self.norm()
        if n < eps:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def rotated(self, angle: float) -> "Vec2":
        c, s = math.cos(angle), math.sin(angle)
        return Vec2(c * self.x - s * self.y, s * self.x + c * self.y)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).norm()

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Planar pose ``(x, y, theta)`` with ``theta`` in [-pi, pi)."""

    x: float
    y: float
    theta: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", wrap_angle(self.theta))

    @property
    def position(self) -> Vec2:
        return Vec2(self.x, self.y)

    def translated(self, delta: Vec2) -> "Pose2D":
        return replace(self, x=self.x + delta.x, y=self.y + delta.y)

    def rotated(self, angle: float) -> "Pose2D":
        return replace(self, theta=wrap_angle(self.theta + angle))

    def local_to_world(self, point: Vec2) -> Vec2:
        return point.rotated(self.theta) + self.position

    def world_to_local(self, point: Vec2) -> Vec2:
        return (point - self.position).rotated(-self.theta)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)


@dataclass(frozen=True, slots=True)
class Twist:
    """Body-frame twist: forward speed ``v`` (m/s) and yaw rate ``omega`` (rad/s)."""

    v: float
    omega: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.v, self.omega)


@dataclass(frozen=True, slots=True)
class WheelSpeeds:
    """Left/right wheel (or thruster) linear speeds in m/s."""

    left: float
    right: float

    @property
    def max_abs(self) -> float:
        return max(abs(self.left), abs(self.right))

    def as_tuple(self) -> tuple[float, float]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Command issued by a controller.

    ``v`` and ``omega`` are the body-frame twist references used by the
    kinematic plant.  The dynamic plant additionally accepts an optional
    rudder-angle command ``delta_rudder`` (rad) and, for direct-adaptive
    controllers, optional generalized forces ``tau_u`` and ``tau_r``.  Keeping
    all three forms in one record makes the controller/plant interface explicit
    and lets the same controller be evaluated on the kinematic and dynamic
    plants without hidden unit conversions.
    """

    v: float
    omega: float
    wheels: WheelSpeeds | None = None
    saturated: bool = False
    delta_rudder: float | None = None
    tau_u: float | None = None
    tau_r: float | None = None
    target: Vec2 | None = None
    metadata: dict[str, float] = field(default_factory=dict, compare=False)

    def as_tuple(self) -> tuple[float, float, float, float]:
        if self.wheels is None:
            return (self.v, self.omega, math.nan, math.nan)
        return (self.v, self.omega, self.wheels.left, self.wheels.right)


@dataclass(frozen=True, slots=True)
class Measurement:
    """Sampled, possibly delayed and noisy measurement delivered to a controller.

    ``time`` is the time at which the controller receives the sample;
    ``source_time`` is the time at which the underlying plant state existed.
    The difference is the measurement latency and is logged in the telemetry.
    """

    time: float
    source_time: float
    pose: Pose2D
    u: float
    v: float
    r: float
    dropped: bool = False

    @property
    def latency(self) -> float:
        return max(0.0, self.time - self.source_time)

    def as_state(self) -> tuple[float, float, float, float, float, float]:
        return (self.pose.x, self.pose.y, self.pose.theta, self.u, self.v, self.r)
