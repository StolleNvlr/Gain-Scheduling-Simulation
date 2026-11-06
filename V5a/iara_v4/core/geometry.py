"""Exact planar geometry for planning and collision assessment.

The planner uses a disc model of the vessel.  The simulator additionally checks
the true oriented-rectangle footprint.  Keeping both models explicit avoids the
usual mistake of planning with one geometry and evaluating with another without
documenting the conservatism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .types import Pose2D, Vec2

EPS = 1e-12


class Obstacle(Protocol):
    """Minimal obstacle interface used by the planner and the range sensor."""

    name: str

    def distance_to_point(self, point: Vec2) -> float: ...

    def distance_to_segment(self, a: Vec2, b: Vec2) -> float: ...

    def contains(self, point: Vec2) -> bool: ...

    def bounding_box(self) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True, slots=True)
class RectangleObstacle:
    """Axis-aligned rectangular obstacle (the CoppeliaSim cuboids)."""

    cx: float
    cy: float
    width: float
    height: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("rectangle dimensions must be positive")

    @property
    def min_x(self) -> float:
        return self.cx - 0.5 * self.width

    @property
    def max_x(self) -> float:
        return self.cx + 0.5 * self.width

    @property
    def min_y(self) -> float:
        return self.cy - 0.5 * self.height

    @property
    def max_y(self) -> float:
        return self.cy + 0.5 * self.height

    def corners(self) -> list[Vec2]:
        return [
            Vec2(self.min_x, self.min_y),
            Vec2(self.max_x, self.min_y),
            Vec2(self.max_x, self.max_y),
            Vec2(self.min_x, self.max_y),
        ]

    def contains(self, point: Vec2, eps: float = EPS) -> bool:
        return (
            self.min_x - eps <= point.x <= self.max_x + eps
            and self.min_y - eps <= point.y <= self.max_y + eps
        )

    def distance_to_point(self, point: Vec2) -> float:
        dx = max(self.min_x - point.x, 0.0, point.x - self.max_x)
        dy = max(self.min_y - point.y, 0.0, point.y - self.max_y)
        return math.hypot(dx, dy)

    def distance_to_segment(self, a: Vec2, b: Vec2) -> float:
        return segment_rectangle_distance(a, b, self)

    def bounding_box(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def inflated(self, margin: float) -> "RectangleObstacle":
        return RectangleObstacle(
            self.cx,
            self.cy,
            self.width + 2.0 * margin,
            self.height + 2.0 * margin,
            self.name,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "kind": "rectangle",
            "name": self.name,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RectangleObstacle":
        return cls(
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=float(data["width"]),
            height=float(data["height"]),
            name=str(data.get("name", "")),
        )


@dataclass(frozen=True, slots=True)
class CircleObstacle:
    """Circular obstacle."""

    cx: float
    cy: float
    radius: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("circle radius must be positive")

    @property
    def center(self) -> Vec2:
        return Vec2(self.cx, self.cy)

    def contains(self, point: Vec2, eps: float = EPS) -> bool:
        return point.distance_to(self.center) <= self.radius + eps

    def distance_to_point(self, point: Vec2) -> float:
        return max(0.0, point.distance_to(self.center) - self.radius)

    def distance_to_segment(self, a: Vec2, b: Vec2) -> float:
        return segment_circle_distance(a, b, self)

    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.radius,
            self.cy - self.radius,
            self.cx + self.radius,
            self.cy + self.radius,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "kind": "circle",
            "name": self.name,
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircleObstacle":
        return cls(
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            radius=float(data["radius"]),
            name=str(data.get("name", "")),
        )


# ---------------------------------------------------------------------------
# Point / segment primitives
# ---------------------------------------------------------------------------


def point_segment_projection(
    point: Vec2, a: Vec2, b: Vec2
) -> tuple[float, Vec2, float]:
    """Return ``(t, closest_point, distance)`` for the projection of ``point``.

    ``t`` is clamped to [0, 1].  Degenerate segments are handled explicitly.
    """

    ab = b - a
    denom = ab.norm2()
    if denom < EPS:
        return 0.0, a, point.distance_to(a)
    t = (point - a).dot(ab) / denom
    t = min(1.0, max(0.0, t))
    closest = a + ab * t
    return t, closest, point.distance_to(closest)


def point_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> float:
    return point_segment_projection(point, a, b)[2]


def _orientation(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b - a).cross(c - a)


def _on_segment(a: Vec2, b: Vec2, p: Vec2, eps: float = EPS) -> bool:
    return (
        min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps
        and min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps
    )


def segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2, eps: float = EPS) -> bool:
    """Robust closed-segment intersection test (including touching/collinear)."""

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    if abs(o1) <= eps and _on_segment(a, b, c):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b):
        return True
    return False


def segment_segment_distance(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> float:
    """Exact minimum distance between two closed 2-D segments."""

    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


# ---------------------------------------------------------------------------
# Segment / rectangle and segment / circle
# ---------------------------------------------------------------------------


def segment_rectangle_intersects(a: Vec2, b: Vec2, rect: RectangleObstacle) -> bool:
    if rect.contains(a) or rect.contains(b):
        return True
    corners = rect.corners()
    for i in range(4):
        if segments_intersect(a, b, corners[i], corners[(i + 1) % 4]):
            return True
    return False


def segment_rectangle_distance(a: Vec2, b: Vec2, rect: RectangleObstacle) -> float:
    if segment_rectangle_intersects(a, b, rect):
        return 0.0
    corners = rect.corners()
    return min(
        segment_segment_distance(a, b, corners[i], corners[(i + 1) % 4])
        for i in range(4)
    )


def segment_circle_distance(a: Vec2, b: Vec2, circle: CircleObstacle) -> float:
    d = point_segment_distance(circle.center, a, b)
    return max(0.0, d - circle.radius)


# ---------------------------------------------------------------------------
# Point / segment clearance against a collection of obstacles
# ---------------------------------------------------------------------------


def point_clearance(point: Vec2, obstacles: Iterable[Obstacle]) -> float:
    return min((ob.distance_to_point(point) for ob in obstacles), default=math.inf)


def segment_clearance(a: Vec2, b: Vec2, obstacles: Iterable[Obstacle]) -> float:
    return min((ob.distance_to_segment(a, b) for ob in obstacles), default=math.inf)


def line_of_sight(
    a: Vec2, b: Vec2, obstacles: Iterable[Obstacle], clearance: float = 0.0
) -> bool:
    return segment_clearance(a, b, obstacles) >= clearance


def polyline_clearance(points: Sequence[Vec2], obstacles: Iterable[Obstacle]) -> float:
    if len(points) < 2:
        return point_clearance(points[0], obstacles) if points else math.inf
    return min(
        segment_clearance(points[i], points[i + 1], obstacles)
        for i in range(len(points) - 1)
    )


def path_length(points: Sequence[Vec2]) -> float:
    return sum(points[i].distance_to(points[i + 1]) for i in range(len(points) - 1))


# ---------------------------------------------------------------------------
# Polygon helpers for the true vessel footprint
# ---------------------------------------------------------------------------


def oriented_rectangle_corners(
    pose: Pose2D, length: float, width: float
) -> list[Vec2]:
    """Corners of a vessel footprint, CCW, with ``length`` along local +x."""

    hl, hw = 0.5 * length, 0.5 * width
    local = [
        Vec2(-hl, -hw),
        Vec2(hl, -hw),
        Vec2(hl, hw),
        Vec2(-hl, hw),
    ]
    return [pose.local_to_world(p) for p in local]


def point_in_polygon(point: Vec2, polygon: Sequence[Vec2], eps: float = EPS) -> bool:
    """Ray-casting test for a simple polygon; boundary counts as inside."""

    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        if point_segment_distance(point, polygon[i], polygon[(i + 1) % n]) <= eps:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        pi, pj = polygon[i], polygon[j]
        if (pi.y > point.y) != (pj.y > point.y):
            x_intersect = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x
            if point.x < x_intersect:
                inside = not inside
        j = i
    return inside


def _project_polygon(polygon: Sequence[Vec2], axis: Vec2) -> tuple[float, float]:
    projections = [p.dot(axis) for p in polygon]
    return min(projections), max(projections)


def polygons_intersect(
    poly_a: Sequence[Vec2], poly_b: Sequence[Vec2], eps: float = EPS
) -> bool:
    """Separating-axis test for two convex polygons."""

    for polygon in (poly_a, poly_b):
        for i in range(len(polygon)):
            edge = polygon[(i + 1) % len(polygon)] - polygon[i]
            axis = Vec2(-edge.y, edge.x).normalized()
            if axis.norm() < EPS:
                continue
            a_min, a_max = _project_polygon(poly_a, axis)
            b_min, b_max = _project_polygon(poly_b, axis)
            if a_max < b_min - eps or b_max < a_min - eps:
                return False
    return True


def point_polygon_distance(point: Vec2, polygon: Sequence[Vec2]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        point_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def polygon_polygon_distance(poly_a: Sequence[Vec2], poly_b: Sequence[Vec2]) -> float:
    if polygons_intersect(poly_a, poly_b):
        return 0.0
    best = math.inf
    for i in range(len(poly_a)):
        for j in range(len(poly_b)):
            best = min(
                best,
                segment_segment_distance(
                    poly_a[i],
                    poly_a[(i + 1) % len(poly_a)],
                    poly_b[j],
                    poly_b[(j + 1) % len(poly_b)],
                ),
            )
    return best


def oriented_rectangle_intersects_rectangle(
    pose: Pose2D, length: float, width: float, rect: RectangleObstacle
) -> bool:
    boat = oriented_rectangle_corners(pose, length, width)
    return polygons_intersect(boat, rect.corners())


def oriented_rectangle_intersects_circle(
    pose: Pose2D, length: float, width: float, circle: CircleObstacle
) -> bool:
    local = pose.world_to_local(circle.center)
    hl, hw = 0.5 * length, 0.5 * width
    dx = max(abs(local.x) - hl, 0.0)
    dy = max(abs(local.y) - hw, 0.0)
    return math.hypot(dx, dy) <= circle.radius + EPS


def oriented_rectangle_clearance(
    pose: Pose2D,
    length: float,
    width: float,
    obstacles: Iterable[Obstacle],
) -> float:
    """Minimum distance from the true oriented footprint to any obstacle.

    Returns ``math.inf`` when there are no obstacles.  Negative values denote
    intersection, zero denotes contact, and positive values denote clearance.
    """

    boat = oriented_rectangle_corners(pose, length, width)
    best = math.inf
    for ob in obstacles:
        if isinstance(ob, RectangleObstacle):
            d = (
                -EPS
                if polygons_intersect(boat, ob.corners())
                else polygon_polygon_distance(boat, ob.corners())
            )
        elif isinstance(ob, CircleObstacle):
            if point_in_polygon(ob.center, boat):
                d = -max(
                    0.0,
                    ob.radius
                    - min(
                        point_segment_distance(
                            ob.center, boat[i], boat[(i + 1) % 4]
                        )
                        for i in range(4)
                    ),
                )
            else:
                d = max(
                    0.0,
                    min(
                        point_segment_distance(ob.center, boat[i], boat[(i + 1) % 4])
                        for i in range(4)
                    )
                    - ob.radius,
                )
        else:  # pragma: no cover - defensive extension point
            d = ob.distance_to_point(pose.position)
        best = min(best, d)
    return best


def footprint_collision(
    pose: Pose2D,
    length: float,
    width: float,
    obstacles: Iterable[Obstacle],
) -> bool:
    for ob in obstacles:
        if isinstance(ob, RectangleObstacle):
            if oriented_rectangle_intersects_rectangle(pose, length, width, ob):
                return True
        elif isinstance(ob, CircleObstacle):
            if oriented_rectangle_intersects_circle(pose, length, width, ob):
                return True
        elif ob.contains(pose.position):
            return True
    return False
