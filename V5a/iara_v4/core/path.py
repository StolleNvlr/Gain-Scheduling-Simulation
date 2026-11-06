"""Arc-length parameterised paths, projection and C2 smoothing.

The global planners return polylines.  A natural cubic spline is fitted in the
chord-length parameter and then validated for clearance and curvature before it
is accepted.  If validation fails, the polyline is returned unchanged and the
caller can still execute it; this fail-safe behaviour is important for a
reproducible experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from .geometry import Obstacle, line_of_sight, polyline_clearance
from .types import Vec2

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PathProjection:
    """Closest point on a path to an arbitrary query point."""

    s: float
    point: Vec2
    tangent: Vec2
    curvature: float
    distance: float
    segment_index: int
    t: float


@dataclass(frozen=True, slots=True)
class SmoothingResult:
    path: "Path"
    smoothed: bool
    reason: str
    max_curvature: float
    min_clearance: float
    shortcut_points: int


def _as_vec2(points: Iterable[Sequence[float] | Vec2]) -> list[Vec2]:
    result: list[Vec2] = []
    for p in points:
        if isinstance(p, Vec2):
            q = p
        else:
            q = Vec2(float(p[0]), float(p[1]))
        if not result or q.distance_to(result[-1]) > 1e-9:
            result.append(q)
    return result


def signed_curvature(a: Vec2, b: Vec2, c: Vec2) -> float:
    """Signed Menger curvature of three points.

    Positive values turn left (counter-clockwise).  The formula is the
    circumcircle curvature ``1 / R`` with the sign of the turn.
    """

    ab = b - a
    bc = c - b
    ca = c - a
    denom = ab.norm() * bc.norm() * ca.norm()
    if denom < _EPS:
        return 0.0
    return 2.0 * ab.cross(bc) / denom


def polyline_curvatures(points: Sequence[Vec2]) -> np.ndarray:
    """Curvature at every vertex.

    Interior vertices use the Menger curvature of their two adjacent edges.
    Endpoints have no defined turn and are assigned zero.  Copying the
    neighbour's value (a common shortcut) would make a three-point path look
    uniformly curved and would needlessly slow the controller after a corner.
    """

    n = len(points)
    kappa = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        kappa[i] = signed_curvature(points[i - 1], points[i], points[i + 1])
    return kappa


class Path:
    """A polyline with arc-length lookup and exact projection."""

    def __init__(
        self,
        points: Sequence[Sequence[float] | Vec2],
        *,
        smoothed: bool = False,
        name: str = "",
    ) -> None:
        pts = _as_vec2(points)
        if len(pts) < 2:
            raise ValueError("a path needs at least two distinct points")
        self._points = np.asarray([[p.x, p.y] for p in pts], dtype=float)
        self._vecs = pts
        seg = np.linalg.norm(np.diff(self._points, axis=0), axis=1)
        self._cum = np.concatenate(([0.0], np.cumsum(seg)))
        self._length = float(self._cum[-1])
        if self._length < _EPS:
            raise ValueError("path length is zero")
        self._curvatures = polyline_curvatures(pts)
        self.smoothed = smoothed
        self.name = name

    # -- basic properties ---------------------------------------------------

    @property
    def points(self) -> list[Vec2]:
        return list(self._vecs)

    @property
    def xy(self) -> np.ndarray:
        return self._points.copy()

    @property
    def length(self) -> float:
        return self._length

    @property
    def cumulative_s(self) -> np.ndarray:
        return self._cum.copy()

    @property
    def start(self) -> Vec2:
        return self._vecs[0]

    @property
    def end(self) -> Vec2:
        return self._vecs[-1]

    def __len__(self) -> int:
        return len(self._vecs)

    def __iter__(self):
        return iter(self._vecs)

    # -- lookup -------------------------------------------------------------

    def _segment(self, s: float) -> tuple[int, float]:
        if s <= 0.0:
            return 0, 0.0
        if s >= self._length:
            return len(self._vecs) - 2, 1.0
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        i = min(max(i, 0), len(self._vecs) - 2)
        seg_len = self._cum[i + 1] - self._cum[i]
        t = 0.0 if seg_len < _EPS else (s - self._cum[i]) / seg_len
        return i, min(1.0, max(0.0, t))

    def point_at(self, s: float) -> Vec2:
        i, t = self._segment(s)
        return self._vecs[i] + (self._vecs[i + 1] - self._vecs[i]) * t

    def tangent_at(self, s: float) -> Vec2:
        i, _ = self._segment(s)
        return (self._vecs[i + 1] - self._vecs[i]).normalized()

    def heading_at(self, s: float) -> float:
        return self.tangent_at(s).angle()

    def curvature_at(self, s: float) -> float:
        i, t = self._segment(s)
        return float((1.0 - t) * self._curvatures[i] + t * self._curvatures[i + 1])

    def project(self, point: Vec2) -> PathProjection:
        """Exact closest-point projection onto the polyline."""

        best_dist = math.inf
        best: tuple[int, float, Vec2] | None = None
        for i in range(len(self._vecs) - 1):
            a, b = self._vecs[i], self._vecs[i + 1]
            ab = b - a
            denom = ab.norm2()
            if denom < _EPS:
                continue
            t = (point - a).dot(ab) / denom
            t = min(1.0, max(0.0, t))
            q = a + ab * t
            d = point.distance_to(q)
            if d < best_dist:
                best_dist = d
                best = (i, t, q)
        if best is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError("degenerate path")
        i, t, q = best
        s = self._cum[i] + t * (self._cum[i + 1] - self._cum[i])
        tangent = (self._vecs[i + 1] - self._vecs[i]).normalized()
        return PathProjection(
            s=s,
            point=q,
            tangent=tangent,
            curvature=self.curvature_at(s),
            distance=best_dist,
            segment_index=i,
            t=t,
        )

    # -- transformations ----------------------------------------------------

    def resample(self, ds: float) -> "Path":
        if ds <= 0.0:
            raise ValueError("ds must be positive")
        n = max(2, int(math.ceil(self._length / ds)) + 1)
        s_values = np.linspace(0.0, self._length, n)
        pts = [self.point_at(float(s)) for s in s_values]
        return Path(pts, smoothed=self.smoothed, name=self.name)

    def subpath(self, s0: float, s1: float) -> "Path":
        s0 = max(0.0, min(self._length, s0))
        s1 = max(0.0, min(self._length, s1))
        if s1 <= s0 + _EPS:
            raise ValueError("subpath requires s1 > s0")
        ds = max(0.01, (s1 - s0) / 200.0)
        n = max(2, int(math.ceil((s1 - s0) / ds)) + 1)
        pts = [self.point_at(float(s)) for s in np.linspace(s0, s1, n)]
        return Path(pts, smoothed=self.smoothed, name=self.name)

    def reversed(self) -> "Path":
        return Path(self._vecs[::-1], smoothed=self.smoothed, name=self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "smoothed": self.smoothed,
            "points": [[p.x, p.y] for p in self._vecs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Path":
        return cls(
            data["points"],
            smoothed=bool(data.get("smoothed", False)),
            name=str(data.get("name", "")),
        )


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def _greedy_shortcut(
    points: Sequence[Vec2],
    obstacles: Iterable[Obstacle],
    clearance: float,
) -> list[Vec2]:
    """Keep the farthest visible vertex from each current vertex."""

    if len(points) <= 2:
        return list(points)
    kept: list[Vec2] = [points[0]]
    i = 0
    n = len(points)
    while i < n - 1:
        chosen = i + 1
        for j in range(n - 1, i, -1):
            if line_of_sight(points[i], points[j], obstacles, clearance):
                chosen = j
                break
        kept.append(points[chosen])
        i = chosen
    return kept


def _cubic_spline_points(points: Sequence[Vec2], ds: float) -> list[Vec2]:
    pts = np.asarray([[p.x, p.y] for p in points], dtype=float)
    chord = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    if chord[-1] < _EPS:
        return list(points)
    # Natural cubic spline: C2 continuity and zero second derivative at the ends.
    sx = CubicSpline(chord, pts[:, 0], bc_type="natural")
    sy = CubicSpline(chord, pts[:, 1], bc_type="natural")
    n = max(2, int(math.ceil(chord[-1] / ds)) + 1)
    u = np.linspace(0.0, chord[-1], n)
    return [Vec2(float(x), float(y)) for x, y in zip(sx(u), sy(u))]


def _validate_smoothed(
    points: Sequence[Vec2],
    obstacles: Iterable[Obstacle],
    clearance: float,
    max_curvature: float,
) -> tuple[bool, str, float, float]:
    if any(not math.isfinite(p.x) or not math.isfinite(p.y) for p in points):
        return False, "non-finite spline point", math.inf, 0.0
    curv = np.abs(polyline_curvatures(points))
    max_k = float(curv.max()) if len(curv) else 0.0
    if max_k > max_curvature + 1e-9:
        return False, f"curvature {max_k:.4f} > {max_curvature:.4f}", max_k, 0.0
    min_c = polyline_clearance(points, obstacles)
    if min_c < clearance - 1e-9:
        return False, f"clearance {min_c:.4f} < {clearance:.4f}", max_k, min_c
    return True, "ok", max_k, min_c


def smooth_path(
    points: Sequence[Vec2],
    obstacles: Iterable[Obstacle],
    *,
    clearance: float,
    max_curvature: float,
    ds: float = 0.02,
    shortcut: bool = True,
) -> SmoothingResult:
    """Shortcut then fit a natural cubic spline, with fail-safe validation.

    The spline is accepted only if every sampled segment keeps the requested
    disc clearance and the Menger curvature stays below ``max_curvature``.
    Otherwise the original polyline is returned (``smoothed=False``).
    """

    base = _as_vec2(points)
    if len(base) < 3:
        return SmoothingResult(
            Path(base, smoothed=False, name="polyline"),
            False,
            "fewer than three points",
            0.0,
            polyline_clearance(base, obstacles),
            len(base),
        )

    candidates: list[tuple[str, list[Vec2]]] = []
    if shortcut:
        candidates.append(("shortcut", _greedy_shortcut(base, obstacles, clearance)))
    candidates.append(("original", base))

    last_reason = "no candidate"
    last_k = 0.0
    last_c = 0.0
    for label, pts in candidates:
        if len(pts) < 3:
            continue
        try:
            dense = _cubic_spline_points(pts, ds)
        except Exception as exc:  # numerical failure: fall back safely
            last_reason = f"{label} spline failed: {exc}"
            continue
        ok, reason, max_k, min_c = _validate_smoothed(
            dense, obstacles, clearance, max_curvature
        )
        last_reason, last_k, last_c = f"{label}: {reason}", max_k, min_c
        if ok:
            return SmoothingResult(
                Path(dense, smoothed=True, name=f"spline-{label}"),
                True,
                reason,
                max_k,
                min_c,
                len(pts),
            )

    return SmoothingResult(
        Path(base, smoothed=False, name="polyline"),
        False,
        last_reason,
        last_k,
        last_c,
        len(base),
    )
