"""Per-episode metrics.

Every metric is computed from the deterministic plant state and the actual
commands, not from the CoppeliaSim renderer.  Definitions are recorded in
``docs/MATHEMATICAL_FOUNDATIONS.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EpisodeMetrics:
    success: bool = False
    collision: bool = False
    timeout: bool = False
    planning_failed: bool = False
    out_of_bounds: bool = False
    start_in_collision: bool = False
    termination: str = "unknown"

    final_distance: float = math.inf
    executed_length: float = 0.0
    planned_length: float = 0.0
    time: float = 0.0
    steps: int = 0

    mean_speed: float = 0.0
    max_speed: float = 0.0
    control_energy: float = 0.0
    smoothness_cmd: float = 0.0
    smoothness_actuator: float = 0.0
    smoothness: float = 0.0
    tracking_rmse: float = 0.0
    min_clearance: float = math.inf

    # -- V4 dynamic-control metrics ----------------------------------------
    heading_rmse: float = 0.0
    max_cross_track: float = 0.0
    mean_abs_rudder: float = 0.0
    rudder_effort: float = 0.0
    yaw_rate_rms: float = 0.0
    peak_yaw_rate: float = 0.0
    mean_adaptation_effort: float = 0.0
    mean_latency: float = 0.0
    controller_time: float = 0.0
    plant_time: float = 0.0

    saturated_fraction: float = 0.0
    controller_saturated_fraction: float = 0.0
    vfh_blocked_fraction: float = 0.0

    planning_time: float = 0.0
    planner_nodes: int = 0
    planner_iterations: int = 0

    smoothing_success: bool = False
    smoothing_reason: str = ""
    max_path_curvature: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        # JSON cannot represent infinity portably; use null and document it.
        if not math.isfinite(data["min_clearance"]):
            data["min_clearance"] = None
        if not math.isfinite(data["final_distance"]):
            data["final_distance"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeMetrics":
        payload = dict(data)
        if payload.get("min_clearance") is None:
            payload["min_clearance"] = math.inf
        if payload.get("final_distance") is None:
            payload["final_distance"] = math.inf
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})
