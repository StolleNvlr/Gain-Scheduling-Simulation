"""Single-episode execution on the V4 dynamic and kinematic plants.

The runner deliberately separates five layers that the V3 implementation fused:

1. **Global planning** -- identical path for every controller condition.
2. **Guidance** -- look-ahead point, speed profile and heading reference; this
   layer is controller independent, so the comparison does not reward a
   controller for using a different look-ahead rule.
3. **Controller** -- one of the V4 controllers, all exposing the same
   ``reset/update/telemetry`` protocol.
4. **Low-level actuation** -- first-order surge and rudder servos, deadband,
   rate limit and saturation (``dynamics.actuator``).
5. **Plant** -- either the minimal 3-DOF manoeuvring model or the preserved V3
   kinematic baseline.

Measurement noise, latency and dropouts are applied *before* the controller.
The plant never sees the noisy measurement; every metric is computed from the
true state.  This distinction is recorded in every result file.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from ..control.base import ControllerContext, Guidance
from ..control.gain_scheduler import GainScheduledController, GainScheduler
from ..control.heading import PathPDController, PDHeadingConfig, PDHeadingController, PurePursuitController
from ..control.mrac import MRACConfig, MRACController
from ..control.pd import PDConfig
from ..control.pure_pursuit import PurePursuit, PurePursuitConfig
from ..core.geometry import polyline_clearance
from ..core.kinematics import DifferentialDrive
from ..core.path import Path, SmoothingResult, smooth_path
from ..core.seeds import planner_rng, rng_for, stable_hash, validate_seed_set
from ..core.types import ControlCommand, Measurement, Pose2D, Vec2, wrap_angle
from ..dynamics.disturbance import DisturbanceConfig
from ..dynamics.model import VehicleParameters, mismatch_reference
from ..dynamics.plant import (
    DynamicPlantConfig,
    DynamicUSVPlant,
    KinematicPlantConfig,
    KinematicUSVPlant,
)
from ..planning.apf import APFConfig, ArtificialPotentialField
from ..planning.astar import AStar, AStarConfig
from ..planning.rrt_star import RRTStar, RRTStarConfig
from ..planning.vfh import VFHConfig, VFHPlus
from .metrics import EpisodeMetrics
from .sensors import RangeScanner
from .world import Scenario


def compute_smoothness(series: Sequence[float], dt: float) -> float:
    """Return the RMS time derivative of a sampled control signal."""

    if len(series) < 2 or dt <= 0.0:
        return 0.0
    diffs = [(series[i] - series[i - 1]) / dt for i in range(1, len(series))]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """Declarative definition of one experimental condition."""

    name: str
    controller: str
    plant: str = "dynamic"
    scheduler: str = "corrected"
    fixed_speed: float | None = None
    adapt: bool = True
    local_planner: str = "none"
    model_mismatch: bool = False
    initial_gain_scale: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "pp": ConditionSpec(
        "pp",
        "pure_pursuit",
        "dynamic",
        description="Pure Pursuit / kinematic-tracker baseline on the 3-DOF plant.",
    ),
    "pd": ConditionSpec(
        "pd",
        "path_pd",
        "dynamic",
        description="Cross-track/heading PD baseline on the 3-DOF plant.",
    ),
    "gs": ConditionSpec(
        "gs",
        "gain_scheduled",
        "dynamic",
        scheduler="corrected",
        description="Corrected continuous gain scheduler (constant damping ratio).",
    ),
    "gs_fixed": ConditionSpec(
        "gs_fixed",
        "gain_scheduled",
        "dynamic",
        scheduler="corrected",
        fixed_speed=0.55,
        description="Corrected scheduler frozen at the 0.55 m/s design point (ablation).",
    ),
    "mrac": ConditionSpec(
        "mrac",
        "mrac",
        "dynamic",
        adapt=True,
        description="Direct MRAC with projection, sigma-modification and anti-windup.",
    ),
    "mrac_fixed": ConditionSpec(
        "mrac_fixed",
        "mrac",
        "dynamic",
        adapt=False,
        description="MRAC with adaptation switched off (ablation).",
    ),
    "mrac_lowgain": ConditionSpec(
        "mrac_lowgain",
        "mrac",
        "dynamic",
        adapt=True,
        initial_gain_scale=0.30,
        description="Adaptive MRAC starting at 30% of the matching gains.",
    ),
    "mrac_lowgain_fixed": ConditionSpec(
        "mrac_lowgain_fixed",
        "mrac",
        "dynamic",
        adapt=False,
        initial_gain_scale=0.30,
        description="Fixed-gain MRAC starting at 30% of the matching gains.",
    ),
    "pp_kinematic": ConditionSpec(
        "pp_kinematic",
        "pure_pursuit",
        "kinematic",
        description="V3 kinematic baseline; plant-isolation control condition.",
    ),
    "gs_mismatch": ConditionSpec(
        "gs_mismatch",
        "gain_scheduled",
        "dynamic",
        scheduler="corrected",
        model_mismatch=True,
        description="Corrected scheduler using a biased internal yaw model (mismatch ablation).",
    ),
    "mrac_mismatch": ConditionSpec(
        "mrac_mismatch",
        "mrac",
        "dynamic",
        adapt=True,
        model_mismatch=True,
        description="Adaptive MRAC with a biased internal yaw model.",
    ),
    "mrac_fixed_mismatch": ConditionSpec(
        "mrac_fixed_mismatch",
        "mrac",
        "dynamic",
        adapt=False,
        model_mismatch=True,
        description="MRAC adaptation disabled under the same biased internal model.",
    ),
    "pp_vfh": ConditionSpec(
        "pp_vfh",
        "pure_pursuit",
        "dynamic",
        local_planner="vfh",
        description="Pure Pursuit plus VFH+ on the 3-DOF plant.",
    ),
    "gs_vfh": ConditionSpec(
        "gs_vfh",
        "gain_scheduled",
        "dynamic",
        scheduler="corrected",
        local_planner="vfh",
        description="Corrected gain scheduler plus VFH+ on the 3-DOF plant.",
    ),
}


def condition_spec(name: str) -> ConditionSpec:
    try:
        return CONDITION_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown condition {name!r}; choose from {sorted(CONDITION_SPECS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Measurement error model applied before the controller.

    The random stream excludes the condition name, so the same
    ``(scenario, seed)`` receives the same noise *sequence* in every condition.
    Latency is expressed in control steps, which makes the timing independent of
    the integration sub-step used by the plant.
    """

    position_std: float = 0.004
    heading_std: float = 0.004
    surge_std: float = 0.008
    sway_std: float = 0.004
    yaw_rate_std: float = 0.004
    latency_steps: int = 0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "position_std",
            "heading_std",
            "surge_std",
            "sway_std",
            "yaw_rate_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.latency_steps < 0:
            raise ValueError("latency_steps must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")


class MeasurementModel:
    """Noisy, delayed, optionally dropped measurement stream."""

    def __init__(self, config: SensorConfig, *, seed: int, scenario: str) -> None:
        self.config = config
        self._rng = rng_for(seed, stable_hash(f"sensor:{scenario}"))
        self._queue: deque[Measurement] = deque(maxlen=config.latency_steps + 1)
        self._last: Measurement | None = None

    def sample(self, pose: Pose2D, u: float, v: float, r: float, time_s: float) -> Measurement:
        c = self.config
        # Draw every field unconditionally so that changing one field does not
        # shift the random sequence seen by the others.
        dx = float(self._rng.normal(0.0, c.position_std))
        dy = float(self._rng.normal(0.0, c.position_std))
        dpsi = float(self._rng.normal(0.0, c.heading_std))
        du = float(self._rng.normal(0.0, c.surge_std))
        dv = float(self._rng.normal(0.0, c.sway_std))
        dr = float(self._rng.normal(0.0, c.yaw_rate_std))
        dropped = bool(self._rng.random() < c.dropout_probability)

        if dropped and self._last is not None:
            held = self._last
            measurement = Measurement(
                time=float(time_s),
                source_time=held.source_time,
                pose=held.pose,
                u=held.u,
                v=held.v,
                r=held.r,
                dropped=True,
            )
        else:
            measurement = Measurement(
                time=float(time_s),
                source_time=float(time_s),
                pose=Pose2D(pose.x + dx, pose.y + dy, pose.theta + dpsi),
                u=float(u + du),
                v=float(v + dv),
                r=float(r + dr),
                dropped=dropped,
            )
        self._last = measurement
        self._queue.append(measurement)
        # deque(maxlen=L+1): after appending, length > L means the oldest sample
        # has waited exactly L control steps.  For L=0 it returns the new sample.
        if len(self._queue) > c.latency_steps:
            return self._queue.popleft()
        return self._queue[0]


# ---------------------------------------------------------------------------
# Episode configuration and result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EpisodeConfig:
    scenario: Scenario
    condition: str
    seed: int
    phase: str = "test"
    spec: ConditionSpec | None = None
    dt: float = 0.05
    substeps: int = 4
    planner: str = "astar"
    planning_clearance: float | None = None
    smoothing: bool = True
    smooth_ds: float = 0.02
    max_path_curvature: float = 4.0
    record_trajectory: bool = False
    record_path: bool = True
    record_telemetry: bool = False
    planner_options: dict[str, Any] = field(default_factory=dict)
    controller_options: dict[str, Any] = field(default_factory=dict)
    local_planner_options: dict[str, Any] = field(default_factory=dict)
    vfh_config: VFHConfig | None = None
    sensor: SensorConfig = field(default_factory=SensorConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
    # The controller may use a different internal model from the plant.  This
    # is how model mismatch is created explicitly instead of being hidden in a
    # controller-specific option.  ``None`` means "same as the plant".
    controller_vehicle: VehicleParameters | None = None

    def __post_init__(self) -> None:
        validate_seed_set([self.seed], self.phase)
        if self.spec is None:
            self.spec = condition_spec(self.condition)
        if self.spec.name != self.condition:
            raise ValueError(
                f"condition {self.condition!r} does not match spec {self.spec.name!r}"
            )
        if self.planner not in {"rrt_star", "astar", "apf"}:
            raise ValueError(f"unknown planner {self.planner!r}")
        if self.spec.plant not in {"dynamic", "kinematic"}:
            raise ValueError(f"unknown plant {self.spec.plant!r}")
        if self.spec.local_planner not in {"none", "vfh"}:
            raise ValueError(f"unknown local planner {self.spec.local_planner!r}")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")
        if self.planning_clearance is not None and self.planning_clearance < 0.0:
            raise ValueError("planning_clearance must be non-negative")
        if self.max_path_curvature <= 0.0:
            raise ValueError("max_path_curvature must be positive")
        if self.spec.local_planner == "vfh" and self.spec.controller not in {
            "pure_pursuit",
            "path_pd",
            "gain_scheduled",
            "mrac",
        }:
            raise ValueError("VFH+ is not wired to this controller")


@dataclass(slots=True)
class EpisodeResult:
    condition: str
    scenario: str
    seed: int
    phase: str
    plant: str
    controller: str
    metrics: EpisodeMetrics
    planned_path: list[list[float]]
    executed_path: list[list[float]]
    trajectory: list[list[float]]
    telemetry: list[dict[str, Any]]
    planning: dict[str, Any]
    smoothing: dict[str, Any]
    config: dict[str, Any]
    wall_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "scenario": self.scenario,
            "seed": self.seed,
            "phase": self.phase,
            "plant": self.plant,
            "controller": self.controller,
            "metrics": self.metrics.to_dict(),
            "planned_path": self.planned_path,
            "executed_path": self.executed_path,
            "trajectory": self.trajectory,
            "telemetry": self.telemetry,
            "planning": self.planning,
            "smoothing": self.smoothing,
            "config": self.config,
            "wall_time": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Planning / controller factories
# ---------------------------------------------------------------------------


def _make_planner(
    name: str,
    clearance: float,
    goal_tolerance: float,
    options: dict[str, Any],
):
    if name == "rrt_star":
        base = {"clearance": clearance, "goal_tolerance": goal_tolerance}
        base.update(options)
        return RRTStar(RRTStarConfig(**base))
    if name == "astar":
        base = {"clearance": clearance}
        base.update(options)
        return AStar(AStarConfig(**base))
    if name == "apf":
        base = {"clearance": clearance, "goal_tolerance": goal_tolerance}
        base.update(options)
        return ArtificialPotentialField(APFConfig(**base))
    raise ValueError(f"unknown planner {name!r}")


def _make_controller(
    spec: ConditionSpec,
    drive: DifferentialDrive,
    goal_tolerance: float,
    options: dict[str, Any],
    vehicle: VehicleParameters,
):
    opts = dict(options)
    v_max = float(opts.pop("v_max", min(0.45, drive.limits.max_linear_speed)))

    if spec.controller == "pure_pursuit":
        cfg = PurePursuitConfig(
            goal_tolerance=goal_tolerance,
            v_max=v_max,
            **opts,
        )
        return PurePursuitController(cfg, drive)

    if spec.controller == "path_pd":
        cfg = PDConfig(goal_tolerance=goal_tolerance, v_max=v_max, **opts)
        return PathPDController(cfg, drive)

    if spec.controller == "pd_heading":
        cfg = PDHeadingConfig(**opts)
        return PDHeadingController(cfg, drive)

    if spec.controller == "gain_scheduled":
        omega_n = float(opts.pop("omega_n", 2.0))
        zeta = float(opts.pop("zeta", 1.0))
        speeds = tuple(opts.pop("speeds", (0.1, 0.8, 1.0)))
        derivative_filter_tau = float(opts.pop("derivative_filter_tau", 0.10))
        gamma_max = float(opts.pop("gamma_max", 2.0))
        if opts:
            raise ValueError(f"unused gain-scheduled options: {sorted(opts)}")
        scheduler = GainScheduler.corrected(
            vehicle,
            omega_n=omega_n,
            zeta=zeta,
            speeds=speeds,
            gamma_max=gamma_max,
        )
        fixed_gains = None
        if spec.fixed_speed is not None:
            fixed = scheduler.gains(spec.fixed_speed)
            fixed_gains = (fixed.kp, fixed.kd)
        return GainScheduledController(
            scheduler,
            drive=drive,
            vehicle=vehicle,
            delta_max=vehicle.rudder_max,
            derivative_filter_tau=derivative_filter_tau,
            fixed_gains=fixed_gains,
        )

    if spec.controller == "mrac":
        if not spec.adapt:
            opts["adapt"] = False
        opts.setdefault("initial_gain_scale", spec.initial_gain_scale)
        cfg = MRACConfig(**opts)
        return MRACController(cfg, drive=drive, vehicle=vehicle)

    raise ValueError(f"unknown controller {spec.controller!r}")


def _config_dict(cfg: EpisodeConfig) -> dict[str, Any]:
    assert cfg.spec is not None
    return {
        "condition": cfg.condition,
        "condition_spec": cfg.spec.to_dict(),
        "plant": cfg.spec.plant,
        "controller": cfg.spec.controller,
        "planner": cfg.planner,
        "dt": cfg.dt,
        "substeps": cfg.substeps,
        "planning_clearance": cfg.planning_clearance,
        "smoothing": cfg.smoothing,
        "smooth_ds": cfg.smooth_ds,
        "max_path_curvature": cfg.max_path_curvature,
        "planner_options": dict(cfg.planner_options),
        "controller_options": dict(cfg.controller_options),
        "local_planner_options": dict(cfg.local_planner_options),
        "sensor": asdict(cfg.sensor),
        "disturbance": asdict(cfg.disturbance),
        "vehicle": asdict(cfg.vehicle),
        "controller_vehicle": asdict(
            cfg.controller_vehicle
            if cfg.controller_vehicle is not None
            else (
                mismatch_reference(cfg.vehicle)
                if cfg.spec.model_mismatch
                else cfg.vehicle
            )
        ),
        "model_mismatch": (
            cfg.controller_vehicle is not None
            and cfg.controller_vehicle != cfg.vehicle
        )
        or cfg.spec.model_mismatch,
        "scenario_timeout": cfg.scenario.timeout,
        "scenario_goal_tolerance": cfg.scenario.goal_tolerance,
    }


def _empty_result(
    cfg: EpisodeConfig,
    metrics: EpisodeMetrics,
    *,
    planning: dict[str, Any] | None = None,
    smoothing: dict[str, Any] | None = None,
    wall_time: float = 0.0,
) -> EpisodeResult:
    assert cfg.spec is not None
    return EpisodeResult(
        condition=cfg.condition,
        scenario=cfg.scenario.name,
        seed=cfg.seed,
        phase=cfg.phase,
        plant=cfg.spec.plant,
        controller=cfg.spec.controller,
        metrics=metrics,
        planned_path=[],
        executed_path=[],
        trajectory=[],
        telemetry=[],
        planning=planning or {},
        smoothing=smoothing or {},
        config=_config_dict(cfg),
        wall_time=wall_time,
    )


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------


class GuidanceLayer:
    """Controller-independent look-ahead, speed and heading reference."""

    def __init__(self, config: PurePursuitConfig, drive: DifferentialDrive) -> None:
        self._inner = PurePursuit(config, drive)

    def compute(self, pose: Pose2D, path: Path) -> Guidance:
        projection = path.project(pose.position)
        remaining = max(0.0, path.length - projection.s)
        lookahead = self._inner.lookahead_distance(remaining)
        s_target = min(path.length, projection.s + lookahead)
        target = path.point_at(s_target)
        speed, _ = self._inner.speed_profile(pose, path, projection)
        path_heading = path.heading_at(projection.s)
        tangent = path.tangent_at(projection.s)
        normal = Vec2(-tangent.y, tangent.x)
        cross_track = (pose.position - projection.point).dot(normal)
        heading_ref = math.atan2(target.y - pose.y, target.x - pose.x)
        return Guidance(
            speed=float(speed),
            heading_ref=float(heading_ref),
            target=target,
            lookahead=float(lookahead),
            path_heading=float(path_heading),
            cross_track=float(cross_track),
        )


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------


def run_episode(cfg: EpisodeConfig) -> EpisodeResult:
    """Run one episode and return metrics, plan and optional telemetry."""

    wall_start = time.perf_counter()
    assert cfg.spec is not None
    scenario = cfg.scenario

    plant_cfg = DynamicPlantConfig(
        vehicle=cfg.vehicle,
        disturbance=cfg.disturbance,
        dt=cfg.dt,
        substeps=cfg.substeps,
    )
    clearance = (
        plant_cfg.planning_radius
        if cfg.planning_clearance is None
        else float(cfg.planning_clearance)
    )
    goal_tolerance = scenario.goal_tolerance

    planner = _make_planner(
        cfg.planner, clearance, goal_tolerance, cfg.planner_options
    )
    rng = planner_rng(cfg.planner, scenario.name, cfg.seed)
    planning = planner.plan(
        scenario.start.position,
        scenario.goal,
        scenario.obstacles,
        scenario.bounds,
        rng,
    )
    planning_meta = {
        "success": planning.success,
        "cost": planning.cost if math.isfinite(planning.cost) else None,
        "iterations": planning.iterations,
        "planning_time": planning.planning_time,
        "nodes": planning.nodes,
        "reason": planning.reason,
        **planning.metadata,
    }

    if not planning.success:
        metrics = EpisodeMetrics(
            planning_failed=True,
            termination=f"planning_failed: {planning.reason}",
            planning_time=planning.planning_time,
            planner_nodes=planning.nodes,
            planner_iterations=planning.iterations,
        )
        return _empty_result(
            cfg,
            metrics,
            planning=planning_meta,
            wall_time=time.perf_counter() - wall_start,
        )

    if planning.path is None:
        metrics = EpisodeMetrics(
            success=True,
            termination="start_at_goal",
            final_distance=scenario.start.position.distance_to(scenario.goal),
            min_clearance=polyline_clearance(
                [scenario.start.position], scenario.all_obstacles
            ),
            planning_time=planning.planning_time,
            planner_nodes=planning.nodes,
            planner_iterations=planning.iterations,
        )
        return _empty_result(
            cfg,
            metrics,
            planning=planning_meta,
            wall_time=time.perf_counter() - wall_start,
        )

    base_path: Path = planning.path
    if cfg.smoothing:
        smoothing = smooth_path(
            base_path.points,
            scenario.obstacles,
            clearance=clearance,
            max_curvature=cfg.max_path_curvature,
            ds=cfg.smooth_ds,
            shortcut=True,
        )
        path = smoothing.path
    else:
        smoothing = SmoothingResult(
            path=base_path,
            smoothed=False,
            reason="disabled",
            max_curvature=0.0,
            min_clearance=polyline_clearance(base_path.points, scenario.obstacles),
            shortcut_points=len(base_path),
        )
        path = base_path

    if (
        path.start.distance_to(scenario.start.position) > 1e-6
        or path.end.distance_to(scenario.goal) > 1e-6
    ):
        path = base_path
        smoothing = SmoothingResult(
            path=base_path,
            smoothed=False,
            reason="smoothed path endpoints changed; fell back to polyline",
            max_curvature=0.0,
            min_clearance=polyline_clearance(base_path.points, scenario.obstacles),
            shortcut_points=len(base_path),
        )

    smoothing_meta = {
        "success": smoothing.smoothed,
        "reason": smoothing.reason,
        "max_curvature": smoothing.max_curvature,
        "min_clearance": smoothing.min_clearance,
        "shortcut_points": smoothing.shortcut_points,
        "path_length": path.length,
    }

    drive = DifferentialDrive(KinematicPlantConfig().limits)
    if cfg.spec.plant == "kinematic":
        plant = KinematicUSVPlant(
            KinematicPlantConfig(dt=cfg.dt),
            scenario.start,
        )
    else:
        plant = DynamicUSVPlant(
            plant_cfg,
            scenario.start,
            seed=cfg.seed,
            scenario=scenario.name,
        )

    if cfg.controller_vehicle is not None:
        controller_vehicle = cfg.controller_vehicle
    elif cfg.spec.model_mismatch:
        controller_vehicle = mismatch_reference(cfg.vehicle)
    else:
        controller_vehicle = cfg.vehicle

    controller = _make_controller(
        cfg.spec,
        drive,
        goal_tolerance,
        cfg.controller_options,
        controller_vehicle,
    )
    guidance_layer = GuidanceLayer(
        PurePursuitConfig(
            goal_tolerance=goal_tolerance,
            v_max=float(cfg.controller_options.get("v_max", 0.45)),
        ),
        drive,
    )

    vfh: VFHPlus | None = None
    scanner: RangeScanner | None = None
    if cfg.spec.local_planner == "vfh":
        if cfg.vfh_config is not None:
            vfh_cfg = cfg.vfh_config
        else:
            options = dict(cfg.local_planner_options)
            options.setdefault("n_sectors", 72)
            options.setdefault("max_range", 3.0)
            options.setdefault("robot_radius", clearance)
            options.setdefault("safety_margin", 0.0)
            vfh_cfg = VFHConfig(**options)
        vfh = VFHPlus(vfh_cfg)
        scanner = RangeScanner(vfh_cfg.n_sectors, vfh_cfg.max_range)

    sensor = MeasurementModel(cfg.sensor, seed=cfg.seed, scenario=scenario.name)
    measurement = sensor.sample(
        plant.pose, plant.state.u, plant.state.v, plant.state.r, 0.0
    )
    controller.reset(measurement=measurement, reference=scenario.goal, seed=cfg.seed)

    executed: list[Vec2] = [plant.pose.position]
    trajectory: list[list[float]] = []
    telemetry: list[dict[str, Any]] = []

    errors: list[float] = []
    heading_errors: list[float] = []
    speed_samples: list[float] = []
    yaw_rates: list[float] = []
    latencies: list[float] = []

    min_clearance = plant.clearance(scenario.all_obstacles)
    executed_length = 0.0
    control_energy = 0.0
    omega_cmd_series: list[float] = []
    delta_applied_series: list[float] = []
    rudder_effort = 0.0
    rudder_abs = 0.0
    max_speed = 0.0
    max_cross_track = 0.0
    saturated_steps = 0
    controller_saturated_steps = 0
    vfh_blocked_steps = 0
    adaptation_effort = 0.0
    controller_time = 0.0
    plant_time = 0.0
    steps = 0
    previous_delta: float | None = None
    last_vfh_heading: float | None = None

    success = False
    collision = False
    timeout = False
    out_of_bounds = False
    termination = "timeout"

    if plant.collision(scenario.all_obstacles):
        metrics = EpisodeMetrics(
            collision=True,
            start_in_collision=True,
            termination="start_in_collision",
            final_distance=plant.pose.position.distance_to(scenario.goal),
            min_clearance=min_clearance,
            planned_length=path.length,
            planning_time=planning.planning_time,
            planner_nodes=planning.nodes,
            planner_iterations=planning.iterations,
            smoothing_success=smoothing.smoothed,
            smoothing_reason=smoothing.reason,
            max_path_curvature=smoothing.max_curvature,
        )
        return _empty_result(
            cfg,
            metrics,
            planning=planning_meta,
            smoothing=smoothing_meta,
            wall_time=time.perf_counter() - wall_start,
        )

    while plant.time < scenario.timeout - 1e-12:
        pose = plant.pose
        if pose.position.distance_to(scenario.goal) <= goal_tolerance:
            success = True
            termination = "success"
            break

        projection = path.project(measurement.pose.position)
        guidance = guidance_layer.compute(measurement.pose, path)
        latencies.append(measurement.latency)
        dropped = measurement.dropped

        if vfh is not None and scanner is not None:
            scan = scanner.scan(measurement.pose, scenario.all_obstacles)
            vfh_result = vfh.guidance(
                guidance.heading_ref, scan, last_vfh_heading
            )
            last_vfh_heading = vfh_result.heading
            if vfh_result.blocked:
                command = drive.command(
                    0.0, 0.0, target=measurement.pose.position
                )
                vfh_blocked_steps += 1
            else:
                ld = max(guidance.lookahead, 0.05)
                target = measurement.pose.position + Vec2(
                    math.cos(vfh_result.heading),
                    math.sin(vfh_result.heading),
                ) * ld
                guidance = replace(
                    guidance,
                    speed=guidance.speed * vfh_result.speed_factor,
                    heading_ref=vfh_result.heading,
                    target=target,
                    speed_factor=vfh_result.speed_factor,
                )
                context = ControllerContext(
                    measurement=measurement,
                    path=path,
                    projection=projection,
                    guidance=guidance,
                    dt=cfg.dt,
                )
                t0 = time.perf_counter()
                command = controller.update(context)
                controller_time += time.perf_counter() - t0
        else:
            context = ControllerContext(
                measurement=measurement,
                path=path,
                projection=projection,
                guidance=guidance,
                dt=cfg.dt,
            )
            t0 = time.perf_counter()
            command = controller.update(context)
            controller_time += time.perf_counter() - t0

        previous_position = pose.position
        t1 = time.perf_counter()
        step = plant.step(command, cfg.dt)
        plant_time += time.perf_counter() - t1
        steps += 1
        pose = step.state.pose

        executed_length += previous_position.distance_to(pose.position)
        executed.append(pose.position)
        clearance_now = plant.clearance(scenario.all_obstacles)
        min_clearance = min(min_clearance, clearance_now)
        max_speed = max(max_speed, abs(step.state.u))
        speed_samples.append(abs(step.state.u))
        yaw_rates.append(step.state.r)

        true_projection = path.project(pose.position)
        errors.append(true_projection.distance)
        max_cross_track = max(max_cross_track, true_projection.distance)
        heading_errors.append(
            wrap_angle(path.heading_at(true_projection.s) - pose.theta)
        )

        delta = float(step.delta_applied)
        rudder_effort += delta * delta * cfg.dt
        rudder_abs += abs(delta) * cfg.dt
        previous_delta = delta
        omega_cmd_series.append(float(command.omega))
        delta_applied_series.append(delta)

        if cfg.spec.plant == "kinematic":
            if command.wheels is not None:
                control_energy += (
                    command.wheels.left * command.wheels.left
                    + command.wheels.right * command.wheels.right
                ) * cfg.dt
            else:
                control_energy += (
                    command.v * command.v + command.omega * command.omega
                ) * cfg.dt
        else:
            control_energy += (
                step.state.tau_u * step.state.tau_u
                + step.state.tau_r * step.state.tau_r
            ) * cfg.dt

        saturated_steps += int(step.saturated)
        controller_saturated_steps += int(command.saturated)
        controller_tel = controller.telemetry()
        adaptation_effort += float(controller_tel.get("adaptation_effort", 0.0))

        if cfg.record_trajectory:
            trajectory.append(
                [
                    float(step.state.time),
                    float(pose.x),
                    float(pose.y),
                    float(pose.theta),
                    float(step.state.u),
                    float(step.state.v),
                    float(step.state.r),
                    float(command.v),
                    float(command.omega),
                    float(command.delta_rudder or 0.0),
                    float(step.delta_applied),
                    float(step.state.tau_u),
                    float(step.state.tau_r),
                    float(true_projection.s),
                    float(true_projection.distance),
                    float(wrap_angle(guidance.heading_ref - pose.theta)),
                    float(measurement.latency),
                ]
            )

        if cfg.record_telemetry:
            telemetry.append(
                {
                    "time": float(step.state.time),
                    "x": float(pose.x),
                    "y": float(pose.y),
                    "psi": float(pose.theta),
                    "u": float(step.state.u),
                    "v": float(step.state.v),
                    "r": float(step.state.r),
                    "x_meas": float(measurement.pose.x),
                    "y_meas": float(measurement.pose.y),
                    "psi_meas": float(measurement.pose.theta),
                    "u_meas": float(measurement.u),
                    "v_meas": float(measurement.v),
                    "r_meas": float(measurement.r),
                    "latency": float(measurement.latency),
                    "dropped": bool(dropped),
                    "s_ref": float(true_projection.s),
                    "cross_track": float(true_projection.distance),
                    "heading_ref": float(guidance.heading_ref),
                    "speed_ref": float(guidance.speed),
                    "delta_cmd": None
                    if command.delta_rudder is None
                    else float(command.delta_rudder),
                    "delta_applied": float(step.delta_applied),
                    "tau_u_cmd": float(step.tau_u_cmd),
                    "tau_u_applied": float(step.state.tau_u),
                    "tau_r_cmd": float(step.tau_r_cmd),
                    "tau_r_applied": float(step.state.tau_r),
                    "saturated": bool(step.saturated),
                    "deadband": bool(step.deadband_active),
                    "disturbance_u": float(step.state.disturbance_u),
                    "disturbance_v": float(step.state.disturbance_v),
                    "disturbance_r": float(step.state.disturbance_r),
                    "controller": controller_tel,
                }
            )

        if plant.collision(scenario.all_obstacles):
            collision = True
            termination = "collision"
            break

        min_x, min_y, max_x, max_y = scenario.bounds
        if not (min_x <= pose.x <= max_x and min_y <= pose.y <= max_y):
            out_of_bounds = True
            termination = "out_of_bounds"
            break

        if pose.position.distance_to(scenario.goal) <= goal_tolerance:
            success = True
            termination = "success"
            break

        # Next measurement is sampled after the plant has advanced.  It is
        # timestamped at the new plant time; the latency queue delays delivery.
        measurement = sensor.sample(
            pose, step.state.u, step.state.v, step.state.r, plant.time
        )
    else:
        timeout = True
        termination = "timeout"

    success = success and not collision and not out_of_bounds
    final_distance = plant.pose.position.distance_to(scenario.goal)
    duration = max(plant.time, 1e-12)
    mean_speed = executed_length / duration
    tracking_rmse = (
        math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
    )
    heading_rmse = (
        math.sqrt(sum(e * e for e in heading_errors) / len(heading_errors))
        if heading_errors
        else 0.0
    )
    yaw_rate_rms = (
        math.sqrt(sum(r * r for r in yaw_rates) / len(yaw_rates))
        if yaw_rates
        else 0.0
    )
    peak_yaw_rate = max((abs(r) for r in yaw_rates), default=0.0)
    mean_latency = (
        sum(latencies) / len(latencies) if latencies else 0.0
    )
    mean_adaptation_effort = (
        adaptation_effort / steps if steps else 0.0
    )

    max_path_curvature = 0.0
    if path.length > 0.0:
        s_values = np.linspace(
            0.0, path.length, min(200, max(10, len(path) * 4))
        )
        max_path_curvature = max(
            abs(path.curvature_at(float(s))) for s in s_values
        )

    smoothness_cmd = compute_smoothness(omega_cmd_series, cfg.dt)
    smoothness_actuator = compute_smoothness(delta_applied_series, cfg.dt)
    metrics = EpisodeMetrics(
        success=success,
        collision=collision,
        timeout=timeout,
        planning_failed=False,
        out_of_bounds=out_of_bounds,
        start_in_collision=False,
        termination=termination,
        final_distance=final_distance,
        executed_length=executed_length,
        planned_length=path.length,
        time=plant.time,
        steps=steps,
        mean_speed=mean_speed,
        max_speed=max_speed,
        control_energy=control_energy,
        smoothness_cmd=smoothness_cmd,
        smoothness_actuator=smoothness_actuator,
        smoothness=smoothness_actuator,
        tracking_rmse=tracking_rmse,
        min_clearance=min_clearance,
        heading_rmse=heading_rmse,
        max_cross_track=max_cross_track,
        mean_abs_rudder=(rudder_abs / duration),
        rudder_effort=rudder_effort,
        yaw_rate_rms=yaw_rate_rms,
        peak_yaw_rate=peak_yaw_rate,
        mean_adaptation_effort=mean_adaptation_effort,
        mean_latency=mean_latency,
        controller_time=controller_time,
        plant_time=plant_time,
        saturated_fraction=(saturated_steps / steps) if steps else 0.0,
        controller_saturated_fraction=(
            controller_saturated_steps / steps if steps else 0.0
        ),
        vfh_blocked_fraction=(vfh_blocked_steps / steps) if steps else 0.0,
        planning_time=planning.planning_time,
        planner_nodes=planning.nodes,
        planner_iterations=planning.iterations,
        smoothing_success=smoothing.smoothed,
        smoothing_reason=smoothing.reason,
        max_path_curvature=max_path_curvature,
    )

    return EpisodeResult(
        condition=cfg.condition,
        scenario=scenario.name,
        seed=cfg.seed,
        phase=cfg.phase,
        plant=cfg.spec.plant,
        controller=cfg.spec.controller,
        metrics=metrics,
        planned_path=[[p.x, p.y] for p in path.points],
        executed_path=(
            [[p.x, p.y] for p in executed] if cfg.record_path else []
        ),
        trajectory=trajectory,
        telemetry=telemetry,
        planning=planning_meta,
        smoothing=smoothing_meta,
        config=_config_dict(cfg),
        wall_time=time.perf_counter() - wall_start,
    )
