"""Episode execution against an injected V5a1 hardware abstraction."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
from iara_v4.core.kinematics import DifferentialDrive
from iara_v4.core.path import Path, smooth_path
from iara_v4.core.types import Measurement, Vec2, wrap_angle
from iara_v4.dynamics.disturbance import DisturbanceModel
from iara_v4.dynamics.plant import KinematicPlantConfig
from iara_v4.simulation.metrics import EpisodeMetrics
from iara_v4.simulation.runner import (
    EpisodeConfig,
    EpisodeResult,
    GuidanceLayer,
    _make_controller,
    _make_planner,
)

from ..hal.base import HALError, HALConfig, HardwareHAL


def _config_dict(cfg: EpisodeConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["scenario"] = cfg.scenario.name
    data["spec"] = cfg.spec.to_dict() if cfg.spec is not None else None
    return data


def run_episode_v5a(
    cfg: EpisodeConfig,
    hal: HardwareHAL,
    hal_config: HALConfig | None = None,
) -> EpisodeResult:
    """Run one controller episode using only the injected HAL for plant state."""

    started = time.perf_counter()
    if cfg.spec is None:
        raise ValueError("EpisodeConfig must contain a condition specification")
    scenario = cfg.scenario
    clearance = (
        cfg.planning_clearance
        if cfg.planning_clearance is not None
        else KinematicPlantConfig().planning_radius
    )
    planner = _make_planner(
        cfg.planner, clearance, scenario.goal_tolerance, cfg.planner_options
    )
    planning = planner.plan(
        scenario.start.position,
        scenario.goal,
        scenario.obstacles,
        scenario.bounds,
        __import__("iara_v4.core.seeds", fromlist=["planner_rng"]).planner_rng(
            cfg.planner, scenario.name, cfg.seed
        ),
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
    if not planning.success or planning.path is None:
        metrics = EpisodeMetrics(
            planning_failed=not planning.success,
            success=planning.path is None and planning.success,
            termination=(
                f"planning_failed: {planning.reason}"
                if not planning.success
                else "start_at_goal"
            ),
            final_distance=scenario.start.position.distance_to(scenario.goal),
            planning_time=planning.planning_time,
            planner_nodes=planning.nodes,
            planner_iterations=planning.iterations,
        )
        return EpisodeResult(
            cfg.condition,
            scenario.name,
            cfg.seed,
            cfg.phase,
            "coppelia",
            cfg.spec.controller,
            metrics,
            [],
            [],
            [],
            [],
            planning_meta,
            {},
            _config_dict(cfg),
            time.perf_counter() - started,
        )

    base_path = planning.path
    if cfg.smoothing:
        smoothing = smooth_path(
            base_path.points,
            scenario.obstacles,
            clearance=clearance,
            max_curvature=cfg.max_path_curvature,
            ds=cfg.smooth_ds,
            shortcut=True,
        )
        path: Path = smoothing.path
    else:
        path = base_path
        smoothing = None
    if path.start.distance_to(scenario.start.position) > 1e-6 or path.end.distance_to(
        scenario.goal
    ) > 1e-6:
        path = base_path
        smoothing = None

    measurement = hal.reset(
        hal_config or HALConfig(dt=cfg.dt, timeout=scenario.timeout, seed=cfg.seed)
    )
    drive = DifferentialDrive(KinematicPlantConfig().limits)
    controller = _make_controller(
        cfg.spec,
        drive,
        scenario.goal_tolerance,
        cfg.controller_options,
        cfg.vehicle,
    )
    guidance_layer = GuidanceLayer(
        __import__(
            "iara_v4.control.pure_pursuit", fromlist=["PurePursuitConfig"]
        ).PurePursuitConfig(
            goal_tolerance=scenario.goal_tolerance,
            v_max=float(cfg.controller_options.get("v_max", 0.45)),
        ),
        drive,
    )
    controller.reset(
        measurement=measurement, reference=scenario.goal, seed=cfg.seed
    )
    disturbance = DisturbanceModel(
        cfg.disturbance, seed=cfg.seed, scenario=scenario.name
    )

    executed = [measurement.pose.position]
    errors: list[float] = []
    heading_errors: list[float] = []
    yaw_rates: list[float] = []
    command_rates: list[float] = []
    max_speed = abs(float(measurement.u))
    time_elapsed = 0.0
    collision = False
    success = False
    termination = "timeout"
    try:
        while time_elapsed < scenario.timeout - 1e-12:
            if measurement.pose.position.distance_to(scenario.goal) <= scenario.goal_tolerance:
                success = True
                termination = "success"
                break
            projection = path.project(measurement.pose.position)
            guidance = guidance_layer.compute(measurement.pose, path)
            command = controller.update(
                __import__(
                    "iara_v4.control.base", fromlist=["ControllerContext"]
                ).ControllerContext(
                    measurement=measurement,
                    path=path,
                    projection=projection,
                    guidance=guidance,
                    dt=cfg.dt,
                )
            )
            disturbance.update(cfg.dt)
            set_disturbance = getattr(hal, "set_disturbance", None)
            if set_disturbance is not None:
                body_force = disturbance.force(measurement.pose.theta)
                cpsi, spsi = math.cos(measurement.pose.theta), math.sin(
                    measurement.pose.theta
                )
                world_force = np.array(
                    [
                        cpsi * body_force[0] - spsi * body_force[1],
                        spsi * body_force[0] + cpsi * body_force[1],
                        0.0,
                    ]
                )
                world_torque = np.array([0.0, 0.0, body_force[2]])
                set_disturbance(world_force, world_torque)
            hal.apply_command(command)
            hal.step(cfg.dt)
            next_measurement = hal.read_measurement()
            errors.append(path.project(next_measurement.pose.position).distance)
            heading_errors.append(
                wrap_angle(
                    path.heading_at(path.project(next_measurement.pose.position).s)
                    - next_measurement.pose.theta
                )
            )
            yaw_rates.append(next_measurement.r)
            command_rates.append(command.omega)
            max_speed = max(max_speed, abs(float(next_measurement.u)))
            executed.append(next_measurement.pose.position)
            time_elapsed = max(time_elapsed + cfg.dt, next_measurement.time)
            measurement = next_measurement
    except Exception as exc:
        if isinstance(exc, HALError):
            raise
        raise HALError("V5a1 episode failed") from exc
    finally:
        hal.close()

    duration = max(time_elapsed, cfg.dt)
    executed_length = sum(
        executed[i].distance_to(executed[i - 1]) for i in range(1, len(executed))
    )
    metrics = EpisodeMetrics(
        success=success,
        collision=collision,
        timeout=not success and termination == "timeout",
        termination=termination,
        final_distance=measurement.pose.position.distance_to(scenario.goal),
        executed_length=executed_length,
        planned_length=path.length,
        time=time_elapsed,
        steps=max(0, len(executed) - 1),
        mean_speed=executed_length / duration,
        max_speed=max_speed,
        smoothness_cmd=(
            math.sqrt(
                sum(
                    ((command_rates[i] - command_rates[i - 1]) / cfg.dt) ** 2
                    for i in range(1, len(command_rates))
                )
                / (len(command_rates) - 1)
            )
            if len(command_rates) > 1
            else 0.0
        ),
        tracking_rmse=(
            math.sqrt(sum(x * x for x in errors) / len(errors))
            if errors
            else 0.0
        ),
        heading_rmse=(
            math.sqrt(sum(x * x for x in heading_errors) / len(heading_errors))
            if heading_errors
            else 0.0
        ),
        yaw_rate_rms=(
            math.sqrt(sum(x * x for x in yaw_rates) / len(yaw_rates))
            if yaw_rates
            else 0.0
        ),
        peak_yaw_rate=max((abs(x) for x in yaw_rates), default=0.0),
        planning_time=planning.planning_time,
        planner_nodes=planning.nodes,
        planner_iterations=planning.iterations,
        smoothing_success=smoothing is not None,
        smoothing_reason="enabled" if smoothing is not None else "disabled_or_fallback",
    )
    return EpisodeResult(
        cfg.condition,
        scenario.name,
        cfg.seed,
        cfg.phase,
        "coppelia",
        cfg.spec.controller,
        metrics,
        [[p.x, p.y] for p in path.points],
        [[p.x, p.y] for p in executed] if cfg.record_path else [],
        [],
        [],
        planning_meta,
        {},
        _config_dict(cfg),
        time.perf_counter() - started,
    )
