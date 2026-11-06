"""Deterministic validation experiments that are not scenario benchmarks.

The benchmark answers "does the closed loop reach the goal?".  These tests
answer narrower questions that a success rate cannot answer:

* does the MRAC step response improve when adaptation starts from a wrong gain?
* do the estimated gains move in the direction of the matching gains?
* is the common-Lyapunov certificate positive definite with a negative margin?
* are the actuator limits and the plant invariants respected?

The step-response harness uses the same ``MRACController`` and ``DynamicUSVPlant``
as the benchmark, but replaces the guidance layer with a constant heading
reference.  That isolates the adaptive law from path geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..control.base import ControllerContext, Guidance
from ..control.mrac import MRACConfig, MRACController
from ..core.kinematics import DifferentialDrive
from ..core.path import Path
from ..core.types import Measurement, Pose2D, Vec2
from ..dynamics.disturbance import DisturbanceConfig
from ..dynamics.model import VehicleParameters
from ..dynamics.plant import DynamicPlantConfig, DynamicUSVPlant


@dataclass(slots=True)
class StepResponseResult:
    adapt: bool
    initial_gain_scale: float
    time: np.ndarray
    heading: np.ndarray
    heading_reference: np.ndarray
    reference_model: np.ndarray
    yaw_rate: np.ndarray
    rudder: np.ndarray
    kr: np.ndarray
    kx1: np.ndarray
    kx2: np.ndarray

    def summary(self, target: float, matching_kr: float) -> dict[str, Any]:
        error = self.heading - self.reference_model
        heading_error = self.heading - target
        return {
            "adapt": bool(self.adapt),
            "initial_gain_scale": float(self.initial_gain_scale),
            "reference_model_rmse": float(np.sqrt(np.mean(error * error))),
            "heading_rmse": float(np.sqrt(np.mean(heading_error * heading_error))),
            "final_heading_error": float(self.heading[-1] - target),
            "max_abs_rudder": float(np.max(np.abs(self.rudder))),
            "mean_abs_rudder": float(np.mean(np.abs(self.rudder))),
            "initial_kr": float(self.kr[0]),
            "final_kr": float(self.kr[-1]),
            "matching_kr": float(matching_kr),
            "kr_fraction_of_matching": float(self.kr[-1] / matching_kr),
            "kr_moved_toward_matching": bool(
                abs(self.kr[-1] - matching_kr) < abs(self.kr[0] - matching_kr)
            ),
            "saturated_fraction": float(
                np.mean(np.abs(self.rudder) >= 0.65 - 1e-12)
            ),
            "all_finite": bool(
                np.all(np.isfinite(self.heading))
                and np.all(np.isfinite(self.kr))
                and np.all(np.isfinite(self.rudder))
            ),
        }


def run_step_response(
    *,
    adapt: bool,
    initial_gain_scale: float = 0.10,
    omega_n: float = 5.5,
    zeta: float = 1.0,
    gamma_psi: float = 600.0,
    gamma_r: float = 300.0,
    gamma_ref: float = 600.0,
    sigma: float = 1e-5,
    speed: float = 0.35,
    heading_reference: float = 0.15,
    duration: float = 20.0,
    dt: float = 0.02,
    vehicle: VehicleParameters | None = None,
) -> StepResponseResult:
    p = vehicle or VehicleParameters()
    config = MRACConfig(
        omega_n=omega_n,
        zeta=zeta,
        gamma_psi=gamma_psi,
        gamma_r=gamma_r,
        gamma_ref=gamma_ref,
        sigma=sigma,
        initial_gain_scale=initial_gain_scale,
        adapt=adapt,
        freeze_on_saturation=False,
    )
    drive = DifferentialDrive()
    controller = MRACController(config, drive=drive, vehicle=p)
    plant = DynamicUSVPlant(
        DynamicPlantConfig(
            vehicle=p,
            disturbance=DisturbanceConfig(enabled=False),
            dt=dt,
            substeps=4,
        ),
        Pose2D(0.0, 0.0, 0.0),
    )
    path = Path([Vec2(0.0, 0.0), Vec2(3.0, 0.0)], name="straight")
    measurement = Measurement(0.0, 0.0, Pose2D(0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
    controller.reset(
        measurement=measurement,
        reference=Vec2(3.0, 0.0),
        seed=0,
    )
    guidance = Guidance(
        speed=speed,
        heading_ref=heading_reference,
        target=Vec2(3.0, 0.0),
        lookahead=0.5,
        path_heading=0.0,
        cross_track=0.0,
    )

    times: list[float] = []
    heading: list[float] = []
    heading_ref: list[float] = []
    reference_model: list[float] = []
    yaw_rate: list[float] = []
    rudder: list[float] = []
    kr: list[float] = []
    kx1: list[float] = []
    kx2: list[float] = []

    for _ in range(int(round(duration / dt))):
        projection = path.project(measurement.pose.position)
        context = ControllerContext(
            measurement=measurement,
            path=path,
            projection=projection,
            guidance=guidance,
            dt=dt,
        )
        command = controller.update(context)
        plant.step(command, dt)
        measurement = Measurement(
            time=plant.time,
            source_time=plant.time,
            pose=plant.pose,
            u=plant.state.u,
            v=plant.state.v,
            r=plant.state.r,
        )
        telemetry = controller.telemetry()
        times.append(float(plant.time))
        heading.append(float(plant.pose.theta))
        heading_ref.append(float(heading_reference))
        reference_model.append(float(telemetry.get("psi_m", math.nan)))
        yaw_rate.append(float(plant.state.r))
        rudder.append(float(telemetry.get("delta_cmd", 0.0)))
        kr.append(float(telemetry.get("kr", math.nan)))
        kx1.append(float(telemetry.get("kx1", math.nan)))
        kx2.append(float(telemetry.get("kx2", math.nan)))

    return StepResponseResult(
        adapt=adapt,
        initial_gain_scale=initial_gain_scale,
        time=np.asarray(times),
        heading=np.asarray(heading),
        heading_reference=np.asarray(heading_ref),
        reference_model=np.asarray(reference_model),
        yaw_rate=np.asarray(yaw_rate),
        rudder=np.asarray(rudder),
        kr=np.asarray(kr),
        kx1=np.asarray(kx1),
        kx2=np.asarray(kx2),
    )


def step_response_validation(
    vehicle: VehicleParameters | None = None,
    *,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p = vehicle or VehicleParameters()
    base: dict[str, Any] = dict(
        omega_n=5.5,
        zeta=1.0,
        gamma_psi=600.0,
        gamma_r=300.0,
        gamma_ref=600.0,
        sigma=1e-5,
        initial_gain_scale=0.10,
        heading_reference=0.15,
        duration=20.0,
    )
    if config_override:
        unknown = set(config_override) - set(base)
        if unknown:
            raise ValueError(f"Unknown step-response overrides: {sorted(unknown)}")
        base.update(config_override)
    model = MRACController(
        MRACConfig(omega_n=base["omega_n"], zeta=base["zeta"]), vehicle=p
    ).reference
    _, matching_kr = model.matching_gains(p, base["heading_reference"])
    adaptive = run_step_response(adapt=True, **base, vehicle=p)
    fixed = run_step_response(adapt=False, **base, vehicle=p)
    a = adaptive.summary(base["heading_reference"], matching_kr)
    f = fixed.summary(base["heading_reference"], matching_kr)
    rmse_improved = bool(
        a["reference_model_rmse"] < f["reference_model_rmse"] - 1e-4
    )
    kr_improved = bool(a["kr_moved_toward_matching"])
    return {
        "matching_kr_at_0_15": float(matching_kr),
        "adaptive": a,
        "fixed": f,
        "relative_rmse_improvement": float(
            (f["reference_model_rmse"] - a["reference_model_rmse"])
            / max(f["reference_model_rmse"], 1e-12)
        ),
        "adaptive_gain_moved_toward_matching": kr_improved,
        "rmse_improved_by_adaptation": rmse_improved,
        "pass": bool(
            a["all_finite"]
            and f["all_finite"]
            and a["max_abs_rudder"] <= 0.65 + 1e-12
        ),
        "adaptation_superiority_demonstrated": bool(rmse_improved and kr_improved),
        "gate_version": "A1-v2-split-gates",
        "series": {
            "adaptive": adaptive,
            "fixed": fixed,
        },
    }
