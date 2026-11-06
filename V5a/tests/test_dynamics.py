from __future__ import annotations

import math

import numpy as np
import pytest

from iara_v4.core.geometry import RectangleObstacle
from iara_v4.core.types import ControlCommand, Pose2D
from iara_v4.dynamics.actuator import ActuatorConfig, ActuatorModel, _deadband
from iara_v4.dynamics.disturbance import DisturbanceConfig, DisturbanceModel
from iara_v4.dynamics.model import VehicleParameters, physical_checks
from iara_v4.dynamics.plant import (
    DynamicPlantConfig,
    DynamicUSVPlant,
    KinematicPlantConfig,
    KinematicUSVPlant,
)


def test_plant_invariants() -> None:
    checks = physical_checks()
    assert checks["m_positive_definite"] is True
    assert checks["coriolis_skew_symmetric"] is True
    assert abs(checks["coriolis_quadratic_form"]) < 1e-12
    assert checks["damping_dissipative"] > 0.0
    assert checks["damping_positive_semidefinite"] is True
    assert checks["yaw_b_low_speed"] > 0.0


def test_plant_step_separates_tau_r_cmd_from_applied() -> None:
    cfg = DynamicPlantConfig(disturbance=DisturbanceConfig(enabled=False))
    plant = DynamicUSVPlant(cfg, Pose2D(0.0, 0.0, 0.0))
    step = plant.step(ControlCommand(v=0.3, omega=0.0, delta_rudder=0.65), 0.05)
    assert step.tau_r_cmd != step.state.tau_r
    assert abs(step.state.tau_r) <= abs(step.tau_r_cmd) + 1e-9


def test_disturbance_is_reproducible_and_seed_dependent() -> None:
    cfg = DisturbanceConfig()
    a = DisturbanceModel(cfg, seed=7, scenario="open_water")
    b = DisturbanceModel(cfg, seed=7, scenario="open_water")
    c = DisturbanceModel(cfg, seed=8, scenario="open_water")
    for _ in range(20):
        a.update(0.05)
        b.update(0.05)
        c.update(0.05)
    fa = a.force(0.1)
    fb = b.force(0.1)
    fc = c.force(0.1)
    assert np.allclose(fa, fb)
    assert not np.allclose(fa, fc)


def test_disturbance_can_be_disabled() -> None:
    model = DisturbanceModel(DisturbanceConfig(enabled=False), seed=1)
    assert np.allclose(model.force(0.0), np.zeros(3))


def test_actuator_saturation_and_rate_limit() -> None:
    model = ActuatorModel(ActuatorConfig(delta_rate_max=3.0, delta_time_constant=0.08))
    applied, _, saturated, _ = model.rudder_angle(10.0, 0.05)
    assert abs(applied) <= 0.65 + 1e-12
    assert saturated is True
    applied2, _, _, _ = model.rudder_angle(-10.0, 0.05)
    assert abs(applied2 - applied) <= 3.0 * 0.05 + 1e-12


def test_actuator_deadband_is_applied_after_lag() -> None:
    assert _deadband(0.001, 0.005) == (0.0, True)
    assert _deadband(0.010, 0.005) == (pytest.approx(0.005), False)
    model = ActuatorModel(ActuatorConfig(delta_deadband=0.05))
    applied, _, _, dead = model.rudder_angle(0.01, 0.05)
    assert applied == 0.0


def test_dynamic_plant_tracks_surge_reference() -> None:
    cfg = DynamicPlantConfig(
        disturbance=DisturbanceConfig(enabled=False), dt=0.05, substeps=4
    )
    plant = DynamicUSVPlant(cfg, Pose2D(0.0, 0.0, 0.0))
    command = ControlCommand(v=0.35, omega=0.0)
    for _ in range(400):
        plant.step(command, 0.05)
    assert plant.time == pytest.approx(20.0)
    assert plant.state.u == pytest.approx(0.35, abs=0.02)
    assert plant.state.pose.x > 5.0
    assert abs(plant.state.pose.y) < 0.05
    assert math.isfinite(plant.state.pose.theta)


def test_dynamic_plant_collision_uses_oriented_footprint() -> None:
    obstacle = RectangleObstacle(0.0, 0.0, 0.5, 0.5, "block")
    plant = DynamicUSVPlant(
        DynamicPlantConfig(disturbance=DisturbanceConfig(enabled=False)),
        Pose2D(0.0, 0.0, 0.0),
    )
    assert plant.collision((obstacle,)) is True
    assert plant.clearance((obstacle,)) < 0.0


def test_kinematic_plant_is_exact() -> None:
    plant = KinematicUSVPlant(KinematicPlantConfig(dt=0.05), Pose2D(0.0, 0.0, 0.0))
    command = ControlCommand(v=0.2, omega=0.1)
    for _ in range(100):
        plant.step(command, 0.05)
    assert plant.state.u == pytest.approx(0.2)
    assert plant.state.r == pytest.approx(0.1)
    assert plant.time == pytest.approx(5.0)


def test_vehicle_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        VehicleParameters(m_u=0.0)
    with pytest.raises(ValueError):
        VehicleParameters(b_0=-1.0)
    with pytest.raises(ValueError):
        VehicleParameters(rudder_max=2.0)
