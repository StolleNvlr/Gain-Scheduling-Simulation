from __future__ import annotations

import math

import pytest

from iara_v4.simulation.runner import (
    EpisodeConfig,
    SensorConfig,
    condition_spec,
    run_episode,
)
from iara_v4.simulation.world import scenario_by_name


def _episode(condition: str, scenario: str, seed: int = 20, **kwargs) -> EpisodeConfig:
    return EpisodeConfig(
        scenario=scenario_by_name(scenario),
        condition=condition,
        seed=seed,
        phase="validation",
        sensor=SensorConfig(latency_steps=2),
        **kwargs,
    )


def test_open_water_episode_succeeds() -> None:
    result = run_episode(_episode("pp", "open_water"))
    assert result.metrics.success is True
    assert result.metrics.collision is False
    assert result.metrics.timeout is False
    assert result.metrics.tracking_rmse < 0.10
    assert result.metrics.mean_speed > 0.2


def test_episode_is_deterministic() -> None:
    cfg = _episode("gs", "coppelia_default")
    first = run_episode(cfg).to_dict()
    second = run_episode(cfg).to_dict()
    for payload in (first, second):
        payload.pop("wall_time")
        payload["planning"].pop("planning_time")
        for key in ("controller_time", "plant_time", "planning_time"):
            payload["metrics"].pop(key, None)
    assert first == second


def test_global_plan_is_shared_across_conditions() -> None:
    a = run_episode(_episode("pp", "coppelia_default"))
    b = run_episode(_episode("gs", "coppelia_default"))
    assert a.planned_path == b.planned_path
    assert a.planning["cost"] == pytest.approx(b.planning["cost"])


def test_kinematic_and_dynamic_plant_are_both_exercised() -> None:
    dynamic = run_episode(_episode("pp", "coppelia_default"))
    kinematic = run_episode(_episode("pp_kinematic", "coppelia_default"))
    assert dynamic.plant == "dynamic"
    assert kinematic.plant == "kinematic"
    assert dynamic.metrics.time != kinematic.metrics.time


def test_mrac_telemetry_is_recorded() -> None:
    result = run_episode(
        _episode(
            "mrac",
            "open_water",
            record_trajectory=True,
            record_telemetry=True,
        )
    )
    assert result.telemetry
    first = result.telemetry[0]
    for key in ("kr", "kx1", "kx2", "error_norm", "delta_cmd"):
        assert key in first["controller"]
    assert len(result.trajectory) == len(result.telemetry)


def test_low_gain_condition_uses_scaled_initial_gain() -> None:
    spec = condition_spec("mrac_lowgain")
    assert spec.initial_gain_scale == pytest.approx(0.30)
    assert spec.adapt is True
    fixed = condition_spec("mrac_lowgain_fixed")
    assert fixed.adapt is False


def test_mismatch_condition_declares_controller_vehicle() -> None:
    result = run_episode(_episode("gs_mismatch", "coppelia_default"))
    assert result.config["model_mismatch"] is True
    assert (
        result.config["controller_vehicle"]["b_n"]
        < result.config["vehicle"]["b_n"]
    )


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(KeyError):
        EpisodeConfig(
            scenario=scenario_by_name("open_water"),
            condition="not_a_controller",
            seed=20,
            phase="validation",
        )


def test_kinematic_plant_accepts_controller_without_wheels(monkeypatch) -> None:
    from iara_v4.core.types import ControlCommand

    class Stub:
        name = "stub"

        def reset(self, *args, **kwargs):
            return None

        def update(self, context):
            return ControlCommand(v=0.2, omega=0.0)

        def telemetry(self):
            return {}

    monkeypatch.setattr(
        "iara_v4.simulation.runner._make_controller",
        lambda *args, **kwargs: Stub(),
    )
    result = run_episode(_episode("pp_kinematic", "open_water"))
    assert result.metrics.control_energy >= 0.0


def test_vfh_reduces_collision_on_hidden_obstacle() -> None:
    baseline = run_episode(_episode("gs", "hidden_block"))
    reactive = run_episode(_episode("gs_vfh", "hidden_block"))
    safety_improved = (
        (reactive.metrics.collision is False and baseline.metrics.collision is True)
        or reactive.metrics.min_clearance > baseline.metrics.min_clearance
    )
    assert safety_improved


def test_smoothness_metrics_are_separate_and_defined_for_both_plants() -> None:
    for condition in ("pp", "pp_kinematic"):
        result = run_episode(_episode(condition, "open_water"))
        assert result.metrics.smoothness_cmd >= 0.0
        assert result.metrics.smoothness_actuator >= 0.0
        assert result.metrics.smoothness == result.metrics.smoothness_actuator


def test_smoothness_matches_manual_rms() -> None:
    from iara_v4.simulation.runner import compute_smoothness

    series = [0.0, 0.1, 0.2, 0.3]
    dt = 0.1
    expected = math.sqrt(sum((0.1 / dt) ** 2 for _ in range(3)) / 3)
    assert compute_smoothness(series, dt) == pytest.approx(expected)
