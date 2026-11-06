from __future__ import annotations

from iara_v4.simulation.runner import EpisodeConfig, SensorConfig, run_episode
from iara_v4.simulation.world import scenario_by_name


def test_same_seed_reproduces_physical_metrics_only() -> None:
    config = EpisodeConfig(
        scenario=scenario_by_name("open_water"),
        condition="gs",
        seed=20,
        phase="validation",
        sensor=SensorConfig(),
    )
    first = run_episode(config)
    second = run_episode(config)
    assert first.metrics.to_dict() | {
        "planning_time": 0.0,
        "controller_time": 0.0,
        "plant_time": 0.0,
    } == second.metrics.to_dict() | {
        "planning_time": 0.0,
        "controller_time": 0.0,
        "plant_time": 0.0,
    }
