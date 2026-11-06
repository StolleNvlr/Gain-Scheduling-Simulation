from __future__ import annotations

from pathlib import Path

from iara_v4.analysis.plots import plot_scenario_trajectories
from iara_v4.simulation.runner import EpisodeConfig, SensorConfig, run_episode
from iara_v4.simulation.world import scenario_by_name


def test_hidden_obstacles_render_to_a_file(tmp_path: Path) -> None:
    cfg = EpisodeConfig(
        scenario=scenario_by_name("coppelia_hidden"),
        condition="gs",
        seed=20,
        phase="validation",
        sensor=SensorConfig(),
    )
    result = run_episode(cfg)
    out = tmp_path / "trajectory.png"
    plot_scenario_trajectories(
        [result.to_dict()],
        out,
        scenario="coppelia_hidden",
        conditions=["gs"],
    )
    assert out.exists()
    assert out.stat().st_size > 10_000
