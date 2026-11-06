from __future__ import annotations

import math

import pytest

from iara_v4.experiments.report import build_summary
from iara_v4.experiments.statistics import (
    bootstrap_ci,
    cliffs_delta,
    competing_risks,
    holm_correction,
    mcnemar,
    wilson_ci,
    wilcoxon,
)
from iara_v4.simulation.runner import EpisodeConfig, SensorConfig, run_episode
from iara_v4.simulation.world import scenario_by_name


def test_wilson_interval_contains_estimate() -> None:
    low, high = wilson_ci(7, 10)
    assert 0.0 <= low < 0.7 < high <= 1.0
    assert low == pytest.approx(0.3968, abs=1e-3)


def test_bootstrap_is_deterministic_with_rng() -> None:
    import numpy as np

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = bootstrap_ci(data, np.median, n_boot=200, rng=np.random.default_rng(3))
    second = bootstrap_ci(data, np.median, n_boot=200, rng=np.random.default_rng(3))
    assert first == second
    assert first[0] <= first[2] <= first[1]


def test_mcnemar_exact_and_holm() -> None:
    result = mcnemar(8, 2)
    assert result.statistic == 2
    assert 0.0 <= result.p_value <= 1.0
    corrected = holm_correction([0.01, 0.04, 0.03])
    assert [row.p_adjusted for row in corrected] == pytest.approx([0.03, 0.06, 0.06])


def test_wilcoxon_and_cliffs_delta() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.5, 2.5, 3.5, 4.5, 5.5]
    result = wilcoxon(x, y)
    assert result.n == 5
    assert result.median_difference == pytest.approx(-0.5)
    assert cliffs_delta(x, y) == pytest.approx(-0.2)


def test_competing_risks_separates_failure_from_success() -> None:
    result = competing_risks([1.0, 2.0, 3.0, 4.0], [1, 2, 1, 2])
    assert result.successes == 2
    assert result.failures == 2
    assert 0.0 <= result.cif_success[-1] <= 1.0
    assert 0.0 <= result.cif_failure[-1] <= 1.0


def test_build_summary_from_paired_episodes() -> None:
    def episode(condition: str):
        return run_episode(
            EpisodeConfig(
                scenario=scenario_by_name("open_water"),
                condition=condition,
                seed=20,
                phase="validation",
                sensor=SensorConfig(latency_steps=2),
            )
        )

    results = [episode("pp"), episode("gs")]
    summary = build_summary(results)
    assert summary["n_episodes"] == 2
    assert summary["n_conditions"] == 2
    assert summary["conditions"][0]["n"] == 1
    assert summary["conditions"][0]["success_rate"] in (0.0, 1.0)
    assert summary["pairwise"]
    success_rows = [r for r in summary["pairwise"] if r["metric"] == "success"]
    assert success_rows
    assert math.isfinite(float(success_rows[0]["p_value"]))
    assert summary["scenario_results"][0]["successes"] in (0, 1)


def test_bootstrap_is_fast_after_vectorization() -> None:
    import time

    import numpy as np

    data = list(np.random.default_rng(0).normal(size=200))
    t0 = time.perf_counter()
    bootstrap_ci(data, np.median, n_boot=2000, rng=np.random.default_rng(1))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"bootstrap_ci took {elapsed:.2f}s; expected < 1s"
