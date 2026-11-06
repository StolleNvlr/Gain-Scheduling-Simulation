"""Non-deterministic, paired multi-repetition V5a1 protocol."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

from iara_v4.experiments.protocol import ExperimentProtocol
from iara_v4.simulation.runner import EpisodeResult

from ..hal.base import HALConfig, HardwareHAL
from .episode import run_episode_v5a


def run_protocol_v5a(
    protocol: ExperimentProtocol,
    hal_factory: Callable[[], HardwareHAL],
    n_repetitions: int = 3,
    hal_config: HALConfig | None = None,
) -> dict[str, Any]:
    """Run paired ``(condition, scenario, seed)`` cells repeatedly.

    The returned ``episodes`` list preserves every raw result.  ``aggregates``
    reports mean and sample standard deviation over repetitions.
    """

    if n_repetitions < 1:
        raise ValueError("n_repetitions must be at least 1")
    results: list[EpisodeResult] = []
    for cfg in protocol.iter_episodes():
        for repetition in range(n_repetitions):
            result = run_episode_v5a(cfg, hal_factory(), hal_config)
            result.config["repetition"] = repetition
            results.append(result)

    grouped: dict[tuple[str, str, int], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[(result.condition, result.scenario, result.seed)].append(result)
    aggregates: list[dict[str, Any]] = []
    for (condition, scenario, seed), cell in sorted(grouped.items()):
        row: dict[str, Any] = {
            "condition": condition,
            "scenario": scenario,
            "seed": seed,
            "n_repetitions": len(cell),
        }
        for metric in ("time", "tracking_rmse", "heading_rmse", "mean_abs_rudder"):
            values = [float(getattr(r.metrics, metric)) for r in cell]
            mean = sum(values) / len(values)
            row[metric] = {
                "mean": mean,
                "std": (
                    math.sqrt(
                        sum((value - mean) ** 2 for value in values)
                        / (len(values) - 1)
                    )
                    if len(values) > 1
                    else 0.0
                ),
            }
        aggregates.append(row)
    return {
        "n_repetitions": n_repetitions,
        "episodes": [result.to_dict() for result in results],
        "aggregates": aggregates,
        "paired_keys": [
            {"condition": c, "scenario": s, "seed": seed}
            for c, s, seed in sorted(grouped)
        ],
    }
