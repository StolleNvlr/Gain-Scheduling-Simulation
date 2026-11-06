"""V5a1 reporting wrapper around the V4 statistical implementation."""

from __future__ import annotations

from typing import Any

from iara_v4.experiments.report import build_summary as build_summary_v4
from iara_v4.simulation.runner import EpisodeResult


def build_summary_v5a(
    results_by_rep: dict[int, list[EpisodeResult]],
) -> dict[str, Any]:
    """Build the V4 summary and add repetition-level summaries."""

    results = [result for group in results_by_rep.values() for result in group]
    if not results:
        raise ValueError("cannot summarise an empty V5a1 result set")
    summary = build_summary_v4(results)
    summary["v5a1_repetitions"] = {
        str(repetition): build_summary_v4(repetition_results)
        for repetition, repetition_results in sorted(results_by_rep.items())
    }
    return summary
