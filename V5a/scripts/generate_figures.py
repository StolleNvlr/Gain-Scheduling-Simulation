#!/usr/bin/env python
"""Generate deterministic figures from a V4 or V5a1 result directory.

The script only reads `summary.json` / `episodes.jsonl`; it never recomputes an
experiment. A V5a1 directory may instead contain the `v5a1_protocol.json`
container written by `scripts/run_v5a1_experiment.py`. Telemetry figures are
generated only when a telemetry protocol was run (config/telemetry.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.analysis.plots import (  # noqa: E402
    plot_adaptation,
    plot_gain_schedule,
    plot_metric_by_condition,
    plot_metric_scatter,
    plot_scenario_trajectories,
    plot_success_rates,
)


def _episode_source(results: Path) -> Path | list[dict[str, Any]]:
    """Accept both V4 JSONL and the V5a1 protocol JSON container."""

    episodes = results / "episodes.jsonl"
    if episodes.exists():
        return episodes
    protocol = results / "v5a1_protocol.json"
    if protocol.exists():
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        records = payload.get("episodes") if isinstance(payload, dict) else payload
        if records:
            return records
        raise SystemExit(f"no episodes found in {protocol}")
    raise SystemExit(
        f"missing episodes.jsonl or v5a1_protocol.json in {results}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help="directory with episodes.jsonl (or v5a1_protocol.json)",
    )
    parser.add_argument("--output", default=None, help="figure directory")
    parser.add_argument("--scenario", default="coppelia_default")
    parser.add_argument(
        "--conditions",
        default="pp,gs,mrac,mrac_lowgain",
        help="comma-separated conditions for the trajectory figure",
    )
    parser.add_argument(
        "--telemetry-results",
        default=None,
        help="result directory from config/telemetry.yaml",
    )
    parser.add_argument(
        "--adaptation-condition", default="mrac_lowgain"
    )
    parser.add_argument("--adaptation-scenario", default="coppelia_default")
    args = parser.parse_args()

    results = Path(args.results)
    summary = results / "summary.json"
    episodes = _episode_source(results)
    out = Path(args.output) if args.output else results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    written: list[Path] = []
    if summary.exists():
        written.append(plot_success_rates(summary, out / "success_rates.png"))
    else:
        print(f"warning: {summary} not found; skipping success_rates.png")
    written.append(plot_gain_schedule(out / "gain_schedule.png"))
    for metric, label in (
        ("tracking_rmse", "cross-track RMSE [m]"),
        ("heading_rmse", "heading RMSE [rad]"),
        ("time", "time to goal [s]"),
        ("mean_abs_rudder", "mean |rudder| [rad]"),
        ("min_clearance", "minimum clearance [m]"),
        ("yaw_rate_rms", "yaw-rate RMS [rad/s]"),
    ):
        written.append(
            plot_metric_by_condition(
                episodes,
                out / f"{metric}.png",
                metric=metric,
                ylabel=label,
            )
        )
    written.append(
        plot_scenario_trajectories(
            episodes,
            out / f"trajectories_{args.scenario}.png",
            scenario=args.scenario,
            conditions=conditions,
        )
    )
    written.append(
        plot_metric_scatter(
            episodes,
            out / "effort_accuracy.png",
            x_metric="mean_abs_rudder",
            y_metric="tracking_rmse",
        )
    )
    if args.telemetry_results:
        telemetry_episodes = Path(args.telemetry_results) / "episodes.jsonl"
        if telemetry_episodes.exists():
            written.append(
                plot_adaptation(
                    telemetry_episodes,
                    out / "adaptation.png",
                    condition=args.adaptation_condition,
                    scenario=args.adaptation_scenario,
                )
            )
            written.append(
                plot_scenario_trajectories(
                    telemetry_episodes,
                    out / f"telemetry_trajectories_{args.adaptation_scenario}.png",
                    scenario=args.adaptation_scenario,
                    conditions=["gs", "mrac", "mrac_lowgain"],
                )
            )

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
