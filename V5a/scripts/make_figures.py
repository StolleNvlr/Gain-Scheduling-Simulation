#!/usr/bin/env python
"""Generate report figures from protocol JSON artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.analysis.plots import (  # noqa: E402
    plot_gain_schedule,
    plot_metric_by_condition,
    plot_success_rates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_gain_schedule(args.output_dir / "gain_schedule.png")
    plot_metric_by_condition(
        args.episodes,
        args.output_dir / "tracking_rmse.png",
        metric="tracking_rmse",
    )
    if args.summary:
        plot_success_rates(args.summary, args.output_dir / "success_rates.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
