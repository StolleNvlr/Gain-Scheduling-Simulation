#!/usr/bin/env python
"""Run the IARA V4 experiment protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.experiments.protocol import (  # noqa: E402
    ExperimentProtocol,
    check_test_authorization,
    freeze_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "experiment.yaml")
    )
    parser.add_argument(
        "--phase", choices=["tune", "validation", "test"], default=None
    )
    parser.add_argument("--output", default=None, help="explicit output directory")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--authorize-test", action="store_true")
    parser.add_argument("--freeze-manifest", default=None)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "experiments" / "protocol_manifest.json"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.freeze_manifest:
        freeze_protocol(args.config, args.freeze_manifest, phase_override=args.phase)
        print(f"frozen manifest: {args.freeze_manifest}")
        return 0

    check_test_authorization(
        args.config,
        args.manifest,
        authorize=args.authorize_test,
        phase_override=args.phase,
    )
    protocol = ExperimentProtocol.from_yaml(args.config)
    if args.phase is not None:
        protocol = protocol.with_phase(args.phase)
    artifacts = protocol.run(
        output_dir=args.output,
        max_episodes=args.max_episodes,
        progress=not args.quiet,
    )
    print(f"episodes: {artifacts.n_episodes}")
    print(f"output:   {artifacts.output_dir}")
    print(f"summary:  {artifacts.summary_path}")
    print(f"report:   {artifacts.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
