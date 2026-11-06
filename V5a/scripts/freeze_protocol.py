#!/usr/bin/env python
"""Freeze a V4 protocol configuration before the test phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.experiments.protocol import freeze_protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "experiment_test.yaml")
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "experiments" / "protocol_manifest.json"),
    )
    parser.add_argument(
        "--phase",
        default="test",
        choices=["tune", "validation", "test"],
        help="phase recorded in the manifest (default test)",
    )
    args = parser.parse_args()

    manifest = freeze_protocol(
        args.config,
        args.manifest,
        phase_override=args.phase,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
