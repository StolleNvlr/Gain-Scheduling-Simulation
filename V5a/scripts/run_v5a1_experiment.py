"""Run a V5a1 protocol against a CoppeliaSim scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Adiciona a raiz do V4 ao caminho do Python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.experiments.protocol import ExperimentProtocol
from iara_v5a.runner.protocol import run_protocol_v5a
from iara_v5a.hal.coppelia import CoppeliaHAL, CoppeliaHALConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/v5a1"))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    protocol = ExperimentProtocol.from_yaml(args.config)
    hal_config = CoppeliaHALConfig(
        scene_path=str(args.scene) if args.scene is not None else None,
        dt=protocol.dt,
    )
    result = run_protocol_v5a(
        protocol,
        lambda: CoppeliaHAL(
            host=hal_config.host,
            port=hal_config.port,
        ),
        n_repetitions=args.repetitions,
        hal_config=hal_config,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "v5a1_protocol.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Keep the flat JSONL artifact expected by scripts/generate_figures.py.
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for episode in result["episodes"]:
            fh.write(json.dumps(episode, ensure_ascii=False) + "\n")
    print(f"wrote {len(result['episodes'])} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
