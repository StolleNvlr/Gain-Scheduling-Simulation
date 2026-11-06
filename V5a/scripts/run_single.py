#!/usr/bin/env python
"""Run one IARA V4 episode, print metrics and optionally plot it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.dynamics.disturbance import DisturbanceConfig  # noqa: E402
from iara_v4.experiments.protocol import _json_default  # noqa: E402
from iara_v4.simulation.runner import (  # noqa: E402
    EpisodeConfig,
    SensorConfig,
    condition_spec,
    run_episode,
)
from iara_v4.simulation.world import scenario_by_name  # noqa: E402


def plot_episode(result, scenario, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for ob in scenario.obstacles:
        if hasattr(ob, "min_x"):
            ax.add_patch(
                Rectangle(
                    (ob.min_x, ob.min_y),
                    ob.max_x - ob.min_x,
                    ob.max_y - ob.min_y,
                    facecolor="0.55",
                    edgecolor="black",
                    alpha=0.7,
                )
            )
    for ob in scenario.hidden_obstacles:
        if hasattr(ob, "min_x"):
            ax.add_patch(
                Rectangle(
                    (ob.min_x, ob.min_y),
                    ob.max_x - ob.min_x,
                    ob.max_y - ob.min_y,
                    facecolor="0.85",
                    edgecolor="red",
                    hatch="//",
                    alpha=0.7,
                )
            )
    if result.planned_path:
        ax.plot(
            [p[0] for p in result.planned_path],
            [p[1] for p in result.planned_path],
            color="#777777",
            linestyle=":",
            lw=1.5,
            label="planned",
        )
    if result.executed_path:
        ax.plot(
            [p[0] for p in result.executed_path],
            [p[1] for p in result.executed_path],
            color="#1f4e79",
            lw=1.4,
            label="executed",
        )
    ax.plot(scenario.start.x, scenario.start.y, "o", color="#222222", label="start")
    ax.plot(
        scenario.goal.x,
        scenario.goal.y,
        "*",
        markersize=13,
        color="#c00000",
        label="goal",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{result.condition} / {result.scenario} / seed {result.seed}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if result.trajectory:
        t = [row[0] for row in result.trajectory]
        axes[0, 1].plot(t, [row[4] for row in result.trajectory], label="$u$ [m/s]")
        axes[0, 1].plot(
            t, [row[7] for row in result.trajectory], "--", label="$v_{cmd}$ [m/s]"
        )
        axes[0, 1].set_xlabel("time [s]")
        axes[0, 1].set_ylabel("speed [m/s]")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(t, [row[9] for row in result.trajectory], label="delta cmd")
        axes[1, 0].plot(
            t, [row[10] for row in result.trajectory], "--", label="delta applied"
        )
        axes[1, 0].set_xlabel("time [s]")
        axes[1, 0].set_ylabel("rudder [rad]")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].plot(t, [row[13] for row in result.trajectory], label="cross-track [m]")
        axes[1, 1].plot(
            t, [row[14] for row in result.trajectory], label="heading error [rad]"
        )
        axes[1, 1].set_xlabel("time [s]")
        axes[1, 1].set_ylabel("error")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="coppelia_default")
    parser.add_argument("--condition", default="gs")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--phase", default="validation", choices=["tune", "validation", "test"]
    )
    parser.add_argument(
        "--planner", default="rrt_star", choices=["rrt_star", "astar", "apf"]
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--latency",
        type=int,
        default=2,
        help="measurement latency in control steps (default 2)",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-disturbance", action="store_true")
    parser.add_argument("--record-trajectory", action="store_true")
    parser.add_argument("--record-telemetry", action="store_true")
    parser.add_argument("--output", default=None, help="write the full episode JSON here")
    parser.add_argument("--json", action="store_true", help="print the full episode JSON")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    spec = condition_spec(args.condition)
    scenario = scenario_by_name(args.scenario)
    cfg = EpisodeConfig(
        scenario=scenario,
        condition=args.condition,
        seed=args.seed,
        phase=args.phase,
        spec=spec,
        planner=args.planner,
        dt=args.dt,
        record_trajectory=args.record_trajectory,
        record_telemetry=args.record_telemetry,
        sensor=SensorConfig(
            latency_steps=args.latency,
            dropout_probability=args.dropout,
        ),
        disturbance=DisturbanceConfig(enabled=not args.no_disturbance),
    )
    result = run_episode(cfg)
    payload = result.to_dict()
    text = json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    elif args.json:
        print(text)

    m = result.metrics
    print(f"condition:  {result.condition} ({result.plant} plant)")
    print(f"scenario:   {result.scenario} (seed {result.seed}, phase {result.phase})")
    print(
        f"result:     success={m.success} collision={m.collision} "
        f"timeout={m.timeout} termination={m.termination}"
    )
    print(
        f"motion:     time={m.time:.2f}s length={m.executed_length:.2f}m "
        f"mean_speed={m.mean_speed:.3f}m/s"
    )
    print(
        f"tracking:   rmse={m.tracking_rmse:.3f}m heading_rmse={m.heading_rmse:.3f}rad "
        f"max_cross_track={m.max_cross_track:.3f}m min_clearance={m.min_clearance:.3f}m"
    )
    print(
        f"control:    mean|delta|={m.mean_abs_rudder:.4f}rad "
        f"yaw_rms={m.yaw_rate_rms:.3f}rad/s "
        f"sat={m.saturated_fraction:.1%} "
        f"adapt_effort={m.mean_adaptation_effort:.5f}"
    )
    print(
        f"planning:   {result.planning.get('reason', 'n/a')} "
        f"time={m.planning_time:.3f}s nodes={m.planner_nodes}"
    )
    print(
        f"timing:     controller={1000.0 * m.controller_time:.1f}ms "
        f"plant={1000.0 * m.plant_time:.1f}ms wall={result.wall_time:.2f}s"
    )
    if args.plot:
        plot_episode(result, scenario, Path(args.plot))
        print(f"plot:       {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
