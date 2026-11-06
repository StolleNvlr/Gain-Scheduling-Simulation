"""Deterministic Matplotlib figures for the V4 report.

Every function returns the path it wrote.  The functions never invent data:
they consume a ``summary.json`` produced by the protocol or an
``episodes.jsonl`` file.  Figures are saved at 200 dpi with a white background
so they can be included in the article without a dark-theme surprise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..control.gain_scheduler import GainScheduler
from ..simulation.world import scenario_by_name

COLORS = [
    "#1f4e79",
    "#c55a11",
    "#548235",
    "#7030a0",
    "#bf9000",
    "#2e75b6",
    "#a61c00",
    "#7f6000",
    "#385723",
    "#833c0b",
]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_gain_schedule(
    path: str | Path,
    *,
    speeds: Sequence[float] | None = None,
) -> Path:
    """Corrected schedule versus the MATLAB absolute-gain reconstruction."""

    if speeds is None:
        speeds = np.linspace(0.05, 1.0, 200)
    corrected = GainScheduler.corrected()
    matlab = GainScheduler.matlab_absolute()
    c_kp = [corrected.gains(float(s)).kp for s in speeds]
    c_kd = [corrected.gains(float(s)).kd for s in speeds]
    m_kp = [matlab.gains(float(s)).kp for s in speeds]
    m_kd = [matlab.gains(float(s)).kd for s in speeds]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(speeds, c_kp, color=COLORS[0], label="corrected $K_p$")
    axes[0].plot(speeds, m_kp, color=COLORS[1], linestyle="--", label="MATLAB $K_p$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("surge speed $u$ [m/s]")
    axes[0].set_ylabel("$K_p$ [rad$^{-1}$]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(speeds, c_kd, color=COLORS[0], label="corrected $K_d$")
    axes[1].plot(speeds, m_kd, color=COLORS[1], linestyle="--", label="MATLAB $K_d$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("surge speed $u$ [m/s]")
    axes[1].set_ylabel("$K_d$ [rad$^{-1}$ s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Gain schedule: corrected design vs MATLAB reconstruction")
    return _save(fig, Path(path))


def plot_success_rates(
    summary_or_path: str | Path | dict[str, Any],
    path: str | Path,
) -> Path:
    summary = (
        load_summary(summary_or_path)
        if isinstance(summary_or_path, (str, Path))
        else summary_or_path
    )
    conditions = [c["condition"] for c in summary["conditions"]]
    rates = [100.0 * c["success_rate"] for c in summary["conditions"]]
    lows = [
        100.0 * (c["success_rate"] - c["success_wilson_low"])
        for c in summary["conditions"]
    ]
    highs = [
        100.0 * (c["success_wilson_high"] - c["success_rate"])
        for c in summary["conditions"]
    ]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.72 * len(conditions)), 3.8))
    x = np.arange(len(conditions))
    ax.bar(
        x,
        rates,
        color=[COLORS[i % len(COLORS)] for i in range(len(conditions))],
        width=0.72,
    )
    ax.errorbar(
        x,
        rates,
        yerr=[lows, highs],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0.0, 105.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Goal reach with 95% Wilson intervals")
    return _save(fig, Path(path))


def plot_metric_by_condition(
    episodes_or_path: str | Path | Sequence[dict[str, Any]],
    path: str | Path,
    *,
    metric: str = "tracking_rmse",
    ylabel: str | None = None,
    success_only: bool = True,
) -> Path:
    episodes = (
        load_episodes(episodes_or_path)
        if isinstance(episodes_or_path, (str, Path))
        else list(episodes_or_path)
    )
    conditions = sorted({e["condition"] for e in episodes})
    data: list[list[float]] = []
    for condition in conditions:
        values = []
        for e in episodes:
            if e["condition"] != condition:
                continue
            if success_only and not e["metrics"].get("success", False):
                continue
            value = e["metrics"].get(metric)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        data.append(values)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.72 * len(conditions)), 3.8))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(4)
    for i, values in enumerate(data, start=1):
        if values:
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                i + jitter,
                values,
                s=8,
                color="#222222",
                alpha=0.45,
                zorder=3,
            )
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel or metric)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        f"{metric}" + (" (successful episodes only)" if success_only else "")
    )
    return _save(fig, Path(path))


def plot_scenario_trajectories(
    episodes_or_path: str | Path | Sequence[dict[str, Any]],
    path: str | Path,
    *,
    scenario: str,
    conditions: Sequence[str],
    seed: int | None = None,
) -> Path:
    episodes = (
        load_episodes(episodes_or_path)
        if isinstance(episodes_or_path, (str, Path))
        else list(episodes_or_path)
    )
    scene = scenario_by_name(scenario)
    subset = [e for e in episodes if e["scenario"] == scenario]
    if seed is not None:
        with_seed = [e for e in subset if e["seed"] == seed]
        if with_seed:
            subset = with_seed
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ob in scene.obstacles:
        if hasattr(ob, "min_x"):
            ax.add_patch(
                plt.Rectangle(
                    (ob.min_x, ob.min_y),
                    ob.max_x - ob.min_x,
                    ob.max_y - ob.min_y,
                    color="#555555",
                    alpha=0.75,
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (ob.cx, ob.cy), ob.radius, color="#555555", alpha=0.75
                )
            )
    hidden_legend_added = False
    for ob in scene.hidden_obstacles:
        label = None if hidden_legend_added else "unmapped"
        hidden_legend_added = True
        if hasattr(ob, "min_x"):
            ax.add_patch(
                plt.Rectangle(
                    (ob.min_x, ob.min_y),
                    ob.max_x - ob.min_x,
                    ob.max_y - ob.min_y,
                    facecolor="#cccccc",
                    edgecolor="#c00000",
                    hatch="//",
                    alpha=0.6,
                    label=label,
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (ob.cx, ob.cy),
                    ob.radius,
                    facecolor="#cccccc",
                    edgecolor="#c00000",
                    hatch="//",
                    alpha=0.6,
                    label=label,
                )
            )
    plotted_plan = False
    for i, condition in enumerate(conditions):
        candidates = [e for e in subset if e["condition"] == condition]
        if not candidates:
            continue
        record = candidates[0]
        if not plotted_plan and record.get("planned_path"):
            plan = np.asarray(record["planned_path"], dtype=float)
            ax.plot(
                plan[:, 0],
                plan[:, 1],
                color="#999999",
                linewidth=1.0,
                linestyle=":",
                label="planned (shared)",
            )
            plotted_plan = True
        path_xy = record.get("executed_path") or []
        if len(path_xy) > 1:
            xy = np.asarray(path_xy, dtype=float)
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=COLORS[i % len(COLORS)],
                linewidth=1.4,
                label=f"{condition} (seed {record['seed']})",
            )
    ax.scatter(
        [scene.start.x], [scene.start.y], marker="o", color="#222222", zorder=5
    )
    ax.scatter(
        [scene.goal.x], [scene.goal.y], marker="*", s=110, color="#c00000", zorder=5
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(scene.bounds[0], scene.bounds[2])
    ax.set_ylim(scene.bounds[1], scene.bounds[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(scenario)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    return _save(fig, Path(path))


def plot_adaptation(
    episode: dict[str, Any] | str | Path,
    path: str | Path,
    *,
    condition: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> Path:
    record: dict[str, Any]
    if isinstance(episode, (str, Path)):
        records = load_episodes(episode)
        candidates = [
            e
            for e in records
            if (condition is None or e["condition"] == condition)
            and (scenario is None or e["scenario"] == scenario)
            and (seed is None or e["seed"] == seed)
        ]
        if not candidates:
            raise ValueError("no episode matches the requested selector")
        record = candidates[0]
    else:
        record = episode
    telemetry = record.get("telemetry") or []
    if not telemetry:
        raise ValueError("episode has no telemetry; enable record_telemetry")
    time = np.array([t["time"] for t in telemetry])
    kr = np.array([t["controller"].get("kr", np.nan) for t in telemetry])
    kx1 = np.array([t["controller"].get("kx1", np.nan) for t in telemetry])
    kx2 = np.array([t["controller"].get("kx2", np.nan) for t in telemetry])
    err = np.array(
        [t["controller"].get("error_norm", np.nan) for t in telemetry]
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].plot(time, kr, color=COLORS[1], label="$\\hat K_r$")
    axes[0].plot(time, -kx1, color=COLORS[0], label="$-\\hat K_{x1}$")
    axes[0].plot(time, -kx2, color=COLORS[2], label="$-\\hat K_{x2}$")
    axes[0].set_ylabel("gain magnitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(time, err, color="#333333")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("$\\|e\\|$ [rad, rad/s]")
    axes[1].grid(alpha=0.25)
    axes[0].set_title(
        f"MRAC telemetry: {record['condition']} / {record['scenario']} / "
        f"seed {record['seed']}"
    )
    return _save(fig, Path(path))


def plot_metric_scatter(
    episodes_or_path: str | Path | Sequence[dict[str, Any]],
    path: str | Path,
    *,
    x_metric: str = "mean_abs_rudder",
    y_metric: str = "tracking_rmse",
) -> Path:
    episodes = (
        load_episodes(episodes_or_path)
        if isinstance(episodes_or_path, (str, Path))
        else list(episodes_or_path)
    )
    conditions = sorted({e["condition"] for e in episodes})
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, condition in enumerate(conditions):
        xs: list[float] = []
        ys: list[float] = []
        for e in episodes:
            if e["condition"] != condition or not e["metrics"].get("success"):
                continue
            x = e["metrics"].get(x_metric)
            y = e["metrics"].get(y_metric)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if xs:
            ax.scatter(
                xs,
                ys,
                s=14,
                alpha=0.65,
                color=COLORS[i % len(COLORS)],
                label=condition,
            )
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Effort–accuracy trade-off")
    return _save(fig, Path(path))
