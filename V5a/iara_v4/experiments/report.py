"""Aggregation, paired comparisons and human-readable V4 reports.

The V3 report logic is preserved where it remains valid.  The V4 metric set adds
the dynamic-control quantities (rudder effort, heading error, yaw rate,
adaptation effort, saturation, latency and controller/plant wall time).  Every
interval and test is computed from the episode records; no value in the report
is typed by hand.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.seeds import stable_hash
from ..simulation.runner import EpisodeResult
from .statistics import (
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    competing_risks,
    holm_correction,
    mcnemar,
    wilcoxon,
    wilson_ci,
)

CONTINUOUS_METRICS = [
    "executed_length",
    "planned_length",
    "time",
    "tracking_rmse",
    "heading_rmse",
    "max_cross_track",
    "control_energy",
    "smoothness_cmd",
    "smoothness_actuator",
    "min_clearance",
    "final_distance",
    "max_speed",
    "mean_speed",
    "mean_abs_rudder",
    "rudder_effort",
    "yaw_rate_rms",
    "peak_yaw_rate",
    "mean_adaptation_effort",
    "mean_latency",
    "controller_time",
    "plant_time",
    "saturated_fraction",
    "controller_saturated_fraction",
    "vfh_blocked_fraction",
    "max_path_curvature",
    "planning_time",
]

PAIRWISE_METRICS = [
    "executed_length",
    "time",
    "tracking_rmse",
    "heading_rmse",
    "max_cross_track",
    "control_energy",
    "smoothness_cmd",
    "smoothness_actuator",
    "min_clearance",
    "mean_abs_rudder",
    "rudder_effort",
    "yaw_rate_rms",
    "controller_saturated_fraction",
    "planned_length",
    "max_path_curvature",
]

# Execution metrics are only comparable when both episodes reached the goal.
SUCCESS_CONDITIONAL_METRICS = {
    "executed_length",
    "time",
    "tracking_rmse",
    "heading_rmse",
    "max_cross_track",
    "control_energy",
    "smoothness_cmd",
    "smoothness_actuator",
    "mean_speed",
    "max_speed",
    "mean_abs_rudder",
    "rudder_effort",
    "yaw_rate_rms",
    "peak_yaw_rate",
    "mean_adaptation_effort",
    "mean_latency",
    "controller_time",
    "plant_time",
    "saturated_fraction",
    "controller_saturated_fraction",
    "vfh_blocked_fraction",
    "min_clearance",
}


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _metric_stats(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> dict[str, Any]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            "n": 0,
            "median": None,
            "ci_low": None,
            "ci_high": None,
            "mean": None,
            "std": None,
        }
    low, high, median = bootstrap_ci(arr, np.median, n_boot=n_boot, rng=rng)
    return {
        "n": int(arr.size),
        "median": float(median),
        "ci_low": float(low),
        "ci_high": float(high),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def condition_summary(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarise an empty result set")
    condition = results[0].condition
    n = len(results)
    successes = sum(1 for r in results if r.metrics.success)
    collisions = sum(1 for r in results if r.metrics.collision)
    timeouts = sum(1 for r in results if r.metrics.timeout)
    planning_failures = sum(1 for r in results if r.metrics.planning_failed)
    out_of_bounds = sum(1 for r in results if r.metrics.out_of_bounds)
    success_low, success_high = wilson_ci(successes, n)

    rng = np.random.default_rng(stable_hash(f"summary:{condition}"))
    metrics: dict[str, Any] = {}
    for metric in CONTINUOUS_METRICS:
        metrics[metric] = _metric_stats(
            (getattr(r.metrics, metric) for r in results), rng=rng
        )

    successful = [r for r in results if r.metrics.success]
    success_metrics: dict[str, Any] = {}
    for metric in CONTINUOUS_METRICS:
        success_metrics[metric] = _metric_stats(
            (getattr(r.metrics, metric) for r in successful), rng=rng
        )

    times: list[float] = []
    event_types: list[int] = []
    for r in results:
        times.append(max(0.0, float(r.metrics.time)))
        event_types.append(1 if r.metrics.success else 2)
    risk = competing_risks(times, event_types)
    risk_summary = {
        "median_success_time": risk.median_success_time,
        "success_probability_at_end": (
            float(risk.cif_success[-1]) if risk.cif_success.size else 0.0
        ),
        "failure_probability_at_end": (
            float(risk.cif_failure[-1]) if risk.cif_failure.size else 0.0
        ),
        "n": risk.n,
        "successes": risk.successes,
        "failures": risk.failures,
        "censored": risk.censored,
    }

    return {
        "condition": condition,
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "success_wilson_low": success_low,
        "success_wilson_high": success_high,
        "collisions": collisions,
        "collision_rate": collisions / n,
        "timeouts": timeouts,
        "timeout_rate": timeouts / n,
        "planning_failures": planning_failures,
        "planning_failure_rate": planning_failures / n,
        "out_of_bounds": out_of_bounds,
        "out_of_bounds_rate": out_of_bounds / n,
        "metrics": metrics,
        "success_metrics": success_metrics,
        "competing_risks": risk_summary,
    }


def scenario_results(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    conditions = sorted({r.condition for r in results})
    scenarios = sorted({r.scenario for r in results})
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for scenario in scenarios:
            subset = [
                r
                for r in results
                if r.condition == condition and r.scenario == scenario
            ]
            if not subset:
                continue
            n = len(subset)
            successes = sum(1 for r in subset if r.metrics.success)
            low, high = wilson_ci(successes, n)
            rows.append(
                {
                    "condition": condition,
                    "scenario": scenario,
                    "n": n,
                    "successes": successes,
                    "success_rate": successes / n,
                    "success_wilson_low": low,
                    "success_wilson_high": high,
                    "collisions": sum(1 for r in subset if r.metrics.collision),
                    "timeouts": sum(1 for r in subset if r.metrics.timeout),
                    "out_of_bounds": sum(
                        1 for r in subset if r.metrics.out_of_bounds
                    ),
                    "planning_failures": sum(
                        1 for r in subset if r.metrics.planning_failed
                    ),
                    "median_tracking_rmse_success": _metric_stats(
                        (
                            r.metrics.tracking_rmse
                            for r in subset
                            if r.metrics.success
                        ),
                        rng=np.random.default_rng(
                            stable_hash(f"cell:{condition}:{scenario}")
                        ),
                        n_boot=1000,
                    )["median"],
                }
            )
    return rows


def _paired_keys(
    a: Sequence[EpisodeResult], b: Sequence[EpisodeResult]
) -> list[tuple[str, int]]:
    keys_a = {(r.scenario, r.seed) for r in a}
    keys_b = {(r.scenario, r.seed) for r in b}
    return sorted(keys_a & keys_b)


def pairwise_comparisons(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    conditions = sorted({r.condition for r in results})
    by_condition: dict[str, list[EpisodeResult]] = {
        c: [r for r in results if r.condition == c] for c in conditions
    }
    rows: list[dict[str, Any]] = []

    for ca, cb in itertools.combinations(conditions, 2):
        keys = _paired_keys(by_condition[ca], by_condition[cb])
        if not keys:
            continue
        map_a = {(r.scenario, r.seed): r for r in by_condition[ca]}
        map_b = {(r.scenario, r.seed): r for r in by_condition[cb]}
        pairs = [(map_a[k], map_b[k]) for k in keys]

        b = sum(
            1
            for ra, rb in pairs
            if ra.metrics.success and not rb.metrics.success
        )
        c = sum(
            1
            for ra, rb in pairs
            if not ra.metrics.success and rb.metrics.success
        )
        mcn = mcnemar(b, c, exact=True)
        succ_a = sum(1 for ra, _ in pairs if ra.metrics.success)
        succ_b = sum(1 for _, rb in pairs if rb.metrics.success)
        delta_success = cliffs_delta(
            [1.0 if ra.metrics.success else 0.0 for ra, _ in pairs],
            [1.0 if rb.metrics.success else 0.0 for _, rb in pairs],
        )
        rows.append(
            {
                "metric": "success",
                "condition_a": ca,
                "condition_b": cb,
                "n_pairs": len(pairs),
                "subset": "all_pairs",
                "statistic": float(mcn.statistic),
                "p_value": float(mcn.p_value),
                "method": f"McNemar {mcn.method}",
                "effect_size": float(delta_success),
                "effect_magnitude": cliffs_delta_magnitude(delta_success),
                "median_a": float(succ_a / len(pairs)),
                "median_b": float(succ_b / len(pairs)),
                "median_diff": float((succ_a - succ_b) / len(pairs)),
                "success_rate_a": float(succ_a / len(pairs)),
                "success_rate_b": float(succ_b / len(pairs)),
                "b": b,
                "c": c,
            }
        )

        for metric in PAIRWISE_METRICS:
            if metric in SUCCESS_CONDITIONAL_METRICS:
                metric_pairs = [
                    (ra, rb)
                    for ra, rb in pairs
                    if ra.metrics.success and rb.metrics.success
                ]
                subset = "successful_pairs"
            else:
                metric_pairs = pairs
                subset = "all_pairs"

            va: list[float] = []
            vb: list[float] = []
            for ra, rb in metric_pairs:
                x = float(getattr(ra.metrics, metric))
                y = float(getattr(rb.metrics, metric))
                if math.isfinite(x) and math.isfinite(y):
                    va.append(x)
                    vb.append(y)
            if len(va) < 3:
                continue
            wx = wilcoxon(va, vb)
            delta = cliffs_delta(va, vb)
            rows.append(
                {
                    "metric": metric,
                    "condition_a": ca,
                    "condition_b": cb,
                    "n_pairs": len(va),
                    "subset": subset,
                    "statistic": float(wx.statistic),
                    "p_value": float(wx.p_value),
                    "method": f"Wilcoxon {wx.method}",
                    "effect_size": float(delta),
                    "effect_magnitude": cliffs_delta_magnitude(delta),
                    "median_a": float(np.median(va)),
                    "median_b": float(np.median(vb)),
                    "median_diff": float(np.median(va) - np.median(vb)),
                    "rank_biserial": float(wx.rank_biserial),
                    "b": None,
                    "c": None,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if math.isfinite(float(row["p_value"])):
            grouped.setdefault(str(row["metric"]), []).append(row)
    for metric_rows in grouped.values():
        corrected = holm_correction([float(r["p_value"]) for r in metric_rows])
        for row, holm in zip(metric_rows, corrected):
            row["p_adjusted"] = holm.p_adjusted
            row["reject"] = holm.reject
    for row in rows:
        row.setdefault("p_adjusted", None)
        row.setdefault("reject", False)
    rows.sort(key=lambda r: (r["metric"], float(r["p_value"])))
    return rows


def build_summary(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot build a summary from zero episodes")
    conditions = sorted({r.condition for r in results})
    summaries = [
        condition_summary([r for r in results if r.condition == c])
        for c in conditions
    ]
    return {
        "phase": results[0].phase,
        "n_episodes": len(results),
        "n_conditions": len(conditions),
        "conditions": summaries,
        "scenario_results": scenario_results(results),
        "pairwise": pairwise_comparisons(results),
        "scenarios": sorted({r.scenario for r in results}),
        "seeds": sorted({r.seed for r in results}),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "condition",
        "n",
        "successes",
        "success_rate",
        "success_wilson_low",
        "success_wilson_high",
        "collision_rate",
        "timeout_rate",
        "out_of_bounds_rate",
        "planning_failure_rate",
        "median_executed_length",
        "median_time",
        "median_tracking_rmse",
        "median_heading_rmse",
        "median_max_cross_track",
        "median_control_energy",
        "median_smoothness_cmd",
        "median_smoothness_actuator",
        "median_smoothness",
        "median_min_clearance",
        "median_planned_length",
        "median_executed_length_success",
        "median_time_success",
        "median_tracking_rmse_success",
        "median_heading_rmse_success",
        "median_max_cross_track_success",
        "median_control_energy_success",
        "median_smoothness_cmd_success",
        "median_smoothness_actuator_success",
        "median_smoothness_success",
        "median_min_clearance_success",
        "median_mean_abs_rudder_success",
        "median_rudder_effort_success",
        "median_yaw_rate_rms_success",
        "median_saturated_fraction_success",
        "median_controller_saturated_fraction_success",
        "median_mean_adaptation_effort_success",
        "median_mean_latency_success",
        "median_controller_time_success",
        "median_plant_time_success",
        "cif_median_success_time",
        "cif_success_at_end",
        "cif_failure_at_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for c in summary["conditions"]:
            m = c["metrics"]
            sm = c["success_metrics"]
            writer.writerow(
                {
                    "condition": c["condition"],
                    "n": c["n"],
                    "successes": c["successes"],
                    "success_rate": c["success_rate"],
                    "success_wilson_low": c["success_wilson_low"],
                    "success_wilson_high": c["success_wilson_high"],
                    "collision_rate": c["collision_rate"],
                    "timeout_rate": c["timeout_rate"],
                    "out_of_bounds_rate": c["out_of_bounds_rate"],
                    "planning_failure_rate": c["planning_failure_rate"],
                    "median_executed_length": m["executed_length"]["median"],
                    "median_time": m["time"]["median"],
                    "median_tracking_rmse": m["tracking_rmse"]["median"],
                    "median_heading_rmse": m["heading_rmse"]["median"],
                    "median_max_cross_track": m["max_cross_track"]["median"],
                    "median_control_energy": m["control_energy"]["median"],
                    "median_smoothness_cmd": m["smoothness_cmd"]["median"],
                    "median_smoothness_actuator": m["smoothness_actuator"]["median"],
                    "median_smoothness": m["smoothness_actuator"]["median"],
                    "median_min_clearance": m["min_clearance"]["median"],
                    "median_planned_length": m["planned_length"]["median"],
                    "median_executed_length_success": sm["executed_length"]["median"],
                    "median_time_success": sm["time"]["median"],
                    "median_tracking_rmse_success": sm["tracking_rmse"]["median"],
                    "median_heading_rmse_success": sm["heading_rmse"]["median"],
                    "median_max_cross_track_success": sm["max_cross_track"]["median"],
                    "median_control_energy_success": sm["control_energy"]["median"],
                    "median_smoothness_cmd_success": sm["smoothness_cmd"]["median"],
                    "median_smoothness_actuator_success": sm["smoothness_actuator"]["median"],
                    "median_smoothness_success": sm["smoothness_actuator"]["median"],
                    "median_min_clearance_success": sm["min_clearance"]["median"],
                    "median_mean_abs_rudder_success": sm["mean_abs_rudder"]["median"],
                    "median_rudder_effort_success": sm["rudder_effort"]["median"],
                    "median_yaw_rate_rms_success": sm["yaw_rate_rms"]["median"],
                    "median_saturated_fraction_success": sm["saturated_fraction"]["median"],
                    "median_controller_saturated_fraction_success": sm[
                        "controller_saturated_fraction"
                    ]["median"],
                    "median_mean_adaptation_effort_success": sm[
                        "mean_adaptation_effort"
                    ]["median"],
                    "median_mean_latency_success": sm["mean_latency"]["median"],
                    "median_controller_time_success": sm["controller_time"]["median"],
                    "median_plant_time_success": sm["plant_time"]["median"],
                    "cif_median_success_time": c["competing_risks"][
                        "median_success_time"
                    ],
                    "cif_success_at_end": c["competing_risks"][
                        "success_probability_at_end"
                    ],
                    "cif_failure_at_end": c["competing_risks"][
                        "failure_probability_at_end"
                    ],
                }
            )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# IARA V4 experiment report\n")
    lines.append(
        "Success means reaching the goal tolerance without collision or leaving "
        "the scenario bounds before the timeout.  Intervals are 95% Wilson "
        "(proportions) or percentile bootstrap (medians, 5000 resamples).  "
        "Paired tests use McNemar (success) and Wilcoxon signed-rank "
        "(continuous), with Holm correction within each metric across the "
        "pairwise condition comparisons.  Time to success uses the "
        "Aalen-Johansen cumulative incidence with success and failure as "
        "competing events; failures are never treated as censoring.  "
        "Execution, control-effort and clearance medians are conditional on "
        "success.  Paired execution tests use successful pairs only, while "
        "planning tests use all finite pairs.\n"
    )
    lines.append(f"- Phase: `{summary['phase']}`")
    lines.append(f"- Episodes: {summary['n_episodes']}")
    lines.append(f"- Conditions: {summary['n_conditions']}")
    lines.append(f"- Scenarios: {', '.join(summary['scenarios'])}")
    lines.append(f"- Seeds: {summary['seeds']}\n")

    lines.append("## Condition summary\n")
    lines.append(
        "| Condition | n | Success (95% CI) | Collision | Timeout | "
        "Out of bounds | Median length [m] | Median time [s] | "
        "Median RMSE [m] | Median heading RMSE [rad] | "
        "Median clearance [m] |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in summary["conditions"]:
        sm = c["success_metrics"]
        lines.append(
            "| {cond} | {n} | {succ} ({lo}-{hi}) | {coll} | {tout} | "
            "{oob} | {length} | {time} | {rmse} | {hrmse} | {clear} |".format(
                cond=c["condition"],
                n=c["n"],
                succ=_pct(c["success_rate"]),
                lo=_pct(c["success_wilson_low"]),
                hi=_pct(c["success_wilson_high"]),
                coll=_pct(c["collision_rate"]),
                tout=_pct(c["timeout_rate"]),
                oob=_pct(c["out_of_bounds_rate"]),
                length=_fmt(sm["executed_length"]["median"]),
                time=_fmt(sm["time"]["median"]),
                rmse=_fmt(sm["tracking_rmse"]["median"]),
                hrmse=_fmt(sm["heading_rmse"]["median"]),
                clear=_fmt(sm["min_clearance"]["median"]),
            )
        )

    lines.append("\n## Control effort and actuator use (successful episodes)\n")
    lines.append(
        "| Condition | Mean |delta| [rad] | Rudder effort | Yaw-rate RMS [rad/s] | "
        "Plant saturation | Controller saturation | Adaptation effort | "
        "Mean latency [s] | Controller time [ms] |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in summary["conditions"]:
        sm = c["success_metrics"]
        lines.append(
            "| {cond} | {rud} | {reff} | {yr} | {sat} | {csat} | {ad} | "
            "{lat} | {ct} |".format(
                cond=c["condition"],
                rud=_fmt(sm["mean_abs_rudder"]["median"]),
                reff=_fmt(sm["rudder_effort"]["median"]),
                yr=_fmt(sm["yaw_rate_rms"]["median"]),
                sat=_pct(sm["saturated_fraction"]["median"]),
                csat=_pct(sm["controller_saturated_fraction"]["median"]),
                ad=_fmt(sm["mean_adaptation_effort"]["median"], 4),
                lat=_fmt(sm["mean_latency"]["median"], 4),
                ct=_fmt(
                    None
                    if sm["controller_time"]["median"] is None
                    else 1000.0 * sm["controller_time"]["median"],
                    2,
                ),
            )
        )

    lines.append("\n## Success by scenario\n")
    lines.append(
        "Each cell is `successes / episodes`.  This table exposes the "
        "scenario-specific mechanisms that an aggregate success rate hides.\n"
    )
    scenarios = list(summary["scenarios"])
    lines.append("| Condition | " + " | ".join(scenarios) + " |")
    lines.append("|---|" + "---:|" * len(scenarios))
    by_cell = {
        (row["condition"], row["scenario"]): row
        for row in summary.get("scenario_results", [])
    }
    for c in summary["conditions"]:
        cells = []
        for scenario in scenarios:
            row = by_cell.get((c["condition"], scenario))
            cells.append(
                f"{row['successes']}/{row['n']}" if row is not None else "n/a"
            )
        lines.append(f"| {c['condition']} | " + " | ".join(cells) + " |")

    lines.append("\n## Paired comparisons\n")
    lines.append(
        "Holm correction is applied within each metric; `p (Holm)` is "
        "comparable only within a metric block.  Success comparisons use all "
        "paired episodes; execution comparisons use pairs in which both "
        "conditions succeeded.\n"
    )
    lines.append(
        "| Metric | A | B | Subset | n | p | p (Holm) | Effect (Cliff) | "
        "Magnitude | Reject H0 |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")
    for row in summary["pairwise"]:
        lines.append(
            "| {metric} | {a} | {b} | {subset} | {n} | {p} | {padj} | "
            "{eff} | {mag} | {rej} |".format(
                metric=row["metric"],
                a=row["condition_a"],
                b=row["condition_b"],
                subset=row.get("subset", "all_pairs"),
                n=row["n_pairs"],
                p=_fmt(row.get("p_value"), 4),
                padj=_fmt(row.get("p_adjusted"), 4),
                eff=_fmt(row.get("effect_size"), 3),
                mag=row.get("effect_magnitude", "n/a"),
                rej="yes" if row.get("reject") else "no",
            )
        )

    lines.append("\n## Time to success (Aalen-Johansen competing risks)\n")
    lines.append(
        "| Condition | Median success time [s] | Success incidence at end | "
        "Failure incidence at end | Successes / n |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for c in summary["conditions"]:
        risk = c["competing_risks"]
        lines.append(
            f"| {c['condition']} | {_fmt(risk['median_success_time'])} | "
            f"{_pct(risk['success_probability_at_end'])} | "
            f"{_pct(risk['failure_probability_at_end'])} | "
            f"{risk['successes']} / {risk['n']} |"
        )

    lines.append(
        "\n## Reading the results\n\n"
        "The global planner sees only the mapped obstacles.  In scenarios with "
        "`hidden_obstacles`, the plant, clearance metric and range sensor see "
        "the unmapped obstacle, so the VFH+ conditions test reactive avoidance "
        "rather than a different global map.  The planning disc radius is the "
        "boat's half-diagonal plus a 0.02 m safety margin; the evaluation uses "
        "the true oriented-rectangle footprint.  `min_clearance` is `n/a` for "
        "a condition whose successful episodes all occur in open water; when "
        "only some successful episodes have obstacles, the reported median "
        "uses the finite subset and the per-metric `n` in `summary.json` "
        "records its size.  `controller_time` and `plant_time` are wall-clock "
        "measurements and are therefore descriptive, not a real-time guarantee.  "
        "The dynamic plant parameters are nominal simulation values, not "
        "identified IARA data; no result in this report is a physical "
        "validation of the vessel.\n"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
