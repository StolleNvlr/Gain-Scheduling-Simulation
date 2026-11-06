"""Statistical tests used by the IEEE experiment protocol.

All functions are deterministic given their inputs and an explicit RNG for the
bootstrap.  The choices follow the reporting recommendations for robotics
experiments: Wilson intervals for proportions, bootstrap intervals for medians,
McNemar for paired success/failure, Wilcoxon for paired continuous outcomes,
Holm for multiplicity, Cliff's delta for ordinal effect size, and the
Aalen-Johansen cumulative incidence for time-to-success with competing risks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class McNemarResult:
    b: int
    c: int
    statistic: float
    p_value: float
    method: str


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n: int
    median_difference: float
    rank_biserial: float
    method: str


@dataclass(frozen=True, slots=True)
class HolmResult:
    index: int
    p_value: float
    p_adjusted: float
    reject: bool


@dataclass(frozen=True, slots=True)
class CompetingRiskResult:
    """Aalen-Johansen cumulative incidence for two competing events."""

    times: np.ndarray
    cif_success: np.ndarray
    cif_failure: np.ndarray
    at_risk: np.ndarray
    median_success_time: float | None
    n: int
    successes: int
    failures: int
    censored: int


def wilson_ci(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209-212.
    """

    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n == 0:
        return (math.nan, math.nan)
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denom
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.median,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a statistic using vectorized draws."""

    values = np.asarray(list(data), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = rng or np.random.default_rng(0)
    estimate = float(statistic(values))
    n = values.size
    samples = values[rng.integers(0, n, size=(n_boot, n))]
    if statistic is np.median:
        boots = np.median(samples, axis=1)
    elif statistic is np.mean:
        boots = samples.mean(axis=1)
    else:
        boots = np.array([statistic(row) for row in samples])
    alpha = 1.0 - confidence
    low, high = np.percentile(
        boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return (float(low), float(high), estimate)


def mcnemar(b: int, c: int, *, exact: bool = True) -> McNemarResult:
    """McNemar test for paired binary outcomes.

    ``b`` is the number of pairs where condition A succeeds and B fails; ``c``
    is the opposite discordant count.  The exact test is the two-sided binomial
    test with p = 0.5 on the discordant pairs.
    """

    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "exact" if exact else "chi2")
    if exact:
        p = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), n, 0.5)))
        return McNemarResult(b, c, float(min(b, c)), p, "exact")
    statistic = (abs(b - c) - 1.0) ** 2 / n
    p = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(b, c, float(statistic), p, "chi2-continuity")


def wilcoxon(
    x: Sequence[float],
    y: Sequence[float] | None = None,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test with rank-biserial effect size."""

    a = np.asarray(list(x), dtype=float)
    if y is None:
        b = np.zeros_like(a)
        method = "one-sample"
    else:
        b = np.asarray(list(y), dtype=float)
        method = "paired"
    if a.shape != b.shape:
        raise ValueError("x and y must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return WilcoxonResult(math.nan, math.nan, 0, math.nan, math.nan, method)
    diffs = a - b
    nonzero = diffs[np.abs(diffs) > 1e-12]
    median_diff = float(np.median(diffs))
    if nonzero.size == 0:
        return WilcoxonResult(0.0, 1.0, 0, median_diff, 0.0, method)

    try:
        result = stats.wilcoxon(
            a, b, alternative=alternative, zero_method="wilcox", correction=False
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError:
        statistic, p_value = 0.0, 1.0

    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    rank_biserial = (positive - negative) / total if total > 0.0 else 0.0
    return WilcoxonResult(
        statistic=statistic,
        p_value=p_value,
        n=int(nonzero.size),
        median_difference=median_diff,
        rank_biserial=float(rank_biserial),
        method=method,
    )


def holm_correction(
    p_values: Sequence[float], alpha: float = 0.05
) -> list[HolmResult]:
    """Holm-Bonferroni step-down correction, returned in input order.

    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2), 65-70.
    """

    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return [
        HolmResult(
            index=int(i),
            p_value=float(p[i]),
            p_adjusted=float(adjusted[i]),
            reject=bool(adjusted[i] <= alpha),
        )
        for i in range(m)
    ]


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta ordinal effect size in [-1, 1].

    Positive values mean ``x`` tends to be larger than ``y``.  The magnitude
    thresholds are the usual |d| < 0.147 negligible, < 0.33 small, < 0.474
    medium, otherwise large.
    """

    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return math.nan
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def cliffs_delta_magnitude(delta: float) -> str:
    if not math.isfinite(delta):
        return "undefined"
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def competing_risks(
    times: Sequence[float],
    event_types: Sequence[int],
    *,
    success_code: int = 1,
    failure_code: int = 2,
) -> CompetingRiskResult:
    """Aalen-Johansen cumulative incidence with competing events.

    ``event_types[i] = success_code`` means the episode reached the goal,
    ``failure_code`` means it collided, timed out or failed to plan, and ``0``
    means right-censored.  Treating failures as censoring (the usual mistake
    when reusing Kaplan-Meier) overestimates the chance of eventual success;
    the cumulative-incidence estimator keeps the two risks separate.

    Reference: Aalen, O. O. & Johansen, S. (1978). An empirical transition
    matrix for non-homogeneous Markov chains based on censored observations.
    *Scandinavian Journal of Statistics*, 5(3), 141-150.
    """

    t = np.asarray(list(times), dtype=float)
    e = np.asarray(list(event_types), dtype=int)
    if t.shape != e.shape:
        raise ValueError("times and event_types must have the same length")
    mask = np.isfinite(t) & (t >= 0.0)
    t, e = t[mask], e[mask]
    if t.size == 0:
        empty = np.array([], dtype=float)
        return CompetingRiskResult(
            empty, empty, empty, empty, None, 0, 0, 0, 0
        )
    valid = {0, success_code, failure_code}
    unknown = sorted(set(int(v) for v in e) - valid)
    if unknown:
        raise ValueError(f"unknown event codes: {unknown}")

    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    unique_times = np.unique(t)

    survival = 1.0
    cif_success = 0.0
    cif_failure = 0.0
    out_t: list[float] = []
    out_success: list[float] = []
    out_failure: list[float] = []
    out_risk: list[int] = []
    median_success: float | None = None

    for time in unique_times:
        at_risk = int(np.sum(t >= time))
        d_success = int(np.sum((t == time) & (e == success_code)))
        d_failure = int(np.sum((t == time) & (e == failure_code)))
        d_any = d_success + d_failure
        if at_risk > 0 and d_any > 0:
            cif_success += survival * d_success / at_risk
            cif_failure += survival * d_failure / at_risk
            survival *= 1.0 - d_any / at_risk
        if median_success is None and cif_success >= 0.5:
            median_success = float(time)
        out_t.append(float(time))
        # Floating-point accumulation can leave 1 + 1e-16; clamp for reporting.
        out_success.append(float(min(1.0, max(0.0, cif_success))))
        out_failure.append(float(min(1.0, max(0.0, cif_failure))))
        out_risk.append(at_risk)

    return CompetingRiskResult(
        times=np.asarray(out_t, dtype=float),
        cif_success=np.asarray(out_success, dtype=float),
        cif_failure=np.asarray(out_failure, dtype=float),
        at_risk=np.asarray(out_risk, dtype=int),
        median_success_time=median_success,
        n=int(t.size),
        successes=int(np.sum(e == success_code)),
        failures=int(np.sum(e == failure_code)),
        censored=int(np.sum(e == 0)),
    )
