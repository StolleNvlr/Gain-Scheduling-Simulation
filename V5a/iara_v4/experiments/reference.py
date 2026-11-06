"""Auditable reconstruction of ``projeto_barco.m`` and numerical certificates.

This module is deliberately independent of the V4 operational controllers.  It
answers three questions:

1. What continuous model and what ZOH/ARX coefficients does the MATLAB script
   actually contain?
2. Do the hard-coded branch coefficients match the model that the branch label
   claims?  (They do not: the labels are shifted by one operating point.)
3. Do the corrected gain-scheduled vertices admit a common quadratic Lyapunov
   function at the declared controller period?

Nothing here is a proof about the real IARA.  The results are algebraic and
numerical verification of the stated models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete, ss2tf

from ..control.gain_scheduler import GainScheduler
from ..dynamics.model import VehicleParameters

AN = 1.6
BN = 0.2
UN = 1.0
K_NOMINAL = 30.0


@dataclass(frozen=True, slots=True)
class MatlabModel:
    q: float
    a: float
    b: float
    k: float
    td: float
    numerator: tuple[float, float]
    denominator: tuple[float, float, float]
    poles: tuple[complex, complex]
    wn: float
    zeta: float
    dc_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "a": self.a,
            "b": self.b,
            "k": self.k,
            "td": self.td,
            "numerator": list(self.numerator),
            "denominator": list(self.denominator),
            "poles": [[p.real, p.imag] for p in self.poles],
            "wn": self.wn,
            "zeta": self.zeta,
            "dc_gain": self.dc_gain,
        }


@dataclass(frozen=True, slots=True)
class ArxModel:
    ts: float
    num: tuple[float, float, float]
    den: tuple[float, float, float]
    c1: float
    c2: float
    d0: float
    d1: float
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ts": self.ts,
            "a1": self.a1,
            "a2": self.a2,
            "b0": self.b0,
            "b1": self.b1,
            "b2": self.b2,
            "c1": self.c1,
            "c2": self.c2,
            "d0": self.d0,
            "d1": self.d1,
        }


@dataclass(frozen=True, slots=True)
class MatlabBranch:
    speed_label: float
    c1: float
    c2: float
    d0: float
    d1: float


def matlab_design(q: float, k_nominal: float = K_NOMINAL) -> MatlabModel:
    """Reconstruct one local model exactly as written in the MATLAB script."""

    if q <= 0.0:
        raise ValueError("q must be positive")
    a = AN * q
    b = BN * q * q
    k = k_nominal / (q * q)
    # The script writes 2.4/q, not 2.4.  This is the source of the
    # speed-dependent damping ratio documented in the audit.
    td = (2.4 / q * math.sqrt(b * k) - a) / (b * k)
    numerator = (k * b * td, k * b)
    denominator = (1.0, a + k * b * td, k * b)
    poles = np.roots(np.array(denominator, dtype=float))
    wn = math.sqrt(k * b)
    zeta = (a + k * b * td) / (2.0 * wn)
    dc_gain = numerator[1] / denominator[2] if denominator[2] != 0.0 else math.inf
    return MatlabModel(
        q=float(q),
        a=float(a),
        b=float(b),
        k=float(k),
        td=float(td),
        numerator=numerator,
        denominator=denominator,
        poles=(complex(poles[0]), complex(poles[1])),
        wn=float(wn),
        zeta=float(zeta),
        dc_gain=float(dc_gain),
    )


def zoh_arx(model: MatlabModel, ts: float = 1.0) -> ArxModel:
    """Discretise the continuous closed loop with ZOH and return ARX coefficients."""

    b1, b0 = model.numerator
    c1, c0 = model.denominator[1], model.denominator[2]
    A = np.array([[0.0, 1.0], [-c0, -c1]], dtype=float)
    B = np.array([[0.0], [1.0]], dtype=float)
    C = np.array([[b0, b1]], dtype=float)
    D = np.array([[0.0]], dtype=float)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), float(ts), method="zoh")
    num, den = ss2tf(Ad, Bd, Cd, Dd)
    num = np.asarray(num, dtype=float).reshape(-1)
    den = np.asarray(den, dtype=float).reshape(-1)
    # ss2tf returns a common scaling factor.  Divide *both* polynomials by the
    # same denominator leading coefficient so that the ARX convention
    # ``y[k] + a1 y[k-1] + a2 y[k-2] = b0 u[k] + ...`` is preserved exactly.
    scale = float(den[0])
    if scale == 0.0:
        raise ValueError("degenerate discrete denominator")
    den = den / scale
    num = num / scale
    if num.size < 3:
        num = np.pad(num, (3 - num.size, 0))
    a1, a2 = float(den[1]), float(den[2])
    b0d, b1d, b2d = float(num[0]), float(num[1]), float(num[2])
    return ArxModel(
        ts=float(ts),
        num=(b0d, b1d, b2d),
        den=(1.0, a1, a2),
        c1=-a1,
        c2=-a2,
        d0=b1d,
        d1=b2d,
        b0=b0d,
        b1=b1d,
        b2=b2d,
        a1=a1,
        a2=a2,
    )


MATLAB_BRANCHES: tuple[MatlabBranch, ...] = (
    MatlabBranch(0.8, 0.279, -0.002798, 0.9669, -0.2431),
    MatlabBranch(0.1, 0.394, -0.0006436, 0.9738, -0.3671),
    MatlabBranch(1.0, 0.9028, -1.002e-16, 0.9991, -0.9019),
)


def branch_coefficients(branch: MatlabBranch) -> tuple[float, float, float, float]:
    return (branch.c1, branch.c2, branch.d0, branch.d1)


def coefficient_error(branch: MatlabBranch, arx: ArxModel) -> dict[str, float]:
    values = np.array(branch_coefficients(branch), dtype=float)
    reference = np.array((arx.c1, arx.c2, arx.d0, arx.d1), dtype=float)
    diff = np.abs(values - reference)
    scale = np.maximum(1.0, np.abs(reference))
    return {
        "max_abs": float(np.max(diff)),
        "max_rel": float(np.max(diff / scale)),
        "c1": float(diff[0]),
        "c2": float(diff[1]),
        "d0": float(diff[2]),
        "d1": float(diff[3]),
    }


def identify_branch_mapping(
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Match every hard-coded branch to the ZOH model that produced it."""

    models = {q: matlab_design(q) for q in (0.1, 0.8, 1.0)}
    arx = {q: zoh_arx(model, 1.0) for q, model in models.items()}
    mapping: list[dict[str, Any]] = []
    for branch in MATLAB_BRANCHES:
        candidates = []
        for q, model_arx in arx.items():
            err = coefficient_error(branch, model_arx)
            candidates.append((err["max_abs"], q, err))
        best = min(candidates, key=lambda item: item[0])
        mapping.append(
            {
                "branch_speed": branch.speed_label,
                "best_model_q": best[1],
                "max_abs_error": best[0],
                "within_tolerance": bool(best[0] <= tolerance),
                "coefficients": {
                    "c1": branch.c1,
                    "c2": branch.c2,
                    "d0": branch.d0,
                    "d1": branch.d1,
                },
                "model_coefficients": arx[best[1]].to_dict(),
            }
        )
    return {
        "models": {f"{q:g}": model.to_dict() for q, model in models.items()},
        "arx": {f"{q:g}": arx[q].to_dict() for q in arx},
        "branches": mapping,
        "label_shift_detected": any(
            abs(item["branch_speed"] - item["best_model_q"]) > 1e-9 for item in mapping
        ),
    }


def simulate_arx(
    arx: ArxModel,
    inputs: Sequence[float],
    *,
    initial_outputs: tuple[float, float] = (0.0, 0.0),
    initial_inputs: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Simulate one ARX model with the MATLAB loop convention.

    The loop stores ``i1=u[k-1]``, ``i2=u[k-2]`` and computes
    ``y[k]=c1 y[k-1]+c2 y[k-2]+d0 i1+d1 i2``.  This is the one-step-delayed
    form produced by a strictly proper ZOH model (``b0=0``).
    """

    s1, s2 = initial_outputs
    i1, i2 = initial_inputs
    out: list[float] = []
    for value in inputs:
        s = arx.c1 * s1 + arx.c2 * s2 + arx.d0 * i1 + arx.d1 * i2
        out.append(float(s))
        i2, i1 = i1, float(value)
        s2, s1 = s1, s
    return np.asarray(out, dtype=float)


def simulate_matlab_branches(
    heading_refs: Sequence[float], speeds: Sequence[float]
) -> np.ndarray:
    """Reproduce the hard-coded branch selection exactly.

    The MATLAB script keeps a *single* set of registers ``s1, s2, i1, i2`` and
    only swaps the coefficients when the speed branch changes.  The ARX state is
    therefore shared across branch changes; keeping one state per branch would
    silently reset the filter and is not what the script does.

    The script has no ``else``.  A speed that is not exactly one of the three
    labels leaves ``s_ang`` undefined.  This function raises a ``ValueError``
    instead of silently holding the previous value, because the latter would
    fabricate behaviour that the script does not contain.
    """

    if len(heading_refs) != len(speeds):
        raise ValueError("heading_refs and speeds must have the same length")
    by_speed = {branch.speed_label: branch for branch in MATLAB_BRANCHES}
    s1 = s2 = i1 = i2 = 0.0
    outputs: list[float] = []
    for ref, speed in zip(heading_refs, speeds):
        branch = by_speed.get(float(speed))
        if branch is None:
            raise ValueError(
                f"MATLAB script has no branch for speed {speed!r}; "
                "the original code leaves s_ang undefined"
            )
        s = branch.c1 * s1 + branch.c2 * s2 + branch.d0 * i1 + branch.d1 * i2
        outputs.append(float(s))
        i2, i1 = i1, float(ref)
        s2, s1 = s1, s
    return np.asarray(outputs, dtype=float)


def simulate_matlab_mission() -> dict[str, Any]:
    """Run the waypoint mission embedded in ``projeto_barco.m``.

    The plant is the kinematic update printed in the script:
    ``x <- x + v dt cos(s_ang)``, ``y <- y + v dt sin(s_ang)``.  This is not a
    3-DOF plant; it is the legacy reference trajectory.
    """

    dt = 0.01
    x, y = 0.0, 0.0
    xr, yr = 13.0, 0.0
    v = 0.8
    i1 = i2 = s1 = s2 = 0.0
    by_speed = {branch.speed_label: branch for branch in MATLAB_BRANCHES}
    trajectory: list[list[float]] = []
    for _ in range(20000):
        trajectory.append([x, y, v, s1])
        if x >= 40.0:
            break
        i_ang = math.atan2(yr - y, xr - x)
        branch = by_speed.get(float(v))
        if branch is None:
            break
        s_ang = branch.c1 * s1 + branch.c2 * s2 + branch.d0 * i1 + branch.d1 * i2
        x += v * dt * math.cos(s_ang)
        y += v * dt * math.sin(s_ang)
        i2, i1 = i1, i_ang
        s2, s1 = s1, s_ang
        if 13.0 <= x < 20.0:
            xr, yr, v = 20.0, 10.0, 0.8
        elif 20.0 <= x < 30.0:
            xr, yr, v = 30.0, -7.0, 0.1
        elif 30.0 <= x < 40.0:
            xr, yr, v = 40.0, y, 1.0
    return {
        "trajectory": trajectory,
        "final_x": x,
        "final_y": y,
        "steps": len(trajectory),
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Common quadratic Lyapunov function (numerical certificate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CQLFResult:
    success: bool
    P: np.ndarray
    q_scale: float
    residual: float
    min_eigenvalue_P: float
    condition_number_P: float
    margins: tuple[float, ...]
    max_spectral_radius: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "P": self.P.tolist(),
            "q_scale": self.q_scale,
            "residual": self.residual,
            "min_eigenvalue_P": self.min_eigenvalue_P,
            "condition_number_P": self.condition_number_P,
            "margins": list(self.margins),
            "max_spectral_radius": self.max_spectral_radius,
            "note": self.note,
        }


def _sym_index(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i, n)]


def _sym_from_vector(values: np.ndarray, n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=float)
    idx = 0
    for i, j in _sym_index(n):
        P[i, j] = values[idx]
        P[j, i] = values[idx]
        idx += 1
    return P


def _sym_equations(A: np.ndarray, q_scale: float) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    pairs = _sym_index(n)
    columns = []
    for i, j in pairs:
        E = np.zeros((n, n), dtype=float)
        E[i, j] = 1.0
        E[j, i] = 1.0
        # A^T P A - P with P=E; the symmetric part is what matters.
        M = A.T @ E @ A - E
        columns.append(np.array([M[p, q] for p, q in pairs], dtype=float))
    lhs = np.column_stack(columns)
    rhs = -q_scale * np.array(
        [1.0 if p == q else 0.0 for p, q in pairs], dtype=float
    )
    return lhs, rhs


def common_lyapunov(
    matrices: Iterable[np.ndarray],
    *,
    q_scales: Sequence[float] = (1.0, 0.5, 0.1, 0.05, 0.01),
    tolerance: float = 1e-10,
) -> CQLFResult:
    """Find ``P>0`` with ``A_i^T P A_i - P <= -q I`` by linear least squares.

    For a small collection of matrices and a symmetric ``P``, the equalities
    ``A_i^T P A_i - P = -q I`` are linear in the entries of ``P``.  We solve
    them in the least-squares sense for several ``q`` and return the first
    candidate that is positive definite and has a strictly negative margin.
    This is a numerical certificate; it is not a formal SDP proof.
    """

    mats = [np.asarray(A, dtype=float) for A in matrices]
    if not mats:
        raise ValueError("at least one matrix is required")
    n = mats[0].shape[0]
    if any(A.shape != (n, n) for A in mats):
        raise ValueError("all matrices must have the same square shape")

    best: CQLFResult | None = None
    for q_scale in q_scales:
        blocks = []
        rhs_blocks = []
        for A in mats:
            lhs_i, rhs_i = _sym_equations(A, q_scale)
            blocks.append(lhs_i)
            rhs_blocks.append(rhs_i)
        lhs = np.vstack(blocks)
        rhs = np.concatenate(rhs_blocks)
        solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        P = _sym_from_vector(solution, n)
        P = 0.5 * (P + P.T)
        residual = float(np.linalg.norm(lhs @ solution - rhs))
        eig_P = np.linalg.eigvalsh(P)
        margins = []
        for A in mats:
            M = A.T @ P @ A - P
            margins.append(float(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))))
        max_rho = max(float(max(abs(np.linalg.eigvals(A)))) for A in mats)
        success = bool(
            residual <= 1e-8
            and float(np.min(eig_P)) > tolerance
            and max(margins) < -tolerance
        )
        candidate = CQLFResult(
            success=success,
            P=P,
            q_scale=float(q_scale),
            residual=residual,
            min_eigenvalue_P=float(np.min(eig_P)),
            condition_number_P=float(np.linalg.cond(P)),
            margins=tuple(margins),
            max_spectral_radius=max_rho,
            note=(
                "numerical least-squares certificate"
                if success
                else "candidate failed positive-definiteness or margin check"
            ),
        )
        if success:
            return candidate
        if best is None or candidate.residual < best.residual:
            best = candidate
    assert best is not None
    return best


def yaw_closed_loop_matrices(
    vehicle: VehicleParameters,
    points: Sequence[tuple[float, float, float]],
    *,
    ts: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Discrete closed-loop yaw matrices for ``(speed, kp, kd)`` points."""

    matrices: list[np.ndarray] = []
    speeds: list[float] = []
    for speed, kp, kd in points:
        a = vehicle.yaw_a(speed)
        b = vehicle.yaw_b(speed)
        A = np.array([[0.0, 1.0], [-b * kp, -a - b * kd]], dtype=float)
        matrices.append(expm(A * float(ts)))
        speeds.append(float(speed))
    return matrices, speeds


def scheduler_grid_points(
    scheduler: GainScheduler, speeds: Sequence[float]
) -> list[tuple[float, float, float]]:
    return [
        (float(speed), scheduler.gains(speed).kp, scheduler.gains(speed).kd)
        for speed in speeds
    ]


def analyze_scheduler(
    vehicle: VehicleParameters,
    scheduler: GainScheduler,
    *,
    ts: float = 0.05,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    if grid is None:
        grid = np.linspace(scheduler.config.speed_min, scheduler.config.speed_max, 31)
    points = scheduler_grid_points(scheduler, grid)
    matrices, speeds = yaw_closed_loop_matrices(vehicle, points, ts=ts)
    cqlf = common_lyapunov(matrices)
    rho = [float(max(abs(np.linalg.eigvals(A)))) for A in matrices]
    return {
        "ts": float(ts),
        "grid": speeds,
        "spectral_radius": rho,
        "max_spectral_radius": max(rho),
        "min_stability_margin": 1.0 - max(rho),
        "cqlf": cqlf.to_dict(),
        "points": [
            {"speed": s, "kp": kp, "kd": kd}
            for s, kp, kd in points
        ],
    }


def audit_report() -> dict[str, Any]:
    mapping = identify_branch_mapping()
    models = [matlab_design(q) for q in (0.1, 0.8, 1.0)]
    arx = {f"{m.q:g}": zoh_arx(m, 1.0).to_dict() for m in models}
    zeta_rows = [
        {
            "q": m.q,
            "a": m.a,
            "b": m.b,
            "k": m.k,
            "td": m.td,
            "wn": m.wn,
            "zeta": m.zeta,
            "poles": [[p.real, p.imag] for p in m.poles],
        }
        for m in models
    ]
    temporal = temporal_interpretations()
    discrepancies = {
        "zeta_constant_claim": (
            "The complement document claims zeta=1.2 for every regime. "
            "The script uses 2.4/q, so the actual values are speed dependent."
        ),
        "td_table_claim": (
            "The Td values in the complement table are compared against the "
            "values reconstructed from the script."
        ),
        "branch_label_shift": (
            "The hard-coded branch labels are shifted relative to their "
            "reconstructed ZOH models."
        ),
        "ts_mismatch": (
            "The script discretises at 1 s but applies coefficients every "
            "0.01 s; the three temporal readings are reported separately."
        ),
    }
    branches = mapping["branches"]
    gates = {
        "label_shift_detected": bool(mapping["label_shift_detected"]),
        "all_branches_within_tolerance": bool(
            all(row["within_tolerance"] for row in branches)
        ),
        "three_temporal_interpretations": len(temporal["rows"]) == 3,
        "zeta_speed_dependent_verified": bool(
            abs(zeta_rows[0]["zeta"] - 12.0) < 1e-6
            and abs(zeta_rows[2]["zeta"] - 1.2) < 1e-6
        ),
        "ts_mismatch_documented": "ts_mismatch" in discrepancies,
    }
    return {
        "constants": {"an": AN, "bn": BN, "un": UN, "K": K_NOMINAL},
        "models": zeta_rows,
        "arx_ts1": arx,
        "branch_mapping": mapping,
        "matlab_mission": simulate_matlab_mission(),
        "temporal_interpretations": temporal,
        "documentation_discrepancies": discrepancies,
        "gates": gates,
    }


def _settling_time(
    time: np.ndarray, output: np.ndarray, target: float, band: float = 0.02
) -> float | None:
    """Last time the response leaves a relative band around the target."""

    if target == 0.0:
        return None
    error = np.abs(output - target) / abs(target)
    outside = np.nonzero(error > band)[0]
    if outside.size == 0:
        return float(time[0])
    if outside[-1] + 1 >= time.size:
        return None
    return float(time[outside[-1] + 1])


def temporal_interpretations() -> dict[str, Any]:
    """Compare the three defensible readings of the MATLAB timing.

    The script obtains a discrete model with ``c2d(..., 1)`` but applies its
    coefficients every ``0.01`` s.  Three interpretations are reported:

    * ``one_hz_zoh``: the 1 s model is applied once per second and held;
    * ``rediscretised_100_hz``: the continuous model is discretised at 0.01 s;
    * ``direct_1hz_at_100hz``: the 1 s coefficients are applied at every 0.01 s
      step (the literal loop, not a valid ZOH implementation).

    The input is a unit step and the output is the closed-loop heading.  These
    are model-level comparisons, not boat experiments.
    """

    q = 1.0
    model = matlab_design(q)
    arx_1s = zoh_arx(model, 1.0)
    arx_100hz = zoh_arx(model, 0.01)

    # 1 Hz application with zero-order hold over 100 plant steps.
    n_slow = 20
    slow_values = simulate_arx(arx_1s, np.ones(n_slow))
    slow_time = np.arange(n_slow, dtype=float) + 1.0
    slow_output = np.repeat(slow_values, 100)
    slow_time_dense = np.arange(slow_output.size, dtype=float) * 0.01 + 0.01

    n_fast = 2000
    fast_time = np.arange(n_fast, dtype=float) * 0.01 + 0.01
    fast_output = simulate_arx(arx_100hz, np.ones(n_fast))
    literal_output = simulate_arx(arx_1s, np.ones(n_fast))

    target = arx_100hz.d0 * 0.0  # placeholder; steady state is estimated below
    # Use the last value of the longest horizon as the steady-state estimate.
    steady = float(fast_output[-1])
    rows = []
    for label, time, output in (
        ("one_hz_zoh", slow_time_dense, slow_output),
        ("rediscretised_100_hz", fast_time, fast_output),
        ("direct_1hz_at_100hz", fast_time, literal_output),
    ):
        rows.append(
            {
                "interpretation": label,
                "final_value": float(output[-1]),
                "peak": float(np.max(output)),
                "overshoot_fraction": float(
                    (np.max(output) - steady) / steady if steady else math.nan
                ),
                "settling_time_2pct": _settling_time(time, output, steady),
                "rms_after_2s": float(
                    np.sqrt(np.mean(output[time >= 2.0] ** 2))
                ),
            }
        )

    return {
        "q": q,
        "input": "unit step",
        "steady_state_reference": steady,
        "arx_1s": arx_1s.to_dict(),
        "arx_100hz": arx_100hz.to_dict(),
        "rows": rows,
    }
