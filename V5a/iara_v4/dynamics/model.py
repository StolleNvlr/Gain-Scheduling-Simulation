"""3-DOF manoeuvring model and nominal parameters for the IARA V4 plant.

The model is a deliberately minimal Fossen-type surface-vessel model:

    M nu_dot + C(nu) nu + D(nu) nu = tau + tau_ext
    eta_dot = R(psi) nu

with ``nu = [u, v, r]^T`` and ``eta = [x, y, psi]^T``.  The parameters below
are *nominal simulation values*, not identified IARA parameters.  They were
chosen so that the linearised yaw subsystem has the same structure as the
local models used in ``projeto_barco.m``:

    psi_ddot + a(q) psi_dot = b(q) delta,       q = u / u_n
    a(q) = a_n q,                               b(q) = b_n q^2

The sway and yaw equations are decoupled by setting ``m_u = m_v`` and by
making the yaw damping proportional to ``a(q)``.  The yaw effectiveness is

    b(q) = b_0 + b_n q^2,

where the constant term ``b_0`` represents the differential-thrust yaw
authority available at low speed and ``b_n q^2`` represents the speed-dependent
rudder/foil term.  This is a deliberate V4 modelling hypothesis: the legacy
MATLAB model is rudder-only (``b_0 = 0``, ``b_n = 0.2``).  The constant term is
needed because the IARA is a differential-drive USV and because a rudder-only
model has a minimum steady turn radius of ``u_n / ((b_n/a_n) delta_max)`` that
is larger than the supplied obstacle courses.  The parameters are nominal
simulation values, not identified IARA values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VehicleParameters:
    """Nominal 3-DOF parameters in SI units."""

    m_u: float = 6.0
    m_v: float = 6.0
    m_r: float = 0.10

    a_n: float = 1.6
    b_0: float = 0.8
    b_n: float = 3.0
    u_nominal: float = 1.0
    u_min_effective: float = 0.02

    d_u_linear: float = 4.0
    d_u_quadratic: float = 2.0
    d_v_linear: float = 12.0
    d_v_quadratic: float = 4.0
    d_r_quadratic: float = 0.02

    rudder_max: float = 0.65

    def __post_init__(self) -> None:
        if min(self.m_u, self.m_v, self.m_r) <= 0.0:
            raise ValueError("mass/inertia parameters must be positive")
        if self.a_n <= 0.0 or self.b_n <= 0.0:
            raise ValueError("yaw model coefficients must be positive")
        if self.b_0 < 0.0:
            raise ValueError("low-speed yaw authority b_0 must be non-negative")
        if self.u_nominal <= 0.0 or self.u_min_effective <= 0.0:
            raise ValueError("speed scales must be positive")
        if min(self.d_u_linear, self.d_u_quadratic) < 0.0:
            raise ValueError("surge damping must be non-negative")
        if min(self.d_v_linear, self.d_v_quadratic) < 0.0:
            raise ValueError("sway damping must be non-negative")
        if self.d_r_quadratic < 0.0:
            raise ValueError("yaw quadratic damping must be non-negative")
        if not 0.0 < self.rudder_max < math.pi / 2.0:
            raise ValueError("rudder_max must be in (0, pi/2)")

    # -- scheduled yaw coefficients ----------------------------------------

    def speed_ratio(self, u: float) -> float:
        return max(abs(float(u)), self.u_min_effective) / self.u_nominal

    def yaw_a(self, u: float) -> float:
        """Linear yaw damping coefficient ``a(q) = a_n q`` [1/s]."""

        return self.a_n * self.speed_ratio(u)

    def yaw_b(self, u: float) -> float:
        """Yaw effectiveness ``b(q) = b_0 + b_n q^2`` [1/(s^2 rad)].

        ``b_0`` is the low-speed differential-thrust term; ``b_n q^2`` is the
        speed-dependent rudder/foil term.  Setting ``b_0 = 0`` recovers the
        legacy rudder-only model.
        """

        q = self.speed_ratio(u)
        return self.b_0 + self.b_n * q * q

    def rudder_gain(self, u: float) -> float:
        """Map rudder angle [rad] to yaw moment [N m]."""

        return self.m_r * self.yaw_b(u)

    def yaw_damping_linear(self, u: float) -> float:
        """Yaw damping [N m s] that reproduces ``a(q)``."""

        return self.m_r * self.yaw_a(u)

    # -- matrices -----------------------------------------------------------

    @property
    def mass_matrix(self) -> np.ndarray:
        return np.diag([self.m_u, self.m_v, self.m_r])

    def coriolis(self, nu: np.ndarray) -> np.ndarray:
        u, v, _ = (float(x) for x in nu)
        return np.array(
            [
                [0.0, 0.0, -self.m_v * v],
                [0.0, 0.0, self.m_u * u],
                [self.m_v * v, -self.m_u * u, 0.0],
            ],
            dtype=float,
        )

    def damping(self, nu: np.ndarray) -> np.ndarray:
        u, v, r = (float(x) for x in nu)
        d_u = self.d_u_linear + self.d_u_quadratic * abs(u)
        d_v = self.d_v_linear + self.d_v_quadratic * abs(v)
        d_r = self.yaw_damping_linear(u) + self.d_r_quadratic * abs(r)
        return np.diag([d_u, d_v, d_r])

    def yaw_state_matrices(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Linear yaw subsystem ``x=[psi, r]``, input ``delta`` [rad]."""

        a = self.yaw_a(u)
        b = self.yaw_b(u)
        A = np.array([[0.0, 1.0], [0.0, -a]], dtype=float)
        B = np.array([[0.0], [b]], dtype=float)
        return A, B

    def continuous_yaw_transfer(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Numerator/denominator of ``psi(s)/delta(s)``."""

        a = self.yaw_a(u)
        b = self.yaw_b(u)
        return np.array([0.0, b], dtype=float), np.array([1.0, a, 0.0], dtype=float)


def default_vehicle() -> VehicleParameters:
    return VehicleParameters()


def mismatch_reference(parameters: VehicleParameters | None = None) -> VehicleParameters:
    """Deliberately biased internal model used by the mismatch experiments.

    The controller over-estimates linear yaw damping by 30 % and
    under-estimates yaw effectiveness by 40 %.  This is a *hypothesis* chosen
    to make the adaptation problem non-trivial; it is not an identified IARA
    parameter set.  The plant always uses the nominal parameters.
    """

    from dataclasses import replace

    p = parameters or default_vehicle()
    return replace(
        p,
        a_n=1.3 * p.a_n,
        b_0=0.6 * p.b_0,
        b_n=0.6 * p.b_n,
    )


def physical_checks(parameters: VehicleParameters | None = None) -> dict[str, float | bool]:
    """Numerical invariants required by the V4 evidence chain."""

    p = parameters or default_vehicle()
    M = p.mass_matrix
    nu = np.array([0.35, 0.05, 0.2], dtype=float)
    C = p.coriolis(nu)
    D = p.damping(nu)
    return {
        "m_min_eigenvalue": float(np.min(np.linalg.eigvalsh(M))),
        "m_positive_definite": bool(np.all(np.linalg.eigvalsh(M) > 0.0)),
        "coriolis_skew_symmetric": bool(np.allclose(C, -C.T, atol=1e-12)),
        "coriolis_quadratic_form": float(nu @ C @ nu),
        "damping_dissipative": float(nu @ D @ nu),
        "damping_positive_semidefinite": bool(np.all(np.diag(D) >= 0.0)),
        "yaw_a": p.yaw_a(nu[0]),
        "yaw_b": p.yaw_b(nu[0]),
        "yaw_b_low_speed": p.b_0,
        "minimum_turn_radius_nominal": float(
            p.u_nominal / (p.yaw_b(p.u_nominal) * p.rudder_max)
        ),
    }
