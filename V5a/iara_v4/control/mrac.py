"""Direct Model Reference Adaptive Control (MRAC) for the yaw subsystem.

Plant (nominal linearisation at surge speed ``u``):

    x = [psi, r]^T
    x_dot = A(u) x + B(u) delta
    A(u) = [[0, 1], [0, -a(u)]],   B(u) = [0, b(u)]^T

Reference model:

    x_m_dot = A_m x_m + B_m psi_ref
    A_m = [[0, 1], [-omega_n^2, -2 zeta omega_n]]
    B_m = [0, omega_n^2]^T

Matching gains:

    K_x^* = [-omega_n^2 / b, (a - 2 zeta omega_n) / b]^T
    K_r^* = omega_n^2 / b

The implementation uses the standard Lyapunov adaptation law with
sigma-modification, projection and optional adaptation freeze during actuator
saturation.  The ideal proof (``V_dot = -e^T Q e``) holds only without
saturation, deadband, disturbance and parameter variation.  The code and the
article report the practical/bounded interpretation instead of claiming ideal
convergence under those effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

from ..core.kinematics import DifferentialDrive
from ..core.types import ControlCommand, Measurement, Vec2, wrap_angle
from ..dynamics.model import VehicleParameters
from .base import ControllerContext, TelemetryBuffer


@dataclass(frozen=True, slots=True)
class MRACConfig:
    omega_n: float = 5.5
    zeta: float = 1.0
    q_lyapunov: float = 1.0
    gamma_psi: float = 120.0
    gamma_r: float = 60.0
    gamma_ref: float = 120.0
    sigma: float = 1e-5
    normalize: bool = True
    normalization_eps: float = 1e-3
    adapt: bool = True
    freeze_on_saturation: bool = True
    initial_speed: float = 0.35
    initial_gain_scale: float = 1.0
    delta_max: float = 0.65
    kx1_min: float = -600.0
    kx1_max: float = 50.0
    kx2_min: float = -600.0
    kx2_max: float = 50.0
    kr_min: float = 0.0
    kr_max: float = 600.0

    def __post_init__(self) -> None:
        if self.omega_n <= 0.0 or self.zeta <= 0.0:
            raise ValueError("reference-model parameters must be positive")
        if min(self.gamma_psi, self.gamma_r, self.gamma_ref) <= 0.0:
            raise ValueError("adaptation rates must be positive")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.delta_max <= 0.0:
            raise ValueError("delta_max must be positive")
        if self.initial_gain_scale <= 0.0:
            raise ValueError("initial_gain_scale must be positive")
        if not (self.kx1_min < self.kx1_max and self.kx2_min < self.kx2_max):
            raise ValueError("projection bounds must be ordered")
        if self.kr_min > self.kr_max:
            raise ValueError("Kr projection bounds must be ordered")


@dataclass(frozen=True, slots=True)
class ReferenceModel:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    P: np.ndarray

    @classmethod
    def build(cls, omega_n: float, zeta: float, q_scale: float = 1.0) -> "ReferenceModel":
        A = np.array([[0.0, 1.0], [-omega_n * omega_n, -2.0 * zeta * omega_n]])
        B = np.array([[0.0], [omega_n * omega_n]])
        Q = q_scale * np.eye(2)
        P = solve_continuous_lyapunov(A.T, -Q)
        P = 0.5 * (P + P.T)
        return cls(A=A, B=B, Q=Q, P=P)

    def check(self) -> dict[str, float | bool]:
        eig_A = np.linalg.eigvals(self.A)
        eig_P = np.linalg.eigvalsh(self.P)
        residual = self.A.T @ self.P + self.P @ self.A + self.Q
        return {
            "hurwitz": bool(np.all(np.real(eig_A) < 0.0)),
            "max_real_eigenvalue_A": float(np.max(np.real(eig_A))),
            "p_min_eigenvalue": float(np.min(eig_P)),
            "p_positive_definite": bool(np.all(eig_P > 0.0)),
            "lyapunov_residual_norm": float(np.linalg.norm(residual)),
        }

    def matching_gains(self, vehicle: VehicleParameters, u: float) -> tuple[np.ndarray, float]:
        a = vehicle.yaw_a(u)
        b = max(vehicle.yaw_b(u), 1e-12)
        # Matching: b kx1 = A_m[1,0], -a + b kx2 = A_m[1,1],
        # b kr = B_m[1,0].
        kx = np.array(
            [
                self.A[1, 0] / b,
                (a + self.A[1, 1]) / b,
            ],
            dtype=float,
        )
        kr = self.B[1, 0] / b
        return kx, float(kr)

    def matching_residual(
        self, vehicle: VehicleParameters, u: float
    ) -> dict[str, float | bool]:
        A, B = vehicle.yaw_state_matrices(u)
        kx, kr = self.matching_gains(vehicle, u)
        left = A + B @ kx.reshape(1, -1)
        right = self.A
        gain_left = B * kr
        return {
            "A_residual_norm": float(np.linalg.norm(left - right)),
            "B_residual_norm": float(np.linalg.norm(gain_left - self.B)),
            "matching_exact": bool(
                np.allclose(left, right, atol=1e-10)
                and np.allclose(gain_left, self.B, atol=1e-10)
            ),
        }


class MRACController:
    """Direct MRAC heading autopilot with projection and sigma-modification."""

    name = "mrac"

    def __init__(
        self,
        config: MRACConfig | None = None,
        *,
        drive: DifferentialDrive | None = None,
        vehicle: VehicleParameters | None = None,
    ) -> None:
        self.config = config or MRACConfig()
        self.drive = drive or DifferentialDrive()
        self.vehicle = vehicle or VehicleParameters()
        self.reference = ReferenceModel.build(
            self.config.omega_n, self.config.zeta, self.config.q_lyapunov
        )
        self._telemetry = TelemetryBuffer()
        self._reference_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.reset()

    # -- reference model ----------------------------------------------------

    def _discretize_reference(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(dt), 12)
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        A = self.reference.A
        B = self.reference.B
        n, m = B.shape
        aug = np.zeros((n + m, n + m), dtype=float)
        aug[:n, :n] = A
        aug[:n, n:] = B
        exp_aug = expm(aug * float(dt))
        Ad = exp_aug[:n, :n]
        Bd = exp_aug[:n, n:]
        self._reference_cache[key] = (Ad, Bd)
        return Ad, Bd

    # -- reset / update -----------------------------------------------------

    def reset(
        self,
        measurement: Measurement | None = None,
        reference: Vec2 | None = None,
        seed: int = 0,
    ) -> None:
        psi0 = measurement.pose.theta if measurement is not None else 0.0
        r0 = measurement.r if measurement is not None else 0.0
        speed0 = (
            measurement.u
            if measurement is not None and measurement.u > 0.05
            else self.config.initial_speed
        )
        self._x_m = np.array([psi0, r0], dtype=float)
        kx, kr = self.reference.matching_gains(self.vehicle, speed0)
        kx = kx * self.config.initial_gain_scale
        kr = kr * self.config.initial_gain_scale
        self._kx = self._project_kx(kx)
        self._kr = float(np.clip(kr, self.config.kr_min, self.config.kr_max))
        self._initial_gains = (self._kx.copy(), self._kr)
        self._previous_delta_saturated = False
        self._telemetry.set(
            time=0.0,
            kx1=float(self._kx[0]),
            kx2=float(self._kx[1]),
            kr=float(self._kr),
            error_psi=0.0,
            error_r=0.0,
            delta_cmd=0.0,
            adaptation_saturated=False,
        )

    def _project_kx(self, kx: np.ndarray) -> np.ndarray:
        c = self.config
        return np.array(
            [
                min(max(float(kx[0]), c.kx1_min), c.kx1_max),
                min(max(float(kx[1]), c.kx2_min), c.kx2_max),
            ],
            dtype=float,
        )

    def update(self, context: ControllerContext) -> ControlCommand:
        c = self.config
        m = context.measurement
        g = context.guidance
        dt = max(context.dt, 1e-9)
        x = np.array([m.pose.theta, m.r], dtype=float)
        psi_ref = float(wrap_angle(g.heading_ref))

        Ad, Bd = self._discretize_reference(dt)
        self._x_m = Ad @ self._x_m + Bd[:, 0] * psi_ref
        self._x_m[0] = wrap_angle(float(self._x_m[0]))
        error = x - self._x_m
        error[0] = wrap_angle(float(error[0]))

        b = max(self.vehicle.yaw_b(m.u), 1e-12)
        B = np.array([[0.0], [b]], dtype=float)
        e_t_p_b = float(error @ self.reference.P @ B[:, 0])

        kx_before = self._kx.copy()
        kr_before = float(self._kr)

        if c.adapt and not (c.freeze_on_saturation and self._previous_delta_saturated):
            norm = 1.0
            if c.normalize:
                norm = 1.0 + float(x @ x) + psi_ref * psi_ref + c.normalization_eps
            dkx_raw = (
                -np.array([c.gamma_psi, c.gamma_r], dtype=float)
                * x
                * e_t_p_b
                / norm
                - c.sigma * np.array([c.gamma_psi, c.gamma_r], dtype=float) * self._kx
            )
            dkr_raw = -c.gamma_ref * psi_ref * e_t_p_b / norm - c.sigma * c.gamma_ref * self._kr
            kx_new = self._project_kx(self._kx + dt * dkx_raw)
            kr_new = float(np.clip(self._kr + dt * dkr_raw, c.kr_min, c.kr_max))
            self._kx = kx_new
            self._kr = kr_new
        else:
            dkx_raw = np.zeros(2, dtype=float)
            dkr_raw = 0.0

        delta_cmd = float(self._kx @ x + self._kr * psi_ref)
        saturated = abs(delta_cmd) > c.delta_max
        delta_cmd = max(-c.delta_max, min(c.delta_max, delta_cmd))
        self._previous_delta_saturated = bool(saturated)

        a = max(self.vehicle.yaw_a(m.u), 1e-9)
        omega_cmd = delta_cmd * b / a
        omega_cmd = max(-2.0, min(2.0, omega_cmd))
        wheels = self.drive.inverse(g.speed, omega_cmd)

        dkx_applied = self._kx - kx_before
        dkr_applied = self._kr - kr_before
        self._telemetry.set(
            time=m.time,
            speed=m.u,
            kx1=float(self._kx[0]),
            kx2=float(self._kx[1]),
            kr=float(self._kr),
            kx1_raw=float(dkx_raw[0]),
            kx2_raw=float(dkx_raw[1]),
            kr_raw=float(dkr_raw),
            dkx1_applied=float(dkx_applied[0]),
            dkx2_applied=float(dkx_applied[1]),
            dkr_applied=float(dkr_applied),
            error_psi=float(error[0]),
            error_r=float(error[1]),
            error_norm=float(np.linalg.norm(error)),
            psi_m=float(self._x_m[0]),
            r_m=float(self._x_m[1]),
            psi_ref=psi_ref,
            delta_cmd=float(delta_cmd),
            omega_cmd=float(omega_cmd),
            e_t_p_b=float(e_t_p_b),
            saturated=bool(saturated),
            adaptation_saturated=bool(c.freeze_on_saturation and self._previous_delta_saturated),
            adaptation_effort=float(np.linalg.norm(dkx_applied) + abs(dkr_applied)),
        )
        return ControlCommand(
            v=g.speed,
            omega=omega_cmd,
            wheels=wheels,
            saturated=bool(saturated),
            delta_rudder=delta_cmd,
            target=g.target,
            metadata={
                "kx1": float(self._kx[0]),
                "kx2": float(self._kx[1]),
                "kr": float(self._kr),
                "error_norm": float(np.linalg.norm(error)),
            },
        )

    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.get()

    # -- analysis helpers ---------------------------------------------------

    def matching_report(self, speeds: tuple[float, ...] = (0.1, 0.35, 0.8, 1.0)) -> dict[str, Any]:
        return {
            "reference_model": self.reference.check(),
            "matching": {
                f"{speed:g}": self.reference.matching_residual(self.vehicle, speed)
                for speed in speeds
            },
            "config": {
                "omega_n": self.config.omega_n,
                "zeta": self.config.zeta,
                "gamma_psi": self.config.gamma_psi,
                "gamma_r": self.config.gamma_r,
                "gamma_ref": self.config.gamma_ref,
                "sigma": self.config.sigma,
                "projection": {
                    "kx1": [self.config.kx1_min, self.config.kx1_max],
                    "kx2": [self.config.kx2_min, self.config.kx2_max],
                    "kr": [self.config.kr_min, self.config.kr_max],
                },
            },
        }
