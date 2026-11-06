"""Deterministic, seeded environmental disturbances.

The model is intentionally simple and is labelled as a *lumped* disturbance:
a slowly varying current force, a sinusoidal wind force and a small wave yaw
moment.  It is not an identified wind/current/wave model and it is not used to
claim physical validity.  Its purpose is to make the comparison robust to a
bounded, reproducible exogenous input.

The random gust is an Ornstein-Uhlenbeck process driven by a generator derived
from ``(seed, scenario)``.  The stream excludes the controller name, so every
condition in a paired episode sees exactly the same disturbance realisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.seeds import rng_for, stable_hash


@dataclass(frozen=True, slots=True)
class DisturbanceConfig:
    enabled: bool = True
    current_speed: float = 0.018
    current_direction: float = 0.35
    wind_force: float = 0.10
    wind_direction: float = -0.25
    wind_frequency: float = 0.13
    wave_yaw_amplitude: float = 0.0025
    wave_frequency: float = 0.27
    gust_std: float = 0.02
    gust_time_constant: float = 1.2

    def __post_init__(self) -> None:
        if min(self.current_speed, self.wind_force, self.wave_yaw_amplitude, self.gust_std) < 0.0:
            raise ValueError("disturbance amplitudes must be non-negative")
        if self.gust_time_constant <= 0.0:
            raise ValueError("gust_time_constant must be positive")


class DisturbanceModel:
    def __init__(
        self,
        config: DisturbanceConfig | None = None,
        *,
        seed: int = 0,
        scenario: str = "",
    ) -> None:
        self.config = config or DisturbanceConfig()
        self._rng = rng_for(seed, stable_hash(f"disturbance:{scenario}"))
        self._phase = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self._gust = 0.0
        self._time = 0.0

    def reset(self) -> None:
        self._gust = 0.0
        self._time = 0.0

    def update(self, dt: float) -> None:
        """Advance the stochastic gust by one plant/control interval."""

        if not self.config.enabled or dt <= 0.0:
            return
        c = self.config
        decay = math.exp(-dt / c.gust_time_constant)
        noise = self._rng.normal(0.0, c.gust_std * math.sqrt(max(0.0, 1.0 - decay * decay)))
        self._gust = decay * self._gust + noise
        self._time += dt

    def force(self, psi: float) -> np.ndarray:
        """Body-frame generalized disturbance ``[F_x, F_y, M_z]``."""

        c = self.config
        if not c.enabled:
            return np.zeros(3, dtype=float)
        t = self._time
        # Current: a constant world-frame force transformed to body axes.
        cur_world = np.array(
            [
                c.current_speed * math.cos(c.current_direction),
                c.current_speed * math.sin(c.current_direction),
            ],
            dtype=float,
        )
        cpsi, spsi = math.cos(psi), math.sin(psi)
        cur_body = np.array(
            [cpsi * cur_world[0] + spsi * cur_world[1], -spsi * cur_world[0] + cpsi * cur_world[1]],
            dtype=float,
        )
        # Wind: slowly modulated force, also transformed to body axes.
        wind_mag = c.wind_force * (1.0 + 0.35 * math.sin(2.0 * math.pi * c.wind_frequency * t + self._phase))
        wind_world = np.array(
            [wind_mag * math.cos(c.wind_direction), wind_mag * math.sin(c.wind_direction)],
            dtype=float,
        )
        wind_body = np.array(
            [cpsi * wind_world[0] + spsi * wind_world[1], -spsi * wind_world[0] + cpsi * wind_world[1]],
            dtype=float,
        )
        gust_body = np.array([self._gust, 0.35 * self._gust], dtype=float)
        yaw = c.wave_yaw_amplitude * math.sin(2.0 * math.pi * c.wave_frequency * t + 0.5 * self._phase)
        return np.array(
            [cur_body[0] + wind_body[0] + gust_body[0], cur_body[1] + wind_body[1] + gust_body[1], yaw],
            dtype=float,
        )
