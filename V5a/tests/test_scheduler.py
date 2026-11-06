from __future__ import annotations

import math

import numpy as np
import pytest

from iara_v4.control.gain_scheduler import GainScheduler
from iara_v4.dynamics.model import VehicleParameters


def test_corrected_gains_equalise_closed_loop_yaw_matrices() -> None:
    p = VehicleParameters()
    scheduler = GainScheduler.corrected(p, omega_n=2.0, zeta=1.0)
    matrices = []
    for speed in (0.1, 0.55, 1.0):
        gains = scheduler.gains(speed)
        a = p.yaw_a(speed)
        b = p.yaw_b(speed)
        matrices.append(
            np.array(
                [[0.0, 1.0], [-b * gains.kp, -a - b * gains.kd]], dtype=float
            )
        )
    for A in matrices[1:]:
        assert np.allclose(A, matrices[0], atol=1e-12)


def test_interpolation_weights_are_convex_and_sum_to_one() -> None:
    scheduler = GainScheduler.corrected(speeds=(0.1, 0.8, 1.0))
    for speed in (0.05, 0.1, 0.3, 0.8, 0.9, 1.0, 1.2):
        result = scheduler.gains(speed)
        assert sum(weight for _, weight in result.interpolation_weights) == pytest.approx(1.0)
        assert all(weight >= 0.0 for _, weight in result.interpolation_weights)


def test_corrected_scheduler_marks_applied_directly() -> None:
    scheduler = GainScheduler.corrected()
    result = scheduler.gains(0.5)
    assert result.applied_directly is True
    by_speed = {p.speed: p for p in scheduler.config.points}
    interpolated = sum(
        weight * by_speed[speed].kp
        for speed, weight in result.interpolation_weights
    )
    assert result.kp != pytest.approx(interpolated, rel=1e-6)


def test_matlab_absolute_scheduler_interpolates_gains() -> None:
    scheduler = GainScheduler.matlab_absolute()
    for speed in (0.3, 0.5, 0.9):
        result = scheduler.gains(speed)
        assert result.applied_directly is False
        by_speed = {p.speed: p for p in scheduler.config.points}
        expected = sum(
            weight * by_speed[point_speed].kp
            for point_speed, weight in result.interpolation_weights
        )
        assert result.kp == pytest.approx(expected, rel=1e-9)


def test_schedule_clamps_outside_envelope() -> None:
    scheduler = GainScheduler.corrected(speeds=(0.1, 0.8, 1.0))
    below = scheduler.gains(-0.5)
    above = scheduler.gains(2.0)
    assert below.clamped is True
    assert above.clamped is True
    assert below.kp == pytest.approx(scheduler.gains(0.1).kp)
    assert above.kp == pytest.approx(scheduler.gains(1.0).kp)


def test_corrected_midpoint_preserves_closed_loop_matrix() -> None:
    scheduler = GainScheduler.corrected(speeds=(0.1, 0.8, 1.0))
    midpoint = scheduler.gains(0.45)
    p = VehicleParameters()
    matrix = np.array(
        [
            [0.0, 1.0],
            [-p.yaw_b(0.45) * midpoint.kp, -p.yaw_a(0.45) - p.yaw_b(0.45) * midpoint.kd],
        ]
    )
    assert matrix[1, 0] == pytest.approx(-4.0)
    assert matrix[1, 1] == pytest.approx(-4.0)


def test_matlab_absolute_gains_are_retained_for_audit() -> None:
    scheduler = GainScheduler.matlab_absolute()
    gains = scheduler.gains(0.8)
    q = 0.8
    a = 1.6 * q
    b = 0.2 * q * q
    kp = 30.0 / (q * q)
    td = (2.4 / q * math.sqrt(b * kp) - a) / (b * kp)
    assert gains.kp == pytest.approx(kp)
    assert gains.kd == pytest.approx(kp * td)


def test_rate_violation_is_declared() -> None:
    scheduler = GainScheduler.corrected(gamma_max=2.0)
    assert scheduler.rate_violation(1.9) is False
    assert scheduler.rate_violation(2.1) is True


def test_gain_scheduler_requires_ordered_points() -> None:
    from iara_v4.control.gain_scheduler import GainSchedulerConfig, OperatingPoint

    with pytest.raises(ValueError):
        GainSchedulerConfig(
            (
                OperatingPoint(1.0, 1.0, 1.0),
                OperatingPoint(0.5, 1.0, 1.0),
            )
        )
