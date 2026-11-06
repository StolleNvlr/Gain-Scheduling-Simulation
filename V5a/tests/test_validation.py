from __future__ import annotations

import numpy as np
import pytest

from iara_v4.experiments.validation import run_step_response, step_response_validation


def test_step_response_has_declared_length_and_finite_values() -> None:
    result = run_step_response(adapt=True, duration=2.0, dt=0.02)
    assert result.time.size == 100
    assert result.heading.size == result.time.size
    assert result.reference_model.size == result.time.size
    assert np.all(np.isfinite(result.heading))
    assert np.all(np.isfinite(result.kr))
    assert np.all(np.abs(result.rudder) <= 0.65 + 1e-12)


def test_fixed_gain_keeps_estimated_gain_constant() -> None:
    result = run_step_response(adapt=False, duration=1.0)
    assert np.allclose(result.kr, result.kr[0])


def test_step_validation_is_deterministic_and_reports_two_gates() -> None:
    first = step_response_validation()
    second = step_response_validation()
    assert first["pass"] is True
    assert first["gate_version"] == "A1-v2-split-gates"
    assert first["adaptive"]["reference_model_rmse"] == pytest.approx(
        second["adaptive"]["reference_model_rmse"]
    )
    assert isinstance(first["adaptation_superiority_demonstrated"], bool)


def test_gate_separation_is_consistent() -> None:
    result = step_response_validation()
    if result["adaptation_superiority_demonstrated"]:
        assert result["pass"] is True
