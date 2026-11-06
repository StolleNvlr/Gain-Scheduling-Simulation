from __future__ import annotations

import numpy as np
import pytest

from iara_v4.control.mrac import MRACConfig, MRACController
from iara_v4.dynamics.model import VehicleParameters
from iara_v4.experiments.validation import step_response_validation


def test_reference_model_is_hurwitz_and_lyapunov_positive() -> None:
    controller = MRACController(MRACConfig(omega_n=5.5, zeta=1.0))
    check = controller.reference.check()
    assert check["hurwitz"] is True
    assert check["p_positive_definite"] is True
    assert check["lyapunov_residual_norm"] < 1e-10


def test_matching_is_exact_at_nominal_speeds() -> None:
    p = VehicleParameters()
    controller = MRACController(MRACConfig(omega_n=5.5), vehicle=p)
    report = controller.matching_report(speeds=(0.1, 0.35, 0.8, 1.0))
    for row in report["matching"].values():
        assert row["matching_exact"] is True
        assert row["A_residual_norm"] < 1e-10
        assert row["B_residual_norm"] < 1e-10


def test_matching_changes_with_low_speed_yaw_authority() -> None:
    p = VehicleParameters()
    controller = MRACController(MRACConfig(omega_n=5.5), vehicle=p)
    _, kr_fast = controller.reference.matching_gains(p, 1.0)
    _, kr_slow = controller.reference.matching_gains(p, 0.1)
    # b(0.1) = 0.83 and b(1.0) = 3.8, so the low-speed gain is larger.
    assert kr_slow > kr_fast > 0.0


def test_initial_gain_scale_is_applied() -> None:
    p = VehicleParameters()
    controller = MRACController(
        MRACConfig(omega_n=5.5, initial_gain_scale=0.25), vehicle=p
    )
    _, matching = controller.reference.matching_gains(p, controller.config.initial_speed)
    telemetry = controller.telemetry()
    assert telemetry["kr"] == pytest.approx(0.25 * matching, rel=1e-6)


def test_projection_bounds_are_enforced() -> None:
    config = MRACConfig(kr_min=0.0, kr_max=10.0)
    controller = MRACController(config)
    controller._kx = np.array([1e9, -1e9])  # deliberately out of bounds
    projected = controller._project_kx(controller._kx)
    assert config.kx1_min <= projected[0] <= config.kx1_max
    assert config.kx2_min <= projected[1] <= config.kx2_max
    assert 0.0 <= controller._kr <= 10.0


def test_invalid_adaptation_rates_are_rejected() -> None:
    with pytest.raises(ValueError):
        MRACConfig(gamma_psi=0.0)
    with pytest.raises(ValueError):
        MRACConfig(sigma=-1.0)
    with pytest.raises(ValueError):
        MRACConfig(initial_gain_scale=0.0)


def test_step_validation_passes_numerical_sanity_gate() -> None:
    result = step_response_validation()
    assert result["pass"] is True
    assert result["gate_version"] == "A1-v2-split-gates"
    assert result["adaptive"]["all_finite"] is True
    assert np.isfinite(result["relative_rmse_improvement"])
    adaptive = result["adaptive"]
    assert np.isfinite(adaptive["final_kr"])
    assert adaptive["max_abs_rudder"] <= 0.65 + 1e-12
    assert result["fixed"]["adapt"] is False
    assert np.all(np.isfinite(result["series"]["adaptive"].heading))
    assert np.all(np.isfinite(result["series"]["adaptive"].kr))


def test_adaptation_superiority_is_a_separate_honest_gate() -> None:
    result = step_response_validation()
    assert isinstance(result["adaptation_superiority_demonstrated"], bool)
    assert isinstance(result["rmse_improved_by_adaptation"], bool)
    assert isinstance(result["adaptive_gain_moved_toward_matching"], bool)
    if result["adaptation_superiority_demonstrated"]:
        assert result["rmse_improved_by_adaptation"] is True
        assert result["adaptive_gain_moved_toward_matching"] is True
