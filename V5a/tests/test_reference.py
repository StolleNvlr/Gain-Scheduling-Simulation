from __future__ import annotations

import math

import numpy as np
import pytest

from iara_v4.experiments.reference import (
    MATLAB_BRANCHES,
    analyze_scheduler,
    audit_report,
    matlab_design,
    temporal_interpretations,
    zoh_arx,
)
from iara_v4.control.gain_scheduler import GainScheduler
from iara_v4.dynamics.model import VehicleParameters


def test_continuous_design_matches_script_coefficients() -> None:
    model = matlab_design(1.0)
    assert model.a == pytest.approx(1.6)
    assert model.b == pytest.approx(0.2)
    assert model.k == pytest.approx(30.0)
    # The script uses 2.4/q, so zeta at q=1 is 1.2.
    assert model.zeta == pytest.approx(1.2)
    assert model.wn == pytest.approx(math.sqrt(6.0))


def test_zeta_depends_on_speed_as_1_2_over_q() -> None:
    for q in (0.1, 0.8, 1.0):
        model = matlab_design(q)
        assert model.zeta == pytest.approx(1.2 / q)


def test_branch_labels_are_shifted() -> None:
    mapping = audit_report()["branch_mapping"]
    best = {
        row["branch_speed"]: row["best_model_q"] for row in mapping["branches"]
    }
    assert best[0.8] == pytest.approx(1.0)
    assert best[0.1] == pytest.approx(0.8)
    assert best[1.0] == pytest.approx(0.1)
    assert mapping["label_shift_detected"] is True
    for row in mapping["branches"]:
        assert row["within_tolerance"] is True


def test_zoh_arx_reproduces_hard_coded_branch() -> None:
    branch = next(b for b in MATLAB_BRANCHES if b.speed_label == 0.1)
    arx = zoh_arx(matlab_design(0.8), 1.0)
    assert arx.c1 == pytest.approx(branch.c1, abs=2e-5)
    assert arx.c2 == pytest.approx(branch.c2, abs=5e-8)
    assert arx.d0 == pytest.approx(branch.d0, abs=1e-5)
    assert arx.d1 == pytest.approx(branch.d1, abs=5e-5)


def test_temporal_interpretations_are_distinct() -> None:
    data = temporal_interpretations()
    labels = [row["interpretation"] for row in data["rows"]]
    assert labels == [
        "one_hz_zoh",
        "rediscretised_100_hz",
        "direct_1hz_at_100hz",
    ]
    assert data["arx_1s"]["ts"] == pytest.approx(1.0)
    assert data["arx_100hz"]["ts"] == pytest.approx(0.01)


def test_corrected_scheduler_admits_cqlf() -> None:
    result = analyze_scheduler(
        VehicleParameters(),
        GainScheduler.corrected(omega_n=2.0, zeta=1.0),
        ts=0.05,
    )
    assert result["max_spectral_radius"] < 1.0
    assert result["cqlf"]["success"] is True
    assert result["cqlf"]["min_eigenvalue_P"] > 0.0
    assert max(result["cqlf"]["margins"]) < 0.0


def test_audit_report_exposes_structured_gates() -> None:
    gates = audit_report()["gates"]
    expected = {
        "label_shift_detected",
        "all_branches_within_tolerance",
        "three_temporal_interpretations",
        "zeta_speed_dependent_verified",
        "ts_mismatch_documented",
    }
    assert set(gates) == expected
    for name, value in gates.items():
        assert value is True, f"gate {name} is {value}"
