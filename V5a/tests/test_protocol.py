from __future__ import annotations

import json

import pytest
import yaml

from iara_v4.experiments.protocol import (
    Condition,
    ExperimentProtocol,
    check_test_authorization,
    freeze_protocol,
)
from iara_v4.core.seeds import pool


def _config_text(phase: str, n_seeds: int = 1) -> str:
    return f"""
phase: {phase}
n_seeds: {n_seeds}
scenarios:
  - open_water
conditions:
  - name: pp
    planner: rrt_star
    planner_options:
      max_iterations: 200
      step_size: 0.20
"""


def test_loads_declarative_condition() -> None:
    condition = Condition.from_dict({"name": "mrac", "controller_options": {"omega_n": 5.5}})
    assert condition.name == "mrac"
    assert condition.controller_options["omega_n"] == 5.5


def test_validation_protocol_resolves_pool_seeds(tmp_path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text(_config_text("validation"), encoding="utf-8")
    protocol = ExperimentProtocol.from_yaml(path)
    assert protocol.n_seeds == 1
    assert protocol.resolved_seeds == [20]


def test_seed_pool_is_enforced() -> None:
    with pytest.raises(ValueError):
        ExperimentProtocol(
            phase="validation",
            conditions=[Condition("pp")],
            n_seeds=1,
            scenarios=["open_water"],
            seeds=[0],
        )


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentProtocol(
            phase="validation",
            conditions=[Condition("pp")],
            n_seeds=1,
            scenarios=["not_a_scene"],
        )


def test_protocol_writes_artifacts(tmp_path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text(_config_text("tune"), encoding="utf-8")
    protocol = ExperimentProtocol.from_yaml(path)
    artifacts = protocol.run(output_dir=tmp_path / "run", max_episodes=1, progress=False)
    assert artifacts.n_episodes == 1
    assert artifacts.episodes_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()
    assert (tmp_path / "run" / "checksums.sha256").exists()
    assert (tmp_path / "run" / "config_snapshot.yaml").exists()
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["n_episodes"] == 1
    assert summary["protocol"]["phase"] == "tune"


def test_test_phase_requires_authorization(tmp_path) -> None:
    path = tmp_path / "protocol_test.yaml"
    path.write_text(_config_text("test"), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError):
        check_test_authorization(
            path,
            manifest,
            authorize=False,
            phase_override="test",
        )


def test_freeze_and_authorize_test_protocol(tmp_path) -> None:
    path = tmp_path / "protocol_test.yaml"
    path.write_text(_config_text("test"), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_protocol(path, manifest_path)
    assert manifest["seeds"] == [40]
    assert manifest["phase"] == "test"
    check_test_authorization(
        path,
        manifest_path,
        authorize=True,
        phase_override="test",
    )


def test_tampered_test_configuration_is_rejected(tmp_path) -> None:
    path = tmp_path / "protocol_test.yaml"
    path.write_text(_config_text("test"), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    freeze_protocol(path, manifest_path)
    path.write_text(
        _config_text("test") + "# changed after freezing\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        check_test_authorization(
            path,
            manifest_path,
            authorize=True,
            phase_override="test",
        )


def test_validation_phase_needs_no_manifest(tmp_path) -> None:
    path = tmp_path / "protocol_validation.yaml"
    path.write_text(_config_text("validation"), encoding="utf-8")
    check_test_authorization(
        path,
        tmp_path / "missing.json",
        authorize=False,
        phase_override="validation",
    )
