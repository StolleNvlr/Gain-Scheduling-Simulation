"""V4 experiment protocol, seed-pool enforcement and artifact writing.

The protocol is intentionally thin: it expands a declarative condition list into
``EpisodeConfig`` objects, refuses to run a test phase without a frozen
manifest, and writes every episode as JSON Lines before aggregation.  The
condition controller/plant semantics live in ``simulation.runner`` so that a
hand-written single-episode script and the batch protocol cannot silently use
different definitions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy
import yaml

from ..core.seeds import SEED_POOLS, pool, validate_pools, validate_seed_set
from ..dynamics.disturbance import DisturbanceConfig
from ..dynamics.model import VehicleParameters
from ..simulation.runner import (
    EpisodeConfig,
    EpisodeResult,
    SensorConfig,
    condition_spec,
    run_episode,
)
from ..simulation.world import default_scenarios, scenario_by_name
from .report import (
    build_summary,
    write_report,
    write_summary_csv,
    write_summary_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Condition:
    """One row of the experiment matrix.

    ``name`` selects a ``ConditionSpec`` from ``simulation.runner``.  The
    remaining fields only override planner/controller/local-planner options;
    they cannot silently change the plant or the controller class.
    """

    name: str
    planner: str = "rrt_star"
    planner_options: dict[str, Any] = field(default_factory=dict)
    controller_options: dict[str, Any] = field(default_factory=dict)
    local_planner_options: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        condition_spec(self.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Condition":
        return cls(
            name=str(data["name"]),
            planner=str(data.get("planner", "rrt_star")),
            planner_options=dict(data.get("planner_options", {})),
            controller_options=dict(data.get("controller_options", {})),
            local_planner_options=dict(data.get("local_planner_options", {})),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        spec = condition_spec(self.name)
        return {
            "name": self.name,
            "planner": self.planner,
            "planner_options": dict(self.planner_options),
            "controller_options": dict(self.controller_options),
            "local_planner_options": dict(self.local_planner_options),
            "description": self.description or spec.description,
            "condition_spec": spec.to_dict(),
        }


@dataclass(slots=True)
class ExperimentProtocol:
    phase: str
    conditions: list[Condition]
    n_seeds: int = 10
    scenarios: list[str] = field(
        default_factory=lambda: [s.name for s in default_scenarios()]
    )
    seeds: list[int] | None = None
    dt: float = 0.05
    substeps: int = 4
    smoothing: bool = True
    smooth_ds: float = 0.02
    max_path_curvature: float = 4.0
    record_trajectory: bool = False
    record_telemetry: bool = False
    sensor: SensorConfig = field(default_factory=SensorConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
    source_path: str = ""

    def __post_init__(self) -> None:
        self.validate()

    @property
    def resolved_seeds(self) -> list[int]:
        if self.seeds is not None:
            return list(self.seeds)
        return list(pool(self.phase))[: self.n_seeds]

    def validate(self) -> None:
        validate_pools()
        if self.phase not in SEED_POOLS:
            raise ValueError(f"unknown phase {self.phase!r}")
        if not 1 <= self.n_seeds <= len(pool(self.phase)):
            raise ValueError(
                f"n_seeds must be in [1, {len(pool(self.phase))}] for phase "
                f"{self.phase!r}"
            )
        if not self.conditions:
            raise ValueError("at least one condition is required")
        names = [c.name for c in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("condition names must be unique")
        known = {s.name for s in default_scenarios()}
        unknown = sorted(set(self.scenarios) - known)
        if unknown:
            raise ValueError(f"unknown scenarios: {unknown}")
        validate_seed_set(self.resolved_seeds, self.phase)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1")
        if self.smooth_ds <= 0.0:
            raise ValueError("smooth_ds must be positive")
        if self.max_path_curvature <= 0.0:
            raise ValueError("max_path_curvature must be positive")

    def iter_episodes(self) -> Iterator[EpisodeConfig]:
        scenario_map = {s.name: s for s in default_scenarios()}
        for condition in self.conditions:
            for scenario_name in self.scenarios:
                scenario = scenario_map[scenario_name]
                for seed in self.resolved_seeds:
                    yield EpisodeConfig(
                        scenario=scenario,
                        condition=condition.name,
                        seed=seed,
                        phase=self.phase,
                        spec=condition_spec(condition.name),
                        dt=self.dt,
                        substeps=self.substeps,
                        planner=condition.planner,
                        smoothing=self.smoothing,
                        smooth_ds=self.smooth_ds,
                        max_path_curvature=self.max_path_curvature,
                        record_trajectory=self.record_trajectory,
                        record_telemetry=self.record_telemetry,
                        planner_options=dict(condition.planner_options),
                        controller_options=dict(condition.controller_options),
                        local_planner_options=dict(
                            condition.local_planner_options
                        ),
                        sensor=self.sensor,
                        disturbance=self.disturbance,
                        vehicle=self.vehicle,
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "n_seeds": self.n_seeds,
            "seeds": self.seeds,
            "scenarios": list(self.scenarios),
            "dt": self.dt,
            "substeps": self.substeps,
            "smoothing": self.smoothing,
            "smooth_ds": self.smooth_ds,
            "max_path_curvature": self.max_path_curvature,
            "record_trajectory": self.record_trajectory,
            "record_telemetry": self.record_telemetry,
            "sensor": _asdict(self.sensor),
            "disturbance": _asdict(self.disturbance),
            "vehicle": _asdict(self.vehicle),
            "conditions": [c.to_dict() for c in self.conditions],
            "source_path": self.source_path,
        }

    def with_phase(self, phase: str) -> "ExperimentProtocol":
        return replace(self, phase=phase)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentProtocol":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        conditions = [
            Condition.from_dict(item) for item in data.get("conditions", [])
        ]
        sensor_data = dict(data.get("sensor", {}))
        disturbance_data = dict(data.get("disturbance", {}))
        vehicle_data = dict(data.get("vehicle", {}))
        return cls(
            phase=str(data.get("phase", "tune")),
            conditions=conditions,
            n_seeds=int(data.get("n_seeds", 10)),
            scenarios=list(
                data.get("scenarios", [s.name for s in default_scenarios()])
            ),
            seeds=data.get("seeds"),
            dt=float(data.get("dt", 0.05)),
            substeps=int(data.get("substeps", 4)),
            smoothing=bool(data.get("smoothing", True)),
            smooth_ds=float(data.get("smooth_ds", 0.02)),
            max_path_curvature=float(data.get("max_path_curvature", 4.0)),
            record_trajectory=bool(data.get("record_trajectory", False)),
            record_telemetry=bool(data.get("record_telemetry", False)),
            sensor=SensorConfig(**sensor_data),
            disturbance=DisturbanceConfig(**disturbance_data),
            vehicle=VehicleParameters(**vehicle_data),
            source_path=str(path),
        )

    def run(
        self,
        output_dir: str | Path | None = None,
        *,
        max_episodes: int | None = None,
        progress: bool = True,
    ) -> "RunArtifacts":
        self.validate()
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError("max_episodes must be positive")
            if self.phase == "test":
                raise ValueError(
                    "a test-phase run must execute the full frozen protocol; "
                    "max_episodes is not allowed"
                )
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = (
                PROJECT_ROOT / "experiments" / "results" / self.phase / stamp
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        snapshot = yaml.safe_dump(
            self.to_dict(), sort_keys=False, allow_unicode=True
        )
        (out / "config_snapshot.yaml").write_text(snapshot, encoding="utf-8")
        (out / "environment.txt").write_text(
            _environment_text(), encoding="utf-8"
        )

        results: list[EpisodeResult] = []
        started = time.perf_counter()
        episodes_path = out / "episodes.jsonl"
        with episodes_path.open("w", encoding="utf-8") as fh:
            for i, cfg in enumerate(self.iter_episodes()):
                if max_episodes is not None and i >= max_episodes:
                    break
                result = run_episode(cfg)
                results.append(result)
                fh.write(
                    json.dumps(
                        result.to_dict(),
                        default=_json_default,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()
                if progress and (i + 1) % 10 == 0:
                    print(
                        f"[{self.phase}] {i + 1} episodes "
                        f"({result.condition}/{result.scenario}/"
                        f"seed={result.seed})",
                        flush=True,
                    )

        if not results:
            raise RuntimeError("no episodes were executed")
        summary = build_summary(results)
        summary["wall_time_s"] = time.perf_counter() - started
        summary["protocol"] = {
            "phase": self.phase,
            "config_sha256": hashlib.sha256(
                snapshot.encode("utf-8")
            ).hexdigest(),
            "source_path": self.source_path,
            "n_seeds": self.n_seeds,
            "seeds": self.resolved_seeds,
            "scenarios": list(self.scenarios),
            "conditions": [c.name for c in self.conditions],
        }
        summary_path = out / "summary.json"
        write_summary_json(summary, summary_path)
        csv_path = out / "summary.csv"
        write_summary_csv(summary, csv_path)
        report_path = out / "report.md"
        write_report(summary, report_path)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "phase": self.phase,
                    "config_sha256": summary["protocol"]["config_sha256"],
                    "n_episodes": len(results),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "conditions": [c.name for c in self.conditions],
                    "scenarios": list(self.scenarios),
                    "seeds": self.resolved_seeds,
                    "environment": _environment_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_checksums(out)
        return RunArtifacts(
            output_dir=out,
            episodes_path=episodes_path,
            summary_path=summary_path,
            csv_path=csv_path,
            report_path=report_path,
            n_episodes=len(results),
        )


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    output_dir: Path
    episodes_path: Path
    summary_path: Path
    csv_path: Path
    report_path: Path
    n_episodes: int


def _asdict(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _environment_dict() -> dict[str, str]:
    commit = "not a git repository"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    except Exception:
        pass
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "git_commit": commit,
    }


def _environment_text() -> str:
    data = _environment_dict()
    lines = [f"{k}: {v}" for k, v in data.items()]
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines.append("pip_freeze:")
    lines.extend(f"  {line}" for line in proc.stdout.splitlines())
    return "\n".join(lines) + "\n"


def _write_checksums(out: Path) -> None:
    lines: list[str] = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def freeze_protocol(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    phase_override: str | None = None,
) -> dict[str, Any]:
    """Freeze a protocol configuration before the test phase."""

    config_path = Path(config_path)
    protocol = ExperimentProtocol.from_yaml(config_path)
    if phase_override is not None:
        protocol = protocol.with_phase(phase_override)
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "config_path": str(config_path),
        "config_sha256": digest,
        "phase": protocol.phase,
        "n_seeds": protocol.n_seeds,
        "seeds": protocol.resolved_seeds,
        "scenarios": list(protocol.scenarios),
        "conditions": [c.name for c in protocol.conditions],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment_dict(),
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def check_test_authorization(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    authorize: bool,
    phase_override: str | None = None,
) -> None:
    """Refuse a test run unless the frozen manifest matches the config."""

    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    protocol = ExperimentProtocol.from_yaml(config_path)
    if phase_override is not None:
        protocol = protocol.with_phase(phase_override)
    if protocol.phase != "test":
        return
    if not authorize:
        raise RuntimeError(
            "test phase requires --authorize-test and a frozen protocol manifest"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"missing frozen protocol manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest.get("config_sha256") != digest:
        raise RuntimeError(
            "test configuration does not match the frozen manifest; "
            "do not modify the protocol after validation"
        )
