"""Reference reconstruction, audit and experiment orchestration."""

from .protocol import (
    Condition,
    ExperimentProtocol,
    RunArtifacts,
    check_test_authorization,
    freeze_protocol,
)
from .report import build_summary, write_report
from .statistics import (
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    competing_risks,
    holm_correction,
    mcnemar,
    wilcoxon,
    wilson_ci,
)

__all__ = [
    "Condition",
    "ExperimentProtocol",
    "RunArtifacts",
    "check_test_authorization",
    "freeze_protocol",
    "build_summary",
    "write_report",
    "bootstrap_ci",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "competing_risks",
    "holm_correction",
    "mcnemar",
    "wilcoxon",
    "wilson_ci",
]
