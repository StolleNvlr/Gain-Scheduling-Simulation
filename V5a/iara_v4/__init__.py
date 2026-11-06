"""IARA V4: dynamic-control research stack.

The package is self-contained with respect to V3: the V3 kinematic plant is
preserved as ``KinematicUSVPlant``, but V4 does not import the V3 package.
Install-free use from the repository root::

    import sys
    sys.path.insert(0, "V4")
    import iara_v4

The public version string identifies the internal iteration.  It is not a
release version and does not imply physical validation.
"""

from __future__ import annotations

__version__ = "4.0.0"

from .control import (
    GainScheduledController,
    GainScheduler,
    MRACConfig,
    MRACController,
    PurePursuit,
    PurePursuitConfig,
    ReferenceModel,
)
from .dynamics import (
    DynamicPlantConfig,
    DynamicUSVPlant,
    KinematicUSVPlant,
    VehicleParameters,
    physical_checks,
)
from .simulation import (
    CONDITION_SPECS,
    EpisodeConfig,
    EpisodeMetrics,
    EpisodeResult,
    run_episode,
    scenario_by_name,
)

__all__ = [
    "__version__",
    "GainScheduledController",
    "GainScheduler",
    "MRACConfig",
    "MRACController",
    "PurePursuit",
    "PurePursuitConfig",
    "ReferenceModel",
    "DynamicPlantConfig",
    "DynamicUSVPlant",
    "KinematicUSVPlant",
    "VehicleParameters",
    "physical_checks",
    "CONDITION_SPECS",
    "EpisodeConfig",
    "EpisodeMetrics",
    "EpisodeResult",
    "run_episode",
    "scenario_by_name",
]
