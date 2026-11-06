"""Unit tests for the calibrated CoppeliaSim force model."""

from unittest.mock import MagicMock

import pytest

from iara_v4.core.types import ControlCommand
from iara_v5a.hal.coppelia import CoppeliaHAL


def _hal_with_sim(sim: MagicMock) -> CoppeliaHAL:
    hal = CoppeliaHAL()
    hal.sim = sim
    hal._initialized = True
    hal._handles = {"boat": 7, "thruster_left": 1, "thruster_right": 2}
    hal._propeller_radius = 0.02
    hal._track_width = 0.132
    return hal


def test_apply_command_side_force_opposes_lateral_velocity():
    """A zero-motion command must still brake an existing sideslip."""

    lateral_speed = 0.4
    sim = MagicMock()
    sim.getObjectPosition.return_value = [0.0, 0.0, 0.0]
    sim.getObjectOrientation.return_value = [0.0, 0.0, 0.0]
    sim.getObjectVelocity.return_value = (
        [0.0, lateral_speed, 0.0],
        [0.0, 0.0, 0.0],
    )
    sim.getSimulationTime.return_value = 0.0
    hal = _hal_with_sim(sim)

    hal.apply_command(ControlCommand(v=0.0, omega=0.0))

    sim.addForceAndTorque.assert_called_once()
    call_args = sim.addForceAndTorque.call_args.args
    force = call_args[1]
    force_forward, force_side = float(force[0]), float(force[1])

    assert force_forward == pytest.approx(0.0)
    assert force_side * lateral_speed < 0.0
    assert force_side != 0.0
