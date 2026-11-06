import numpy as np
import pytest

from iara_v4.core.types import ControlCommand, WheelSpeeds
from iara_v5a.hal.coppelia import CoppeliaHAL, CoppeliaHALConfig
from iara_v5a.hal.base import HALError


class _RecordingSim:
    def __init__(self):
        self.calls = []

    def getObjectPosition(self, handle, relative_to):
        return [0.0, 0.0, 0.0]

    def getObjectOrientation(self, handle, relative_to):
        return [0.0, 0.0, 0.0]

    def getObjectVelocity(self, handle):
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def getSimulationTime(self):
        return 0.0

    def setJointTargetVelocity(self, handle, value):
        self.calls.append(("setJointTargetVelocity", handle, value))

    def addForceAndTorque(self, handle, force, torque, relative_to):
        self.calls.append(("addForceAndTorque", handle, force, torque, relative_to))


def test_coppelia_hal_construction_is_lazy():
    hal = CoppeliaHAL()
    assert hal.sim is None
    assert CoppeliaHALConfig(scene_path=None).dt == 0.05


def test_coppelia_defaults_use_requested_scene_names():
    cfg = CoppeliaHALConfig()
    assert cfg.boat_path == "/Cuboid"
    assert cfg.thruster_left_path == (
        "/Cuboid/Dummy_thrusterLeft/thrusterJoint_left"
    )
    assert cfg.thruster_right_path == (
        "/Cuboid/Dummy_thrusterRight/thrusterJoint_right"
    )


def test_missing_scene_is_reported_before_connection():
    hal = CoppeliaHAL()
    try:
        hal.reset(CoppeliaHALConfig(scene_path="does-not-exist.ttt"))
    except HALError as exc:
        assert "scene not found" in str(exc)
    else:
        raise AssertionError("missing scene must raise HALError")


def _configured_hal():
    hal = CoppeliaHAL()
    hal.sim = _RecordingSim()
    hal._initialized = True
    hal._handles = {"boat": 7, "thruster_left": 1, "thruster_right": 2}
    hal._propeller_radius = 0.02
    hal._track_width = 0.132
    return hal


def test_apply_command_with_wheels_calls_both_thrusters():
    hal = _configured_hal()
    hal.apply_command(
        ControlCommand(v=0.2, omega=0.0, wheels=WheelSpeeds(0.2, 0.2))
    )
    joint_calls = [
        call for call in hal.sim.calls if call[0] == "setJointTargetVelocity"
    ]
    assert len(joint_calls) == 2
    assert [call[1] for call in joint_calls] == [1, 2]
    assert [call[2] for call in joint_calls] == [10.0, 10.0]
    assert hal.sim.calls[-1][0] == "addForceAndTorque"


def test_apply_command_without_wheels_uses_differential_fallback():
    hal = _configured_hal()
    hal.apply_command(ControlCommand(v=0.2, omega=0.1))
    joint_calls = [
        call for call in hal.sim.calls if call[0] == "setJointTargetVelocity"
    ]
    left_omega = joint_calls[0][2]
    right_omega = joint_calls[1][2]
    assert left_omega == pytest.approx(9.67)
    assert right_omega == pytest.approx(10.33)


def test_missing_handle_raises_hal_error():
    class _Sim:
        def getObject(self, path):
            raise Exception(f"not found: {path}")

    hal = CoppeliaHAL()
    hal.sim = _Sim()
    hal._paths = {
        "boat": "/boat",
        "thruster_left": "/left",
        "thruster_right": "/right",
    }
    with pytest.raises(HALError, match="missing CoppeliaSim object"):
        hal._discover_handles()


def test_set_disturbance_calls_simulation_api():
    hal = _configured_hal()
    hal.set_disturbance(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert hal.sim.calls == [
        ("addForceAndTorque", 7, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -1)
    ]
