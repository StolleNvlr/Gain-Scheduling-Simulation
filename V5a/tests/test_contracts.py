from __future__ import annotations

from iara_v4.core.types import ControlCommand


def test_control_command_is_platform_neutral() -> None:
    command = ControlCommand(v=0.3, omega=0.2)
    assert command.wheels is None
    assert command.as_tuple()[:2] == (0.3, 0.2)

