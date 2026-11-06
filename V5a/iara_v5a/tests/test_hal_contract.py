import pytest

from iara_v5a.hal.base import HALError, HardwareHAL


def test_hal_protocol_is_runtime_checkable():
    class MockHAL:
        name = "mock"

        def reset(self, cfg): ...

        def read_measurement(self): ...

        def apply_command(self, cmd): ...

        def step(self, dt): ...

        def is_running(self):
            return 0.0

        def close(self): ...

    assert isinstance(MockHAL(), HardwareHAL)


def test_hal_error_is_runtime_error():
    with pytest.raises(RuntimeError):
        raise HALError("test")
