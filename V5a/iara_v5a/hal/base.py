"""Hardware abstraction contract shared by simulated and physical plants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from iara_v4.core.types import ControlCommand, Measurement


class HALError(RuntimeError):
    """Raised when a plant cannot fulfil a HAL operation."""


@runtime_checkable
class HardwareHAL(Protocol):
    """Contract implemented by CoppeliaSim, Raspberry Pi, and real hardware."""

    name: str

    def reset(self, config: "HALConfig") -> Measurement: ...

    def read_measurement(self) -> Measurement: ...

    def apply_command(self, command: ControlCommand) -> None: ...

    def step(self, dt: float) -> None: ...

    def is_running(self) -> float: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HALConfig:
    """Configuration common to all plant implementations."""

    dt: float = 0.05
    timeout: float = 60.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.timeout <= 0.0:
            raise ValueError("timeout must be positive")
