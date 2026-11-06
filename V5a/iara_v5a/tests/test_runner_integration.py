import pytest

from iara_v4.core.types import Measurement, Pose2D, Vec2
from iara_v4.simulation.runner import EpisodeConfig
from iara_v4.simulation.world import Scenario
from iara_v5a.hal.base import HALConfig, HardwareHAL
from iara_v5a.runner.episode import run_episode_v5a


def test_fake_hal_has_the_runtime_contract():
    class FakeHAL:
        name = "fake"

        def reset(self, config: HALConfig):
            return Measurement(0.0, 0.0, Pose2D(0.0, 0.0), 0.0, 0.0, 0.0)

        def read_measurement(self):
            return self.reset(HALConfig())

        def apply_command(self, command):
            self.command = command

        def step(self, dt):
            self.time = dt

        def is_running(self):
            return getattr(self, "time", 0.0)

        def close(self):
            pass

    assert isinstance(FakeHAL(), HardwareHAL)


class _SpeedSequenceHAL:
    """HAL stub that reports a scripted sequence of surge speeds."""

    name = "speed-sequence"

    def __init__(self, speeds):
        self._speeds = list(speeds)
        self._index = 0
        self._time = 0.0

    def _measurement(self):
        speed = self._speeds[min(self._index, len(self._speeds) - 1)]
        return Measurement(
            time=self._time,
            source_time=self._time,
            pose=Pose2D(0.0, 0.0, 0.0),
            u=speed,
            v=0.0,
            r=0.0,
        )

    def reset(self, config):
        self._index = 0
        self._time = 0.0
        return self._measurement()

    def read_measurement(self):
        return self._measurement()

    def apply_command(self, command):
        pass

    def step(self, dt):
        self._index += 1
        self._time += dt

    def is_running(self):
        return self._time

    def close(self):
        pass


def test_max_speed_records_the_peak_surge_speed():
    scenario = Scenario(
        name="open_water",
        start=Pose2D(0.0, 0.0, 0.0),
        goal=Vec2(3.5, 0.0),
        obstacles=(),
        bounds=(-0.5, -1.0, 4.0, 1.0),
        timeout=0.15,
        goal_tolerance=0.08,
    )
    cfg = EpisodeConfig(
        scenario=scenario,
        condition="pp",
        seed=20,
        phase="validation",
        dt=0.05,
        substeps=1,
        record_path=True,
    )
    hal = _SpeedSequenceHAL([0.0, 0.9, 2.5, 0.4])

    result = run_episode_v5a(cfg, hal)

    assert result.metrics.max_speed == pytest.approx(2.5)
    assert result.metrics.time == pytest.approx(0.15)
