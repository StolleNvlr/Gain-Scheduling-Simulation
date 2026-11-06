"""Manual V5a1 smoke test; requires CoppeliaSim and the configured scene."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iara_v4.core.types import ControlCommand, WheelSpeeds
from iara_v5a.hal.base import HALError
from iara_v5a.hal.coppelia import CoppeliaHAL, CoppeliaHALConfig


def main() -> int:
    print("=== V5a1 smoke test ===")
    print("Prerequisites:")
    print("  1. CoppeliaSim running with the V2 scene loaded")
    print("  2. ZMQ Remote API server active (default port 23000)")
    print("  3. Handles in CoppeliaHALConfig match the scene")
    print()
    config = CoppeliaHALConfig(
        scene_path=str(
            ROOT.parent
            / "V2_0"
            / "IAR-CPS-0001-26- CENA DO IARA LUA V0.3.ttt"
        )
    )
    hal = CoppeliaHAL(host=config.host, port=config.port)
    try:
        measurement = hal.reset(config)
    except HALError as exc:
        print(f"CONNECTION FAILED: {exc}", file=sys.stderr)
        print(
            "Run scripts/v5a1_diagnose_scene.py to see actual object paths.",
            file=sys.stderr,
        )
        return 2
    m0 = measurement
    print(
        f"t=0.00  x={measurement.pose.x:+.3f} "
        f"y={measurement.pose.y:+.3f}"
    )
    try:
        for index in range(30):
            hal.apply_command(
                ControlCommand(
                    v=0.2, omega=0.0, wheels=WheelSpeeds(left=0.2, right=0.2)
                )
            )
            hal.step(config.dt)
            if index % 10 == 0:
                measurement = hal.read_measurement()
                print(
                    f"t={measurement.time:.2f}  x={measurement.pose.x:+.3f} "
                    f"y={measurement.pose.y:+.3f}  u={measurement.u:+.3f}"
                )
        dx = abs(measurement.pose.x - m0.pose.x)
        if dx < 0.001:
            print(
                f"BOAT DID NOT MOVE (dx={dx:.4f}); check thruster joints, "
                "velocity mode, and propeller_radius",
                file=sys.stderr,
            )
            return 3
        for index in range(40):
            hal.apply_command(
                ControlCommand(
                    v=0.2, omega=0.0, wheels=WheelSpeeds(left=0.2, right=0.2)
                )
            )
            hal.step(config.dt)
            if index % 10 == 0:
                measurement = hal.read_measurement()
                print(
                    f"t={measurement.time:.2f}  x={measurement.pose.x:+.3f} "
                    f"y={measurement.pose.y:+.3f}  u={measurement.u:+.3f}"
                )
        for index in range(50):
            hal.apply_command(ControlCommand(v=0.2, omega=0.0))
            hal.step(config.dt)
            if index % 10 == 0:
                measurement = hal.read_measurement()
                print(
                    f"t={measurement.time:.2f}  x={measurement.pose.x:+.3f} "
                    f"y={measurement.pose.y:+.3f}  u={measurement.u:+.3f}"
                )
    finally:
        hal.close()
    print("OK: smoke command paths completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
