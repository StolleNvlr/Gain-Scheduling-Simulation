# V5a1 to V5a2 handoff

## Status at the end of V5a1

The HAL layer is ready and was validated in CoppeliaSim:

- `iara_v5a/hal/base.py` defines the runtime-checkable `HardwareHAL` contract
  (`reset`, `read_measurement`, `apply_command`, `step`, `is_running`,
  `close`) and the shared `HALConfig`.
- `iara_v5a/hal/coppelia.py` implements that contract with the ZMQ Remote API,
  synchronous stepping, body-frame velocity measurements, and calibrated
  virtual forces.
- `iara_v5a/runner/episode.py` accepts the HAL by dependency injection, so the
  V4 planner, guidance, controllers, metrics, and statistical protocol are
  reused without a plant-specific branch.
- `iara_v5a/runner/protocol.py` repeats each `(condition, scenario, seed)`
  cell and writes both `v5a1_protocol.json` and the flat `episodes.jsonl`
  consumed by `scripts/generate_figures.py`.

Reference validation: `pp` on `coppelia_default` with seed 11 reached the goal
with `success: true`, `timeout: false`, and `tracking_rmse: 0.030 m`.

Hard constraints for the next agent:

1. Do **not** modify `iara_v4/`; it is frozen.
2. Do **not** change the calibrated `force_forward`, `force_side`, or
   `torque_z` gains/limits in `CoppeliaHAL.apply_command` without a new
   calibration and a new validation record. See `V5A1_LIMITATIONS.md`.
3. Keep the HAL dependency-injection boundary. The runner must not import
   CoppeliaSim directly.

## Task 1 - implement `PiHAL`

Create `iara_v5a/hal/pi.py` (or an equivalent module) implementing the same
`HardwareHAL` methods:

- `reset(config)`: open the serial/CAN/UART links, configure the motor driver,
  IMU, compass, and GPS, and return the first `Measurement`.
- `read_measurement()`: build a `Measurement` with the same body-frame
  convention as `CoppeliaHAL` (`u` surge, `v` sway, `r` yaw rate), a monotonic
  `time`, and the correct `source_time`.
- `apply_command(command)`: map `command.v`/`command.omega` (or
  `command.wheels`) to the left/right PWM or motor-driver setpoints. Keep the
  differential-drive conversion in one place and document the sign convention.
- `step(dt)`, `is_running()`, `close()`: respect the real-time loop, return the
  elapsed time, and release the hardware safely.

Acceptance criteria:

- unit tests with a mocked serial/GPIO layer (no Raspberry Pi required in CI);
- the same `HardwareHAL` runtime-contract test used by V5a1;
- a short bench log showing commands and measured yaw rate in the correct sign;
- no changes to the V5a1 runner or to the calibrated CoppeliaSim forces.

## Task 2 - insert the EKF

Use the reserved `iara_v5a/estimation/` package. Implement a predictor/corrector
with at least:

- state `[x, y, psi, u, v, r]` (or the reduced state justified in the report);
- `predict(dt, command)` using the V4 kinematic/dynamic model;
- `update(measurement)` fusing IMU yaw rate, compass heading, GPS position, and
  surge speed;
- covariance propagation, innovation covariance, and a documented outlier gate
  (for example Mahalanobis or NIS);
- telemetry fields for covariance trace, accepted/rejected updates, and NIS.

Pipeline change: `raw Measurement -> EKF prediction -> EKF update -> controller`.
Keep the raw measurement available for diagnostics and keep the controller
interface unchanged. Validate first on recorded V5a1 logs (replay), then on a
CoppeliaSim run, and only then on `PiHAL`. Report whether the EKF improves
`tracking_rmse`/`heading_rmse` without hiding sensor dropouts.

Acceptance criteria:

- deterministic replay test with a fixed log;
- covariance stays positive definite and finite;
- dropout/latency tests show the filter degrades gracefully;
- the comparison against the no-EKF V5a1 baseline is paired and repeated.

## Task 3 - enable V4 VFH+ with CoppeliaSim proximity sensors

V4's VFH+ is already implemented in `iara_v4/planning/vfh.py` and consumes a
`Scan` with world-frame angles. V5a1 leaves it inactive because no range data
are read. To activate it:

1. Add proximity sensors to the CoppeliaSim scene (or discover them by path).
2. Read each sensor with `sim.readProximitySensor(handle)`; the API returns
   `(result, distance, detected_point, detected_object_handle, detected_normal)`.
3. Convert the sensor geometry to a `Scan`: angles must be world-frame
   (`pose.theta + sensor_mount_angle`), distances clipped to
   `VFHConfig.max_range`, and "no detection" must become `inf`/`max_range` as
   the V4 scanner expects.
4. Feed the resulting `Scan` into the same `VFHPlus.update(scan, preferred_heading)`
   call used by V4, and record `vfh_blocked_fraction` and the selected heading.
5. Add a `vfh` condition to the V5a2 protocol, preferably on
   `coppelia_hidden` (mapped cuboids plus an unmapped block) so the local
   planner has something to react to.

Acceptance criteria:

- a replay/unit test that a synthetic scan produces the expected VFH+ heading;
- a CoppeliaSim run where VFH+ changes the commanded heading before contact;
- the no-VFH and VFH+ conditions use the same global plan and seed;
- sensor noise, `max_range`, and mounting angle are documented.

## Recommended order

1. `PiHAL` with mocks and a bench smoke test.
2. EKF replay on V5a1 logs, then closed-loop in CoppeliaSim.
3. Proximity-sensor adapter, then VFH+ ablation.
4. Update `V5A2_LIMITATIONS.md` and the protocol manifest before claiming any
   result.
