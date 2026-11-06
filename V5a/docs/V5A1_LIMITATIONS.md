# V5a1 limitations

V5a1 reintegrates CoppeliaSim as an injected 3D plant while keeping the frozen
V4 planning, guidance, control, and metric definitions. The HAL contract is
`HardwareHAL`; the CoppeliaSim implementation is `CoppeliaHAL`. The default
scene contract is `/Cuboid`, `/Cuboid/Dummy_thrusterLeft/thrusterJoint_left`,
and `/Cuboid/Dummy_thrusterRight/thrusterJoint_right`, with both thrusters in
velocity mode.

## 1. CoppeliaSim is not bit-for-bit deterministic

The physics engine, contact solver, and rendering loop do not guarantee the same
floating-point trajectory for two runs with the same nominal seed. V5a1
therefore repeats every `(condition, scenario, seed)` cell `n_repetitions`
times and reports the mean and sample standard deviation across repetitions.
Comparisons remain paired by `(condition, scenario, seed)`, but an individual
episode file cannot be reproduced exactly. A single run is not evidence of a
stable difference between controllers; use the repeated-cell aggregate.

## 2. The water is represented by a solid floor tile

The CoppeliaSim scene does not simulate a fluid. The boat is supported by a
solid 50 m x 50 m floor tile that plays the role of the water surface. There is
no buoyancy solver, no free-surface deformation, no waves, and no
hydrodynamic-added-mass model. Contacts and friction with the tile, not fluid
pressure, hold the boat up. Consequently, absolute accelerations, turning
radius, and stopping distance are properties of this scene approximation and
must not be reported as identified IARA hydrodynamic parameters.

## 3. The propulsion and steering forces are empirical simulation hypotheses

Because there is no fluid, the calibrated `CoppeliaHAL.apply_command` applies
three virtual body-frame effects directly with `sim.addForceAndTorque`:

- `force_forward`: surge drive/brake force;
- `force_side`: lateral resistance that opposes sideslip and keeps the boat
  from skidding sideways;
- `torque_z`: yaw torque used to track the commanded angular rate.

`force_forward` and `force_side` are **not** measured propeller curves and are
not real buoyancy or fluid-drag forces. They are simulation hypotheses
calibrated empirically by hand as a virtual PID around the current surge and
sway speeds, because the scene has no real fluid. The same applies to
`torque_z` for yaw. Their gains, sign conventions, and saturation limits
(currently +/-60 N for the linear forces and +/-0.5 N m for the yaw torque) are
part of the calibrated plant contract. Changing them changes the plant and
invalidates the validation figures; do not tune them without re-running the
CoppeliaSim validation and documenting the new calibration.

## 4. Scope not covered by V5a1

V5a1 does not implement a Raspberry Pi HAL (`PiHAL`), an EKF, real-sensor VFH+,
or physical validation. The V4 VFH+ layer is present but inactive because V5a1
does not yet read range sensors from CoppeliaSim. The `set_disturbance` method
is wired to the force/torque API, but disturbance rejection must not be claimed
until a dedicated CoppeliaSim run demonstrates the deviation and the controller
response.

The reference V5a1 CoppeliaSim smoke/validation episode used `pp` on
`coppelia_default` with seed 11: `success: true`, `timeout: false`, and
`tracking_rmse: 0.030 m`. This is a scene-calibration result, not a physical
validation of the vessel.
