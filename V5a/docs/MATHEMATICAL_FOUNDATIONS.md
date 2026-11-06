# V4 mathematical foundations

All episode metrics are computed from the true simulated state at controller
sample times. Wall-clock measurements are diagnostics and are not physical
outcomes.

For samples `k = 1, ..., N`, sample period `dt`, path point `p_k`, vehicle
position `x_k`, heading error `e_psi,k`, cross-track error `e_y,k`,
yaw-rate command `omega_cmd,k`, applied rudder angle `delta_k`, and clearance
`c_k`:

- `tracking_rmse = sqrt((1/N) sum ||x_k - p_k||^2)`. Units: m.
- `heading_rmse = sqrt((1/N) sum e_psi,k^2)`, with wrapped angle error. Units: rad.
- `max_cross_track = max |e_y,k|`. Units: m.
- `control_energy` depends on the plant:
  - dynamic: `sum (tau_u,k^2 + tau_r,k^2) dt`;
  - kinematic with wheel speeds: `sum (left_k^2 + right_k^2) dt`;
  - kinematic without wheel speeds: `sum (v_k^2 + omega_k^2) dt`.
  These definitions are not comparable across plants.
- `smoothness_cmd` is the RMS derivative of `omega_cmd,k`, in rad s^-2.
- `smoothness_actuator` is the RMS derivative of `delta_k`, in rad s^-1.
- `smoothness` is a deprecated alias equal to `smoothness_actuator`.
- `min_clearance = min c_k`, where negative means footprint intersection,
  zero means contact, and positive means separation.
- `saturation_fraction = (1/N) sum 1[actuator saturation at k]`.

Success, collision, timeout, planning failure, and out-of-bounds termination
are auditable categorical outcomes. Wall-clock times are excluded from control
performance comparisons.
