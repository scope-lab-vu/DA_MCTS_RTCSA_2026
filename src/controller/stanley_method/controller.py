"""
Stanley path-following controller.

Faithful port of the reference at
    https://github.com/CL2-UWaterloo/f1tenth_ws/blob/main/src/stanley_avoidance/stanley_avoidance/stanley_avoidance.py

Differences from the ROS2 reference are *only* I/O:
  • The reference publishes `(steering_angle, speed)` directly (Ackermann).
    Our codebase's `ControlCmdMsg` is `(acceleration, steering_vel)`. We
    convert: `sv = (angle_target − state.delta) / dt` and
    `accel = K_p * (target_speed − state.v)`.
  • The reference loads waypoints from a comma-CSV with `(x, y, v)` columns;
    we use `src.track.ReferenceTrajectory` which loads our TUM-format
    semicolon-CSV and exposes `.positions[:,:2]` and `.velocities`.

Algorithm (verbatim from the reference, see drive_to_target_stanley):
  1. Front-axle pose = base_link + 0.33 m along heading (full wheelbase).
  2. Closest raceline waypoint to the REAR pose → target_velocity.
  3. Closest raceline waypoint to the FRONT pose → in world AND car frames.
  4. path_heading = atan2(front_wpt.y − rear_wpt.y, front_wpt.x − rear_wpt.x).
  5. current_heading = direction from base_link → front-axle (= state.psi).
  6. Wrap both to [0, 2π], then heading_err = wrap_to_pi(path − current).
  7. crosstrack = atan2(K_E · front_pt_car_y, K_V + target_velocity)   # K_V = 0
  8. angle = K_H · heading_err + crosstrack, clip to ±steering_limit (degrees).
  9. velocity = target_velocity · velocity_percentage.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import ReferenceTrajectory

log = logging.getLogger(__name__)


class StanleyMethodController:
    """Pure Stanley path-tracker matching the CL2-UWaterloo reference."""

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig | None = None,
        simulation_config: SimulationConfig | None = None,
    ):
        self._vehicle_params = vehicle_params
        self.dt = simulation_config.dt_s if simulation_config is not None else 0.04

        # ── Reference parameters (defaults from CL2-UWaterloo) ─────────────
        self.K_E = getattr(config, "K_E", 2.0)              # cross-track gain
        self.K_H = getattr(config, "K_H", 1.5)              # heading gain
        self.K_p = getattr(config, "K_p_speed", 0.5)        # speed P gain
        self.velocity_percentage = getattr(config, "velocity_percentage", 1.0)
        # The reference's `steering_limit` is in DEGREES (default 25).
        self.steering_limit_deg = getattr(config, "steering_limit", 25.0)
        # If the user passed it in radians (≤ ~π), convert by detection.
        if self.steering_limit_deg < 1.6:  # pragma: no cover
            self.steering_limit_deg = math.degrees(self.steering_limit_deg)

        # ── Vehicle geometry ───────────────────────────────────────────────
        # F1TENTH gym puts (x, y) at the REAR axle. Front axle is +L along
        # heading, where L = lf + lr (= 0.33 m for stock F1TENTH; matches the
        # reference's literal "33cm in front of base_link").
        self.L = float(vehicle_params.lf + vehicle_params.lr)

        # Output-conversion limits.
        self.sv_min = float(vehicle_params.sv_min)
        self.sv_max = float(vehicle_params.sv_max)
        self.a_max = float(vehicle_params.a_max)

        # ── Raceline ───────────────────────────────────────────────────────
        if map_config is None or not getattr(map_config, "wpt_path", None):
            raise ValueError(
                "StanleyMethodController requires map_config.wpt_path "
                "(the raceline CSV referenced by config_<map>_map.yaml)."
            )
        self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
        self.waypoints_world = np.asarray(self.ref_traj.positions, dtype=np.float64)
        self.velocities = np.asarray(self.ref_traj.velocities, dtype=np.float64)

        log.info(
            "Stanley ready: %d raceline points  L=%.3fm  K_E=%.2f K_H=%.2f K_p=%.2f"
            "  v_percentage=%.2f steering_limit=%.1f° dt=%.3fs",
            len(self.waypoints_world), self.L, self.K_E, self.K_H, self.K_p,
            self.velocity_percentage, self.steering_limit_deg, self.dt,
        )

    # ------------------------------------------------------------------ API

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
        remaining_budget_s: float = 0.0,
        **_kwargs,
    ) -> ControlCmdMsg:
        # 1. Front-axle (wheelbase-front) pose.
        cos_p = math.cos(state.psi); sin_p = math.sin(state.psi)
        x_front = state.x + self.L * cos_p
        y_front = state.y + self.L * sin_p

        # 2. Closest raceline point to the REAR pose → target_velocity.
        rear_idx = self._closest_waypoint_idx(state.x, state.y)
        rear_pt_world = self.waypoints_world[rear_idx]
        target_velocity = float(self.velocities[rear_idx])

        # 3. Closest raceline point to the FRONT pose → in world AND car frame.
        front_idx = self._closest_waypoint_idx(x_front, y_front)
        front_pt_world = self.waypoints_world[front_idx]
        # Transform to car frame (rotate by −ψ around the front-axle origin).
        dx = front_pt_world[0] - x_front
        dy = front_pt_world[1] - y_front
        front_pt_car_y = -sin_p * dx + cos_p * dy

        # 4. Path heading: direction from rear waypoint to front waypoint.
        path_heading = math.atan2(
            front_pt_world[1] - rear_pt_world[1],
            front_pt_world[0] - rear_pt_world[0],
        )

        # 5. Current heading: direction from base_link to wheelbase-front.
        # That's literally state.psi (since we placed front-axle at +L along ψ).
        current_heading = math.atan2(y_front - state.y, x_front - state.x)

        # 6. Wrap both to [0, 2π], compute diff, wrap to [−π, π].
        if current_heading < 0.0:  current_heading += 2.0 * math.pi
        if path_heading < 0.0:     path_heading += 2.0 * math.pi
        heading_error = path_heading - current_heading
        if heading_error > math.pi:    heading_error -= 2.0 * math.pi
        elif heading_error < -math.pi: heading_error += 2.0 * math.pi

        # 7. Stanley terms (K_V = 0 hardcoded in the reference).
        K_V = 0.0
        crosstrack_error = math.atan2(
            self.K_E * front_pt_car_y,
            K_V + target_velocity,
        )
        heading_error *= self.K_H

        # 8. Final steering angle, clipped to ±steering_limit (degrees → rad).
        angle_target = heading_error + crosstrack_error
        steering_limit_rad = math.radians(self.steering_limit_deg)
        angle_target = float(np.clip(angle_target, -steering_limit_rad, steering_limit_rad))

        # 9. Velocity command (raw raceline × percentage).
        target_speed = target_velocity * self.velocity_percentage

        # ── I/O conversion ─────────────────────────────────────────────────
        # Steering: one-step finite difference. Gym will rate-limit if
        # demand exceeds sv_max — that's the gym's job, not ours.
        sv = (angle_target - state.delta) / self.dt
        sv = float(np.clip(sv, self.sv_min, self.sv_max))

        # Speed: P-controller on velocity error → acceleration.
        accel = self.K_p * (target_speed - state.v)
        accel = float(np.clip(accel, -self.a_max, self.a_max))

        return ControlCmdMsg(acceleration=accel, steering_vel=sv)

    # ----------------------------------------------------------- internals

    def _closest_waypoint_idx(self, x: float, y: float) -> int:
        """Index of the raceline waypoint nearest to world point (x, y).
        Mirrors `get_closest_waypoint_with_velocity` and `get_waypoint_stanley`
        in the reference (both reduce to argmin distance after the rigid
        transform — which preserves distances)."""
        diff = self.waypoints_world - np.array([x, y])
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        return int(np.argmin(d2))

    def reset(self) -> None:
        """No internal state to reset (Stanley is memoryless)."""
        pass

    # Convenience accessors used by the experiment harness.
    def get_best_trajectory(self) -> list:
        return []

    def get_last_weights(self) -> tuple:
        return (1 / 3, 1 / 3, 1 / 3)
