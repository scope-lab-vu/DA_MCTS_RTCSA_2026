"""
LQR lateral path-follower — faithful port of f1tenth_planning's LQRPlanner.

Source:
  https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/lqr/lqr.py
  https://github.com/f1tenth/f1tenth_planning/blob/main/examples/control/lqr.py

State (4D, body-frame): x = [e_cg, ė_cg, θ_e, θ̇_e]
  • e_cg     — signed lateral cross-track error of the front axle.
  • ė_cg     — finite-difference of e_cg over `ts`.
  • θ_e      — yaw error w.r.t. raceline tangent.
  • θ̇_e      — finite-difference of θ_e over `ts`.

Discrete linearisation (kinematic bicycle, see `_update_matrix`):
  A_d = [[1, ts, 0,  0],     B_d = [[0],
         [0,  0, v,  0],            [0],
         [0,  0, 1, ts],             [0],
         [0,  0, 0,  0]]            [v / wheelbase]]

Solve discrete Riccati for K with diag(Q), diag(R) (max_iter, eps).
Steering = K · x  +  κ_ref · wheelbase   (feedback + feed-forward)

Differences from the reference (I/O only — algorithm is unchanged):
  • The reference returns (steering_angle, speed) and the example feeds
    them straight into f110-gym's `(steer, speed)` interface. Our codebase
    takes `(acceleration, steering_velocity)`, so we convert:
        sv    = (δ − state.delta) / dt        (clipped to ±sv_max)
        accel = K_p · (target_speed − state.v) (clipped to ±a_max)
  • The reference reads path heading from waypoints[:, 3]. Our TUM-format
    raceline CSV stores `psi_s` (curvilinear yaw, measured against the
    inward normal) — empirically `psi_path_tangent = psi_csv + π/2`
    (verified to ~1° rms against finite-difference tangents). We add π/2
    once at load time.
  • Reference's plan() doesn't clip the steering angle (gym MUX clips it);
    we clip to ±steering_limit as a defensive measure.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import ReferenceTrajectory

log = logging.getLogger(__name__)


def _nearest_point(point: np.ndarray, trajectory: np.ndarray):
    """Vectorized port of f1tenth_planning.utils.utils.nearest_point.

    Returns (projection, dist, t, segment_index) where the projection is
    the closest point on the piecewise-linear trajectory, with linear
    interpolation along the closest segment.
    """
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    rel = point - trajectory[:-1]
    dots = rel[:, 0] * diffs[:, 0] + rel[:, 1] * diffs[:, 1]
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1] + (t[:, None] * diffs)
    dists = np.linalg.norm(point - projections, axis=1)
    i = int(np.argmin(dists))
    return projections[i], float(dists[i]), float(t[i]), i


def _pi_2_pi(angle: float) -> float:
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi
    return angle


def _update_matrix(v: float, ts: float, wheelbase: float):
    """Discrete linearised kinematic-bicycle error dynamics (4 states, 1 input).
    Direct port of f1tenth_planning.utils.utils.update_matrix."""
    A = np.zeros((4, 4))
    A[0, 0] = 1.0
    A[0, 1] = ts
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = ts
    B = np.zeros((4, 1))
    B[3, 0] = v / wheelbase
    return A, B


def _solve_lqr(A, B, Q, R, tol: float, max_iter: int):
    """Iterative discrete-time Riccati for K = (BᵀPB + R)⁻¹ BᵀPA.
    Direct port of f1tenth_planning.utils.utils.solve_lqr."""
    AT, BT = A.T, B.T
    P = Q.copy()
    M = np.zeros((Q.shape[0], R.shape[1]))
    MT = M.T
    for _ in range(max_iter):
        P_next = (
            AT @ P @ A
            - (AT @ P @ B + M)
            @ np.linalg.pinv(R + BT @ P @ B)
            @ (BT @ P @ A + MT)
            + Q
        )
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)
    return K


class LQRController:
    """Faithful port of f1tenth_planning's LQRPlanner."""

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

        # ── LQR weights (defaults from LQRPlanner.plan() signature) ──────
        self.matrix_q = np.diag([
            float(getattr(config, "matrix_q_1", 0.999)),
            float(getattr(config, "matrix_q_2", 0.0)),
            float(getattr(config, "matrix_q_3", 0.0066)),
            float(getattr(config, "matrix_q_4", 0.0)),
        ])
        self.matrix_r = np.diag([float(getattr(config, "matrix_r", 0.75))])
        self.lqr_iter = int(getattr(config, "lqr_iterations", 50))
        self.lqr_eps = float(getattr(config, "lqr_eps", 0.001))

        # ── Speed / clipping knobs (shared with Stanley) ─────────────────
        self.K_p = float(getattr(config, "K_p_speed", 1.0))
        self.velocity_percentage = float(getattr(config, "velocity_percentage", 1.0))
        self.steering_limit_deg = float(getattr(config, "steering_limit", 24.0))
        if self.steering_limit_deg < 1.6:
            self.steering_limit_deg = math.degrees(self.steering_limit_deg)

        # ── Vehicle geometry ─────────────────────────────────────────────
        self.wheelbase = float(vehicle_params.lf + vehicle_params.lr)
        self.sv_min = float(vehicle_params.sv_min)
        self.sv_max = float(vehicle_params.sv_max)
        self.a_max = float(vehicle_params.a_max)

        # ── Raceline ─────────────────────────────────────────────────────
        if map_config is None or not getattr(map_config, "wpt_path", None):
            raise ValueError(
                "LQRController requires map_config.wpt_path "
                "(the raceline CSV referenced by config_<map>_map.yaml)."
            )
        ref_traj = ReferenceTrajectory(map_config.wpt_path)
        positions = np.asarray(ref_traj.positions, dtype=np.float64)
        velocities = np.asarray(ref_traj.velocities, dtype=np.float64)
        # TUM `psi_s` → path-tangent heading: add π/2.
        headings = np.asarray(ref_traj.headings, dtype=np.float64) + math.pi / 2.0
        curvatures = np.asarray(ref_traj.curvatures, dtype=np.float64)

        # waypoints layout matches the reference: [x, y, v, heading, kappa].
        self.waypoints = np.column_stack([positions, velocities, headings, curvatures])

        # ── Persistent error state (used for finite-difference of x[1], x[3]) ──
        # `None` until the first compute_action call; on that call we seed
        # prev := current so the first-tick derivatives are 0. The reference
        # initialises both to 0.0, which assumes the car starts exactly on
        # the path with zero heading error — true for the f1tenth_planning
        # demo but not for arbitrary spawn poses, where it produces a
        # spurious O(e/dt) "lateral velocity" feedback on tick 0.
        self._e_cg_prev: float | None = None
        self._theta_e_prev: float | None = None
        # Last target waypoint (world-frame), exposed to the renderer.
        self._last_target_pt: np.ndarray | None = None

        log.info(
            "LQR ready: %d raceline points  wheelbase=%.3fm  ts=%.3fs  "
            "Q=diag(%.4g,%.4g,%.4g,%.4g)  R=%.4g  iter≤%d  eps=%.1e  "
            "K_p=%.2f  v_percentage=%.2f  steering_limit=%.1f°",
            len(self.waypoints), self.wheelbase, self.dt,
            *np.diag(self.matrix_q), self.matrix_r[0, 0], self.lqr_iter,
            self.lqr_eps, self.K_p, self.velocity_percentage,
            self.steering_limit_deg,
        )

    # ------------------------------------------------------------------ API
    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
        remaining_budget_s: float = 0.0,
        **_kwargs,
    ) -> ControlCmdMsg:
        # Map state → reference's [pose_x, pose_y, pose_theta, velocity].
        x, y, psi, v = state.x, state.y, state.psi, state.v

        # ── calc_control_points ───────────────────────────────────────────
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        fx = x + self.wheelbase * cos_p
        fy = y + self.wheelbase * sin_p
        front_axle = np.array([fx, fy])

        nearest_pt, _dist, _t, target_idx = _nearest_point(
            front_axle, self.waypoints[:, 0:2]
        )
        vec = front_axle - nearest_pt
        # Cross-track error: signed lateral offset of front axle, positive
        # to the vehicle's right.
        e_cg = float(vec[0] * sin_p + vec[1] * (-cos_p))

        theta_raceline = float(self.waypoints[target_idx, 3])
        theta_e = _pi_2_pi(theta_raceline - psi)
        v_ref = float(self.waypoints[target_idx, 2])
        kappa_ref = float(self.waypoints[target_idx, 4])

        self._last_target_pt = self.waypoints[target_idx, 0:2].copy()

        # ── controller (feedback + feed-forward) ──────────────────────────
        ts = self.dt
        A, B = _update_matrix(v, ts, self.wheelbase)
        K = _solve_lqr(A, B, self.matrix_q, self.matrix_r,
                       self.lqr_eps, self.lqr_iter)

        # First-tick guard: seed prev = current so derivatives are 0.
        if self._e_cg_prev is None:
            self._e_cg_prev = e_cg
            self._theta_e_prev = theta_e
        x_state = np.array([
            [e_cg],
            [(e_cg - self._e_cg_prev) / ts],
            [theta_e],
            [(theta_e - self._theta_e_prev) / ts],
        ])
        steer_fb = float((K @ x_state)[0, 0])
        steer_ff = kappa_ref * self.wheelbase
        steer_target = steer_fb + steer_ff

        # Persist for next step's finite-difference.
        self._e_cg_prev = e_cg
        self._theta_e_prev = theta_e

        # Clip steering angle (defensive).
        steering_limit_rad = math.radians(self.steering_limit_deg)
        steer_target = float(np.clip(steer_target, -steering_limit_rad, steering_limit_rad))

        # ── Speed / I/O conversion (our gym takes (accel, sv)) ────────────
        target_speed = v_ref * self.velocity_percentage
        sv = (steer_target - state.delta) / ts
        sv = float(np.clip(sv, self.sv_min, self.sv_max))
        accel = self.K_p * (target_speed - state.v)
        accel = float(np.clip(accel, -self.a_max, self.a_max))

        return ControlCmdMsg(acceleration=accel, steering_vel=sv)

    # ------------------------------------------------------------- internals
    def reset(self) -> None:
        """Clear the persistent error state (start of a new episode).
        Re-seeds prev = current on the next compute_action call."""
        self._e_cg_prev = None
        self._theta_e_prev = None
        self._last_target_pt = None

    def get_lookahead_point(self) -> np.ndarray | None:
        """World-frame (x, y) of the LQR target waypoint from the most recent
        compute_action call. Same accessor name as Stanley so the experiment
        runner can highlight it in the renderer without controller-specific
        branching."""
        return self._last_target_pt

    def get_best_trajectory(self) -> list:
        return []

    def get_last_weights(self) -> tuple:
        return (1 / 3, 1 / 3, 1 / 3)
