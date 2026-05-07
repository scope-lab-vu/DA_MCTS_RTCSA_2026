"""
F1TENTH Gym vehicle interface.

Wraps the f1tenth_gym OpenAI Gym environment.
Translates between our IDL messages (ControlCmdMsg with acceleration/steering_vel)
and the gym's action format ([[steering_velocity, acceleration]]).

The gym has been patched (base_classes.py) to accept raw acceleration and
steering velocity directly, bypassing the internal PID.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional

import gym
import numpy as np
from f110_gym.envs.base_classes import Integrator

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import MapConfig, SimulationConfig, VehicleParams

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Krasser SVGP model loader + prediction (pure numpy, no C++ needed)
# ---------------------------------------------------------------------------

def _load_krasser_model(path: str) -> dict:
    """Load Krasser SVGP model from a .npz file."""
    d = np.load(path, allow_pickle=True)
    return {
        "theta": np.array(d["model__theta"], dtype=np.float64),
        "X_m": np.array(d["model__X_m"], dtype=np.float64),
        "mu_m": np.array(d["model__mu_m"], dtype=np.float64),
        "K_mm_inv": np.array(d["model__K_mm_inv"], dtype=np.float64),
        "A_m": np.array(d["model__A_m"], dtype=np.float64),
        "x_mean": np.array(d["xscaler__mean"], dtype=np.float64),
        "x_scale": np.array(d["xscaler__scale"], dtype=np.float64),
        "y_mean": np.array(d["yscaler__mean"], dtype=np.float64),
        "y_scale": np.array(d["yscaler__scale"], dtype=np.float64),
    }


def _predict_krasser_svgp(x_raw: np.ndarray, model: dict) -> float:
    """Predict mean using Krasser SVGP with RBF ARD kernel.

    x_raw: (D,) raw input features (not scaled).
    Returns: scalar prediction in original output scale.
    """
    # Normalize input
    x_scaled = (x_raw - model["x_mean"].ravel()) / model["x_scale"].ravel()

    # RBF ARD kernel: K(x, X_m)
    theta = model["theta"]
    D_input = model["X_m"].shape[1]
    lengthscales = theta[:D_input]
    sigma_f = theta[D_input]

    X_m = model["X_m"]  # (M, D)
    diff = (x_scaled[None, :] - X_m) / lengthscales[None, :]  # (M, D)
    dist_sq = np.sum(diff ** 2, axis=1)  # (M,)
    K_xm = (sigma_f ** 2) * np.exp(-0.5 * dist_sq)  # (M,)

    # Mean prediction: K_xm @ alpha, where alpha = K_mm_inv @ mu_m
    alpha = (model["K_mm_inv"] @ model["mu_m"]).ravel()  # (M,)
    f_mean_scaled = K_xm @ alpha  # scalar

    # Unnormalize
    y_mean = float(model["y_mean"].ravel()[0])
    y_scale = float(model["y_scale"].ravel()[0])
    return float(f_mean_scaled * y_scale + y_mean)


# ---------------------------------------------------------------------------
# Extended kinematic model (pure numpy, matches Ekinematic.casadi)
# ---------------------------------------------------------------------------

def _ekinematic_dxdt(state: np.ndarray, u: np.ndarray, lf: float, lr: float) -> np.ndarray:
    """Extended kinematic bicycle model.

    state: [x, y, delta, v, psi, omega, beta]
    u: [acceleration, steering_velocity]
    Returns: dxdt array of length 7.
    """
    x, y, delta, v, psi, omega, beta = state
    accl, sv = u

    x_dot = v * math.cos(beta + psi)
    y_dot = v * math.sin(beta + psi)
    delta_dot = sv
    v_dot = accl
    psi_dot = omega
    omega_dot = (delta_dot * v + delta * v_dot) / (lf + lr)
    beta_dot = (lf * (delta_dot * v + delta * v_dot)) / (lf + lr)

    return np.array([x_dot, y_dot, delta_dot, v_dot, psi_dot, omega_dot, beta_dot])


# ---------------------------------------------------------------------------
# Compute omega and beta for s_t from s_t and s_{t+1} observations
# ---------------------------------------------------------------------------

def _omega_beta_st(x_t, y_t, psi_t, x_tp1, y_tp1, psi_tp1, dt):
    """Compute omega and beta at time t using positions at t and t+1."""
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    dx, dy = x_tp1 - x_t, y_tp1 - y_t

    # yaw rate
    omega = wrap(psi_tp1 - psi_t) / dt

    # sideslip
    if dy == 0 and dx == 0:
        beta = 0.0
    else:
        beta = wrap(np.arctan2(dy, dx) - psi_t)

    return round(omega, 8), round(beta, 8)


# ---------------------------------------------------------------------------
# Nominal cornering stiffness
# ---------------------------------------------------------------------------

_C_ALPHA_NOMINAL = 104.89
_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# GymVehicleInterface
# ---------------------------------------------------------------------------

class GymVehicleInterface:
    """Vehicle interface backed by the f1tenth_gym simulator."""

    def __init__(
        self,
        vehicle_params: VehicleParams,
        simulation_config: SimulationConfig,
        map_config: MapConfig,
    ) -> None:
        self._vehicle_params = vehicle_params
        self._map_config = map_config
        self._dt = simulation_config.dt_s

        self._last_action = ControlCmdMsg()
        self._state = VehicleStateMsg()

        # SVGP state propagation config
        self._use_svgp_state = getattr(simulation_config, 'use_svgp_state', False)
        self._svgp_omega_model: Optional[dict] = None
        self._svgp_beta_model: Optional[dict] = None
        self._svgp_states: Optional[np.ndarray] = None  # [x, y, delta, v, psi, omega, beta]

        if self._use_svgp_state:
            project_root = Path(__file__).resolve().parents[2]
            omega_path = str(project_root / simulation_config.svgp_state_omega_model_path)
            beta_path = str(project_root / simulation_config.svgp_state_beta_model_path)
            log.info("Loading SVGP state omega model: %s", omega_path)
            self._svgp_omega_model = _load_krasser_model(omega_path)
            log.info("Loading SVGP state beta model: %s", beta_path)
            self._svgp_beta_model = _load_krasser_model(beta_path)
            log.info("SVGP state propagation enabled.")

        # Optional physics sub-stepping: integrate at `physics_dt_s` and
        # call the controller every `dt_s`, applying the same command for
        # `n_substeps = round(dt_s / physics_dt_s)` ticks. Required for
        # Stanley/LQR at dt_s ≥ ~0.07 s — Euler at that step size goes
        # unstable for the dynamic single-track model. MCTS configs leave
        # physics_dt_s = None so the gym integrates one step of dt_s as
        # before (no behaviour change at dt_s = 0.04).
        physics_dt = getattr(simulation_config, "physics_dt_s", None)
        if physics_dt is not None and physics_dt < simulation_config.dt_s:
            self._physics_dt = float(physics_dt)
            self._n_substeps = max(1, round(simulation_config.dt_s / self._physics_dt))
        else:
            self._physics_dt = float(simulation_config.dt_s)
            self._n_substeps = 1

        # Build gym environment with vehicle params.
        self._env = gym.make(
            "f110_gym:f110-v0",
            map=map_config.map_path,
            map_ext=map_config.map_ext,
            num_agents=1,
            timestep=self._physics_dt,
            integrator=Integrator.Euler,
            params=vehicle_params.to_gym_dict(),
        )

        # Reset at starting pose.
        self._start_pose = np.array([[
            map_config.sx,
            map_config.sy,
            map_config.stheta,
        ]])
        obs, _, _, _ = self._env.reset(self._start_pose)
        self._last_obs = obs
        self._state = self._extract_state()

        # Initialize SVGP state from ground truth at t=0
        if self._use_svgp_state:
            agent_state = self._env.sim.agents[0].state
            self._svgp_states = np.array([
                float(agent_state[0]),  # x
                float(agent_state[1]),  # y
                float(agent_state[2]),  # delta
                float(agent_state[3]),  # v
                float(agent_state[4]),  # psi
                float(agent_state[5]),  # omega
                float(agent_state[6]),  # beta
            ])
            # C_alpha estimation state
            self._prev_svgp_state: Optional[VehicleStateMsg] = None
            self._calpha_buffer: list = []
            self._cornering_stiffness: float = _C_ALPHA_NOMINAL

        log.info(
            "Gym env ready: map=%s  start=(%.2f, %.2f, %.2f)  svgp_state=%s",
            map_config.map_path,
            map_config.sx,
            map_config.sy,
            map_config.stheta,
            self._use_svgp_state,
        )

    # -- VehicleInterface contract ------------------------------------------

    def get_state(self) -> VehicleStateMsg:
        return self._state

    def get_last_applied_action(self) -> ControlCmdMsg:
        return self._last_action

    def apply_action(self, cmd: ControlCmdMsg) -> bool:
        """Apply action and return True if the episode is done.

        When `physics_dt_s < dt_s` was set in the simulation config, the
        same command is held across `_n_substeps` integrator ticks; we
        return early if the gym signals done mid-substep.
        """
        self._last_action = cmd

        # Gym (patched) expects [[steering_velocity, acceleration]].
        action = np.array([[cmd.steering_vel, cmd.acceleration]])
        done = False
        for _ in range(self._n_substeps):
            obs, _, done, _ = self._env.step(action)
            if done:
                break
        self._last_obs = obs

        if self._use_svgp_state:
            self._state = self._extract_svgp_state(cmd)
        else:
            self._state = self._extract_state()

        # Track collision flag from the gym environment.
        self._collision = bool(self._env.collisions[self._env.ego_idx])

        if done:
            reason = "collision" if self._collision else "lap completed"
            log.warning("Gym env done (%s).", reason)

        return bool(done)

    @property
    def collision(self) -> bool:
        """True if the last step resulted in a collision."""
        return getattr(self, '_collision', False)

    def set_planned_trajectory(self, xy_points: np.ndarray) -> None:
        """Set MCTS planned trajectory for visualization. xy_points: (N, 2)."""
        self._planned_trajectory = xy_points

    def set_lookahead_point(self, xy) -> None:
        """Highlight a single point (Stanley's lookahead). xy: (2,) array or
        None to clear."""
        self._lookahead_point = (
            np.asarray(xy, dtype=np.float64) if xy is not None else None
        )

    def set_mcts_trajectories(
        self,
        trajectories,
        best_idx: int = -1,
        ranks=None,
    ) -> None:
        """Set MCTS principal-variation trajectories for visualization.

        trajectories: iterable of (N, 2) arrays — one polyline per
                      first-level child of the MCTS root (most-visited
                      path).
        best_idx:     index (within `trajectories`) of the chosen action's
                      PV → drawn in bright cyan with thick line. -1 to
                      treat all equally.
        ranks:        optional list of float in [0, 1] per polyline,
                      0=highest-visit, 1=lowest. Used to fade runners-up
                      via a yellow→grey gradient. None → all drawn at
                      same intensity.
        """
        polylines = []
        for t in trajectories:
            arr = np.asarray(t, dtype=np.float64) if t is not None else None
            if arr is not None and len(arr) >= 2:
                polylines.append(arr)
        self._mcts_trajectories = polylines
        self._mcts_best_idx = int(best_idx)
        if ranks is None:
            self._mcts_ranks = [0.5] * len(polylines)
        else:
            self._mcts_ranks = [float(r) for r in ranks][:len(polylines)]

    def setup_rendering(self, waypoints: WaypointsMsg) -> None:
        """Register render callback that draws the raceline and follows the car."""
        from pyglet.gl import GL_POINTS, GL_QUADS, GL_TRIANGLE_FAN, GL_LINES

        # Pre-compute scaled waypoint vertices for drawing.
        points = np.column_stack([waypoints.x, waypoints.y])
        scaled_points = 50.0 * points
        self._drawn_waypoints: List = []
        self._drawn_trajectory: List = []
        self._drawn_lookahead: List = []
        self._drawn_mcts_pvs: List = []
        self._planned_trajectory: np.ndarray = np.empty((0, 2))
        self._lookahead_point: Optional[np.ndarray] = None
        self._mcts_trajectories: List[np.ndarray] = []
        self._mcts_best_idx: int = -1
        self._mcts_ranks: List[float] = []

        def render_callback(env_renderer):
            e = env_renderer

            # Update camera to follow car.
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left = left - 800
            e.right = right + 800
            e.top = top + 800
            e.bottom = bottom - 800

            # Draw waypoints once (they never move, so no need to update vertices).
            if len(self._drawn_waypoints) < scaled_points.shape[0]:
                for i in range(scaled_points.shape[0]):
                    b = e.batch.add(
                        1, GL_POINTS, None,
                        ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                        ('c3B/stream', [183, 193, 222]),
                    )
                    self._drawn_waypoints.append(b)

            # Draw MCTS planned trajectory (updated each step).
            # Remove old trajectory points.
            for b in self._drawn_trajectory:
                b.delete()
            self._drawn_trajectory.clear()

            traj = self._planned_trajectory
            if traj is not None and len(traj) > 0:
                sz = 8.0  # half-size of each quad in screen units
                for i in range(len(traj)):
                    # Color gradient: green (start) -> red (end)
                    t = i / max(len(traj) - 1, 1)
                    r = int(255 * t)
                    g = int(255 * (1 - t))
                    b_color = 0
                    sx, sy = 50.0 * traj[i, 0], 50.0 * traj[i, 1]
                    b = e.batch.add(
                        4, GL_QUADS, None,
                        ('v3f/stream', [
                            sx - sz, sy - sz, 0.,
                            sx + sz, sy - sz, 0.,
                            sx + sz, sy + sz, 0.,
                            sx - sz, sy + sz, 0.,
                        ]),
                        ('c3B/stream', [r, g, b_color] * 4),
                    )
                    self._drawn_trajectory.append(b)

            # Draw MCTS principal-variation trajectories as quad bands
            # (thin filled rectangles per segment) so line thickness is
            # explicit rather than depending on glLineWidth state.
            # Chosen PV: bright cyan + thicker. Runners-up: yellow→orange
            # gradient faded by visit-rank (0=most-visited runner-up,
            # 1=least), thinner.
            for b in self._drawn_mcts_pvs:
                b.delete()
            self._drawn_mcts_pvs.clear()
            best_idx = self._mcts_best_idx
            HALF_W_BEST = 4.0      # ~8 px wide in screen units
            HALF_W_OTHER = 2.0     # ~4 px wide for runners-up
            for k, polyline in enumerate(self._mcts_trajectories):
                if len(polyline) < 2:
                    continue
                is_best = (k == best_idx)
                if is_best:
                    color = (0, 220, 255)         # bright cyan
                    half_w = HALF_W_BEST
                else:
                    rank = (self._mcts_ranks[k]
                            if k < len(self._mcts_ranks) else 0.5)
                    # Yellow (255,210,80) → grey-orange (140,90,70)
                    r = int(255 - rank * (255 - 140))
                    g = int(210 - rank * (210 - 90))
                    b_c = int(80 - rank * (80 - 70))
                    color = (r, g, b_c)
                    half_w = HALF_W_OTHER

                pts = 50.0 * polyline
                n_segs = len(pts) - 1
                verts = []
                for i in range(n_segs):
                    x1, y1 = pts[i, 0], pts[i, 1]
                    x2, y2 = pts[i + 1, 0], pts[i + 1, 1]
                    dx, dy = x2 - x1, y2 - y1
                    length = math.hypot(dx, dy)
                    if length < 1e-6:
                        continue
                    # Perpendicular unit vector (rotate 90 deg).
                    px, py = -dy / length, dx / length
                    ox, oy = px * half_w, py * half_w
                    # Quad: p1+offset, p2+offset, p2-offset, p1-offset.
                    verts.extend([x1 + ox, y1 + oy, 0.0])
                    verts.extend([x2 + ox, y2 + oy, 0.0])
                    verts.extend([x2 - ox, y2 - oy, 0.0])
                    verts.extend([x1 - ox, y1 - oy, 0.0])
                if not verts:
                    continue
                n_verts = len(verts) // 3
                b = e.batch.add(
                    n_verts, GL_QUADS, None,
                    ('v3f/stream', verts),
                    ('c3B/stream', list(color) * n_verts),
                )
                self._drawn_mcts_pvs.append(b)

            # Draw lookahead point (single highlighted waypoint, e.g. Stanley's)
            # as a small bright-yellow filled circle via a triangle fan.
            for b in self._drawn_lookahead:
                b.delete()
            self._drawn_lookahead.clear()
            if self._lookahead_point is not None:
                radius = 6.0       # smaller than the previous 16 px square
                n_segs = 20        # perimeter samples — looks round at this size
                cx = 50.0 * self._lookahead_point[0]
                cy = 50.0 * self._lookahead_point[1]
                verts = [cx, cy, 0.0]
                for i in range(n_segs + 1):  # +1 to close the fan
                    theta = 2.0 * math.pi * i / n_segs
                    verts.extend([cx + radius * math.cos(theta),
                                  cy + radius * math.sin(theta), 0.0])
                n_verts = n_segs + 2  # 1 centre + n_segs + 1 closing
                b = e.batch.add(
                    n_verts, GL_TRIANGLE_FAN, None,
                    ('v3f/stream', verts),
                    # Bright yellow — distinct from raceline (blue) and
                    # planned-trajectory gradient (red→green).
                    ('c3B/stream', [255, 255, 0] * n_verts),
                )
                self._drawn_lookahead.append(b)

        self._env.add_render_callback(render_callback)
        log.info("Render callback registered with %d waypoints.", len(waypoints.x))

    def render(self, mode: str = "human") -> None:
        self._env.render(mode=mode)

    def get_scan(self) -> np.ndarray:
        """Return the latest lidar scan ranges."""
        return np.array(self._last_obs['scans'][self._env.ego_idx])

    def get_scan_angles(self) -> np.ndarray:
        """Return the lidar beam angles."""
        from f110_gym.envs.base_classes import RaceCar
        return RaceCar.scan_angles.copy()

    # -- Internal helpers ---------------------------------------------------

    def _extract_state(self) -> VehicleStateMsg:
        """Extract full 7-state from the gym simulator's internal agent state.

        Gym internal state vector: [x, y, delta, v, psi, omega, beta]
        """
        agent_state = self._env.sim.agents[0].state
        return VehicleStateMsg(
            x=float(agent_state[0]),
            y=float(agent_state[1]),
            delta=float(agent_state[2]),
            v=float(agent_state[3]),
            psi=float(agent_state[4]),
            omega=float(agent_state[5]),
            beta=float(agent_state[6]),
        )

    def _extract_svgp_state(self, cmd: ControlCmdMsg) -> VehicleStateMsg:
        """Extract state using SVGP state propagation with 2 sub-steps.

        The SVGP models were trained at Ts=0.02s but we run at dt=0.04s,
        so we do 2 sub-steps of 0.02s each for ekin+GP prediction.

        Matches the reference MPC code flow:
        1. Retroactively fix omega_t, beta_t using omega_beta_st()
        2. Run 2x (ekin + GP) sub-steps at Ts=0.02 to predict delta, omega, beta
        3. Return state: x,y,v,psi from observations; delta,omega,beta from ekin+GP
        """
        obs = self._last_obs
        dt = self._dt           # 0.04s
        sub_dt = 0.02           # SVGP training timestep
        n_substeps = round(dt / sub_dt)  # 2
        lf = self._vehicle_params.lf
        lr = self._vehicle_params.lr

        # Observations from gym (equivalent to sensor readings on real car)
        obs_x = float(obs['poses_x'][0])
        obs_y = float(obs['poses_y'][0])
        obs_vel = float(obs['linear_vels_x'][0])
        obs_psi = float(obs['poses_theta'][0])

        # Current state before this step
        states_st = self._svgp_states.copy()

        # Retroactively compute omega_t and beta_t using t and t+1 positions
        omega_st, beta_st = _omega_beta_st(
            states_st[0], states_st[1], states_st[4],
            obs_x, obs_y, obs_psi, dt,
        )
        # Fix s_t with correct omega and beta
        states_st[5] = omega_st
        states_st[6] = beta_st

        # Action: [acceleration, steering_velocity]
        accl = cmd.acceleration
        sv = cmd.steering_vel
        u = np.array([accl, sv])

        # Estimate cornering stiffness for GP input
        c_alpha = self._estimate_c_alpha(states_st, omega_st, beta_st)

        # Run n_substeps of ekin + GP at sub_dt
        state_sub = states_st.copy()
        for _ in range(n_substeps):
            # GP corrections at current sub-state
            gp_input = np.array([
                state_sub[2],  # delta
                state_sub[3],  # v
                state_sub[5],  # omega
                state_sub[6],  # beta
                accl,          # acceleration
                sv,            # steering velocity
                c_alpha,       # cornering stiffness
            ])
            gp_omega_corr = _predict_krasser_svgp(gp_input, self._svgp_omega_model)
            gp_beta_corr = _predict_krasser_svgp(gp_input, self._svgp_beta_model)

            # Ekin forward step
            dxdt = _ekinematic_dxdt(state_sub, u, lf, lr)
            state_sub = state_sub + sub_dt * np.array(dxdt)

            # Apply GP corrections to omega and beta
            state_sub[5] += gp_omega_corr
            state_sub[6] += gp_beta_corr

        # Predicted delta, omega, beta from ekin+GP sub-stepping
        next_delta = float(state_sub[2])
        next_ekin_omega = float(state_sub[5])
        next_ekin_beta = float(state_sub[6])

        # Observations for next state
        next_x = obs_x
        next_y = obs_y
        next_v = obs_vel
        next_psi = obs_psi

        # Bound yaw angle to [0, 2*pi]
        if next_psi > 2 * math.pi:
            next_psi -= 2 * math.pi
        elif next_psi < 0:
            next_psi += 2 * math.pi

        # Store state internally (omega/beta will be overwritten
        # by omega_beta_st() at the start of the next iteration)
        self._svgp_states = np.array([
            next_x, next_y, next_delta, next_v, next_psi,
            next_ekin_omega, next_ekin_beta,
        ])

        # Return state: controller sees ekin+GP omega/beta
        # (these get corrected by omega_beta_st at the next iteration
        # before being fed to the GP)
        return VehicleStateMsg(
            x=next_x,
            y=next_y,
            delta=next_delta,
            v=next_v,
            psi=next_psi,
            omega=next_ekin_omega,
            beta=next_ekin_beta,
        )

    def _estimate_c_alpha(
        self, states_st: np.ndarray, omega: float, beta: float,
    ) -> float:
        """Estimate cornering stiffness from consecutive SVGP states."""
        curr_state = VehicleStateMsg(
            x=states_st[0], y=states_st[1], delta=states_st[2],
            v=states_st[3], psi=states_st[4], omega=omega, beta=beta,
        )

        if self._prev_svgp_state is not None:
            prev = self._prev_svgp_state
            dt = self._dt
            lf = self._vehicle_params.lf
            lr = self._vehicle_params.lr
            mass = self._vehicle_params.m
            iz = self._vehicle_params.I

            v_x_prev = prev.v * math.cos(prev.beta)
            v_y_prev = prev.v * math.sin(prev.beta)
            v_y_curr = curr_state.v * math.sin(curr_state.beta)

            v_y_dot = (v_y_curr - v_y_prev) / dt
            omega_dot = (curr_state.omega - prev.omega) / dt

            F_yr = (
                lf * mass * (v_y_dot + v_x_prev * prev.omega) - iz * omega_dot
            ) / (lf + lr)

            alpha_r = math.atan2(
                lr * prev.omega - v_y_prev, abs(v_x_prev) + _EPSILON
            )

            cs_raw = abs(F_yr) / (abs(alpha_r) + _EPSILON)
            cs_clipped = float(np.clip(cs_raw, 0.0, _C_ALPHA_NOMINAL))
        else:
            cs_clipped = _C_ALPHA_NOMINAL

        self._prev_svgp_state = curr_state
        self._calpha_buffer.append(cs_clipped)
        if len(self._calpha_buffer) > 10:
            self._calpha_buffer = self._calpha_buffer[-10:]

        self._cornering_stiffness = float(np.mean(self._calpha_buffer))
        return self._cornering_stiffness
