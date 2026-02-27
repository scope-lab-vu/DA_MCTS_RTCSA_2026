"""
MCTS DPW controller with SVGP action prior for F1TENTH racing — C++ accelerated.

Extends the base MCTS DPW C++ controller by replacing uniform action sampling
with a learned Gaussian prior from a trained SVGP model. At each timestep:

1. Estimate cornering stiffness from consecutive vehicle states
2. Compute curvature lookahead (max_kappa, avg_kappa) from the track
3. Build GP input: (v, delta, omega, beta, max_kappa, avg_kappa, C_alpha)
4. Run SVGP inference -> mean and std for (acceleration, steering_velocity)
5. Pass (mean, std) to C++ via set_action_prior()
6. C++ MCTS samples actions from N(mean, std) instead of uniform
"""

import logging
import math
import sys
import os
from typing import Optional

import numpy as np
import torch

import gpytorch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory

try:
    from mcts_f1tenth_cpp import F1TenthDPW
except ImportError as e:
    raise ImportError(
        "mcts_f1tenth_cpp C++ extension not found. "
        "Build it with: pip install -e src/controller/mcts_dpw_cpp/"
    ) from e

# Add SVGP training code to path for load_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "MCTS_SVGP_action_sampler"))
from train import load_model

log = logging.getLogger(__name__)

_EPSILON = 1e-8
_C_ALPHA_NOMINAL = 104.89


def _compute_cornering_stiffness(
    prev: VehicleStateMsg,
    curr: VehicleStateMsg,
    dt: float,
    lf: float,
    lr: float,
    mass: float,
    iz: float,
) -> float:
    """Estimate rear cornering stiffness from two consecutive states."""
    v_x_prev = prev.v * math.cos(prev.beta)
    v_y_prev = prev.v * math.sin(prev.beta)
    v_y_curr = curr.v * math.sin(curr.beta)

    v_y_dot = (v_y_curr - v_y_prev) / dt
    omega_dot = (curr.omega - prev.omega) / dt

    F_yr = (
        lf * mass * (v_y_dot + v_x_prev * prev.omega) - iz * omega_dot
    ) / (lf + lr)

    alpha_r = math.atan2(lr * prev.omega - v_y_prev, abs(v_x_prev) + _EPSILON)

    cs = abs(F_yr) / (abs(alpha_r) + _EPSILON)
    return min(cs, _C_ALPHA_NOMINAL)


class MCTSKnownDynActionPriorCppController:
    """MCTS DPW controller with known dynamics + SVGP action prior (C++ accelerated)."""

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig,
        simulation_config: SimulationConfig = None,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._vehicle_params = vehicle_params
        self._config = config

        self.dt = simulation_config.dt_s if simulation_config is not None else 0.04
        self.seed = simulation_config.seed if simulation_config is not None else 42

        # MCTS hyperparameters
        self.n_simulations = config.n_simulations
        self.K = config.exploration_constant
        self.alpha = config.alpha_dpw
        self.beta = config.beta_dpw
        self.discount = config.discount
        self.rollout_horizon = config.rollout_horizon
        self.rollout_horizon_straight = config.rollout_horizon_straight
        self.rollout_horizon_curve = config.rollout_horizon_curve
        self.kappa_threshold = config.kappa_threshold
        self.v_ref = config.v_ref
        self.speed_weight = config.speed_weight
        self.alpha_threshold = config.alpha_threshold
        self.a_lat_budget = config.a_lat_budget
        self.curvature_lookahead = config.curvature_lookahead
        self.mcts_time_budget = config.mcts_time_budget
        self.normalize_reward = getattr(config, 'normalize_reward', False)

        # Track
        if map_config.centerline_path is None:
            raise ValueError("MCTS controller requires centerline_path in map config.")
        self.track = Track(map_config.centerline_path)

        # Occupancy map
        self.track_occ_map = TrackOccupancyMap(
            track=self.track, cell_size=0.01, lane_width=config.lane_width,
        )
        self._occ_grid = self.track_occ_map._map.ravel().astype(np.float32)
        self._occ_x_min = self.track_occ_map.x_min
        self._occ_y_min = self.track_occ_map.y_min
        self._occ_cell_size = self.track_occ_map.cell_size
        self._occ_nx = self.track_occ_map.nx
        self._occ_ny = self.track_occ_map.ny

        # Reference trajectory
        self.ref_traj = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning("Could not load reference trajectory: %s", e)

        # ── Load SVGP action prior ────────────────────────────────────────
        model_path = config.action_prior_model_path
        if model_path is None:
            raise ValueError("action_prior_model_path must be set in controller config.")
        self._gp_model, self._gp_likelihood, self._x_scaler, self._y_scaler, gp_config = \
            load_model(model_path, device="cpu")
        self._gp_use_avg_kappa = gp_config.get("use_avg_kappa", True)
        n_ind = gp_config.get("num_inducing") or gp_config.get("best_params", {}).get("num_inducing", "?")
        log.info("Loaded SVGP action prior from %s (%s inducing points, avg_kappa=%s)",
                 model_path, n_ind, self._gp_use_avg_kappa)

        # ── Cornering stiffness estimation ────────────────────────────────
        self._calpha_window_size = config.calpha_window_size
        self._lf = vehicle_params.lf
        self._lr = vehicle_params.lr
        self._mass = vehicle_params.m
        self._iz = vehicle_params.I

        # Runtime state
        self._tree = None
        self._last_action = None
        self._prev_state: Optional[VehicleStateMsg] = None
        self._cornering_stiffness: float = _C_ALPHA_NOMINAL
        self._calpha_buffer: list = []
        self._max_kappa_lookahead = 0.0
        self._avg_kappa_lookahead = 0.0
        self._dynamic_horizon = self.rollout_horizon
        self._last_action_mean = np.zeros(2)
        self._last_action_std = np.ones(2)
        self._last_best_trajectory = []
        self._timestep = 0

        log.info(
            "MCTS SVGP Action Prior ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f",
            self.n_simulations, self.K, self.alpha, self.beta,
        )

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity and curvature stats."""
        v_max = 12.0
        kappas = self.track.get_curvature_lookahead(
            position=state_np[:2], v=state_np[3],
            horizon=self.curvature_lookahead, dt=self.dt,
        )

        self._max_kappa_lookahead = float(np.max(np.abs(kappas)))
        self._avg_kappa_lookahead = float(np.mean(np.abs(kappas)))

        # Dynamic horizon
        if self._max_kappa_lookahead < self.kappa_threshold:
            self._dynamic_horizon = self.rollout_horizon_straight
        else:
            self._dynamic_horizon = self.rollout_horizon_curve

        window_size = self.rollout_horizon_curve
        v_ref_window = np.zeros(window_size)
        for i in range(window_size):
            max_kappa = np.max(kappas[i:])
            v_ref_window[i] = min(v_max, np.sqrt(self.a_lat_budget / max(max_kappa, 1e-6)))

        min_v_ref = min(v_ref_window)
        for i in range(window_size):
            v_ref_window[i] = min_v_ref

        self._v_ref_window = v_ref_window.copy()
        return v_ref_window

    def _estimate_cornering_stiffness(self, state: VehicleStateMsg) -> float:
        """Estimate and smooth cornering stiffness."""
        if self._prev_state is not None:
            cs_raw = _compute_cornering_stiffness(
                self._prev_state, state, self.dt,
                self._lf, self._lr, self._mass, self._iz,
            )
            cs_clipped = float(np.clip(cs_raw, 0.0, _C_ALPHA_NOMINAL))
        else:
            cs_clipped = _C_ALPHA_NOMINAL

        self._cornering_stiffness = cs_clipped

        # Rolling window
        self._calpha_buffer.append(cs_clipped)
        if len(self._calpha_buffer) > self._calpha_window_size:
            self._calpha_buffer = self._calpha_buffer[-self._calpha_window_size:]
        return float(np.mean(self._calpha_buffer))

    def _predict_action_prior(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """Run SVGP inference to get action prior (mean, std) in original scale."""
        if self._gp_use_avg_kappa:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, avg_kappa, c_alpha]],
                             dtype=np.float32)
        else:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, c_alpha]],
                             dtype=np.float32)
        x_scaled = self._x_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self._gp_likelihood(self._gp_model(x_tensor))

        # preds.mean: (1, 2), preds.variance: (1, 2)  in scaled space
        mean_scaled = preds.mean[0].numpy()   # (2,)
        var_scaled = preds.variance[0].numpy()  # (2,)

        # Inverse transform to original scale
        mean_original = mean_scaled * self._y_scaler.scale_ + self._y_scaler.mean_
        std_original = np.sqrt(var_scaled) * self._y_scaler.scale_

        return mean_original, std_original

    def _make_tree(self, state_np, v_ref_window):
        """Create a fresh C++ DPW tree."""
        p = self._vehicle_params
        return F1TenthDPW(
            initial_obs=state_np,
            alpha=self.alpha, beta=self.beta, K=self.K,
            mu=p.mu, C_Sf=p.C_Sf, C_Sr=p.C_Sr,
            lf=p.lf, lr=p.lr, h=p.h,
            m=p.m, I=p.I,
            s_min=p.s_min, s_max=p.s_max,
            sv_min=p.sv_min, sv_max=p.sv_max,
            v_switch=p.v_switch, a_max=p.a_max,
            v_min=p.v_min, v_max=p.v_max,
            width=p.width, length=p.length,
            occ_grid=self._occ_grid,
            occ_x_min=self._occ_x_min, occ_y_min=self._occ_y_min,
            occ_cell_size=self._occ_cell_size,
            occ_nx=self._occ_nx, occ_ny=self._occ_ny,
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            v_ref=self.v_ref,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount, dt=self.dt,
            normalize_reward=self.normalize_reward,
            seed=self.seed,
        )

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # 0. Startup: full throttle + zero steer until we reach 1.5 m/s (first 20 steps only)
        self._timestep += 1
        if state.v < 1.5 and self._timestep <= 20:
            self._prev_state = state
            a_max = self._vehicle_params.a_max
            log.debug("Startup phase: v=%.2f < 1.5, step=%d, applying a_max=%.1f", state.v, self._timestep, a_max)
            return ControlCmdMsg(acceleration=float(a_max), steering_vel=0.0)

        # 1. Compute curvature + v_ref window
        v_ref_window = self._get_v_ref_window(state_np)

        # 2. Estimate cornering stiffness
        c_alpha = self._estimate_cornering_stiffness(state)

        # 3. SVGP action prior inference
        action_mean, action_std = self._predict_action_prior(
            state.v, state.delta, state.omega, state.beta,
            self._max_kappa_lookahead, self._avg_kappa_lookahead, c_alpha,
        )
        self._last_action_mean = action_mean
        self._last_action_std = action_std

        # 4. Tree reuse or creation
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.set_state(state_np)
                self._tree.set_v_ref_window(v_ref_window)
            except RuntimeError:
                self._tree = None

        if self._tree is None:
            self._tree = self._make_tree(state_np, v_ref_window)
            self._tree.set_v_ref_window(v_ref_window)

        # 5. Push action prior and dynamic horizon to C++
        self._tree.set_action_prior(action_mean, action_std)
        self._tree.set_rollout_horizon(self._dynamic_horizon)

        # 6. Run MCTS
        if self.mcts_time_budget > 0:
            n_sims = self._tree.learn_timed(self.mcts_time_budget)
        else:
            self._tree.learn(self.n_simulations)
            n_sims = self.n_simulations
        log.debug("MCTS ran %d simulations", n_sims)

        # 7. Extract best trajectory from tree (for visualization)
        try:
            self._last_best_trajectory = self._tree.get_best_trajectory(
                self._dynamic_horizon)
        except Exception:
            self._last_best_trajectory = []

        # 8. Select best action
        action = self._tree.best_action()
        self._last_action = action
        self._prev_state = state

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        self._tree = None
        self._last_action = None
        self._prev_state = None
        self._cornering_stiffness = _C_ALPHA_NOMINAL
        self._calpha_buffer = []
        self._timestep = 0

    # ── Introspection for logging ─────────────────────────────────────────

    def get_max_kappa(self) -> float:
        return self._max_kappa_lookahead

    def get_avg_kappa(self) -> float:
        return self._avg_kappa_lookahead

    def get_cornering_stiffness(self) -> float:
        return self._cornering_stiffness

    def get_action_prior_mean(self) -> np.ndarray:
        return self._last_action_mean

    def get_action_prior_std(self) -> np.ndarray:
        return self._last_action_std

    def get_best_trajectory(self) -> list:
        """Return the MCTS best trajectory as list of dicts with state/action/reward."""
        return getattr(self, '_last_best_trajectory', [])
