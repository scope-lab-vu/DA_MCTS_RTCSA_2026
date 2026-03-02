"""
MCTS DPW controller + SVGP dynamics + action prior — C++ accelerated.

Uses Extended Kinematic base model with SVGP residual corrections for
omega (yaw rate) and beta (slip angle), all running in C++.

Extends with SVGP/SGPR action prior for guided action sampling, dual
occupancy boundaries (imaginary + real), curvature-based dynamic horizon,
cornering stiffness estimation, and backward-progress check.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory

try:
    from mcts_f1tenth_svgp_cpp import F1TenthDPWSvgp as F1TenthDPW
except ImportError as e:
    raise ImportError(
        "mcts_f1tenth_svgp_cpp C++ extension not found. "
        "Build with: pip install -e src/controller/mcts_dpw_svgp_cpp/"
    ) from e

# Add SVGP/SGPR training code to path for load_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "MCTS_SVGP_action_sampler"))
from train import load_model as load_model_svgp
from train_krasserm import load_model as load_model_krasserm

log = logging.getLogger(__name__)

# Task index → friction mapping (for logging)
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}

_EPSILON = 1e-8
_C_ALPHA_NOMINAL = 104.89


# ---------------------------------------------------------------------------
# Helper: load SVGP dynamics model from .npz
# ---------------------------------------------------------------------------

def _load_model_and_scalers_npz(path: str):
    """Load SVGP model, scalers from a .npz file."""
    data = np.load(path, allow_pickle=False)
    model_json = json.loads(str(data["model_json"]))

    model = dict(model_json)
    for key in data.files:
        if key.startswith("model__"):
            k = key.split("__", 1)[1]
            model[k] = data[key]

    def unpack_scaler(prefix):
        s = StandardScaler(
            with_mean=bool(data[f"{prefix}__with_mean"]),
            with_std=bool(data[f"{prefix}__with_std"]),
        )
        s.mean_ = data[f"{prefix}__mean"]
        s.scale_ = data[f"{prefix}__scale"]
        if f"{prefix}__var" in data.files:
            s.var_ = data[f"{prefix}__var"]
        s.n_features_in_ = int(data[f"{prefix}__n_features_in"])
        return s

    xscaler = unpack_scaler("xscaler")
    yscaler = unpack_scaler("yscaler")
    return model, xscaler, yscaler


def _pack_svgp_data(model_dict, xscaler, yscaler, hp_mu, M, D_input, D_hidden):
    """Pack all SVGP parameters into a flat float64 array for C++.

    Layout: theta(D_aug+1) | alpha(M) | Z_aug(M*D_aug) |
            xmean(D_input) | xscale(D_input) | ymean(1) | yscale(1) |
            hp_mu(D_hidden)
    """
    theta = np.array(model_dict['theta_opt'], dtype=np.float64)
    K_mm_inv = np.array(model_dict['K_mm_inv'], dtype=np.float64)
    mu_m = np.array(model_dict['mu_m_opt'], dtype=np.float64)
    alpha = (K_mm_inv @ mu_m).ravel()

    Zx = np.array(model_dict['Zx_opt'], dtype=np.float64)
    Zh = np.array(model_dict['Zh_opt'], dtype=np.float64)
    Z_aug = np.concatenate([Zx, Zh], axis=1).ravel()

    xmean = np.array(xscaler.mean_, dtype=np.float64)
    xscale = np.array(xscaler.scale_, dtype=np.float64)
    ymean_val = float(np.asarray(yscaler.mean_).ravel()[0])
    yscale_val = float(np.asarray(yscaler.scale_).ravel()[0])
    hp = np.array(hp_mu, dtype=np.float64)

    return np.concatenate([
        theta, alpha, Z_aug, xmean, xscale,
        np.array([ymean_val, yscale_val], dtype=np.float64),
        hp,
    ])


# ---------------------------------------------------------------------------
# Helper: estimate cornering stiffness from consecutive states
# ---------------------------------------------------------------------------

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


class MCTSDPWSvgpCppController:
    """MCTS DPW controller with SVGP dynamics + action prior (C++ accelerated)."""

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig,
        simulation_config: SimulationConfig = None,
    ) -> None:
        self._vehicle_params = vehicle_params
        self._config = config

        self.dt = simulation_config.dt_s if simulation_config is not None else 0.04
        self.base_timestep = 0.02
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

        # Occupancy maps: imaginary (soft penalty) + real (hard penalty, done)
        self.track_occ_map = TrackOccupancyMap(
            track=self.track, cell_size=0.01, lane_width=config.lane_width,
        )
        self._occ_grid = self.track_occ_map._map.ravel().astype(np.float32)
        self._occ_x_min = self.track_occ_map.x_min
        self._occ_y_min = self.track_occ_map.y_min
        self._occ_cell_size = self.track_occ_map.cell_size
        self._occ_nx = self.track_occ_map.nx
        self._occ_ny = self.track_occ_map.ny

        real_width = getattr(config, 'real_track_width', 2.2)
        self.track_occ_map_real = TrackOccupancyMap(
            track=self.track, cell_size=0.01, lane_width=real_width,
        )
        self._occ_grid_real = self.track_occ_map_real._map.ravel().astype(np.float32)
        log.info("Dual boundaries: imaginary=%.1f (soft), real=%.1f (hard/done)",
                 config.lane_width, real_width)

        # Reference trajectory
        self.ref_traj = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning("Could not load reference trajectory: %s", e)

        # --- Load SVGP dynamics models ---
        project_root = Path(__file__).resolve().parents[3]
        task_index = getattr(config, 'svgp_task_index', 0)

        omega_path = str(project_root / config.svgp_omega_model_path)
        beta_path = str(project_root / config.svgp_beta_model_path)

        log.info("Loading omega SVGP from: %s", omega_path)
        omega_dict, omega_xs, omega_ys = _load_model_and_scalers_npz(omega_path)

        log.info("Loading beta SVGP from: %s", beta_path)
        beta_dict, beta_xs, beta_ys = _load_model_and_scalers_npz(beta_path)

        M = omega_dict['Zx_opt'].shape[0]
        D_input = omega_dict['Zx_opt'].shape[1]
        D_hidden = omega_dict['Zh_opt'].shape[1]
        self._svgp_M = M
        self._svgp_D_input = D_input
        self._svgp_D_hidden = D_hidden

        omega_hp_mu = np.array(omega_dict['h_mu_opt'][task_index], dtype=np.float64)
        beta_hp_mu = np.array(beta_dict['h_mu_opt'][task_index], dtype=np.float64)

        self._omega_data = _pack_svgp_data(
            omega_dict, omega_xs, omega_ys, omega_hp_mu, M, D_input, D_hidden,
        )
        self._beta_data = _pack_svgp_data(
            beta_dict, beta_xs, beta_ys, beta_hp_mu, M, D_input, D_hidden,
        )

        log.info(
            "SVGP dynamics loaded: M=%d D_input=%d D_hidden=%d task=%d (mu≈%.4f)",
            M, D_input, D_hidden, task_index, TASK_FRICTION_MAP.get(task_index, -1),
        )

        # --- Load action prior model ---
        model_path = config.action_prior_model_path
        if model_path is None:
            raise ValueError("action_prior_model_path must be set in controller config.")
        self._model_type = getattr(config, 'action_prior_model_type', 'svgp')

        if self._model_type == "sgpr_krasserm":
            self._gp_models, self._x_scaler, self._y_scaler, gp_config = \
                load_model_krasserm(model_path, device="cpu")
            self._gp_input_cols = gp_config["input_cols"]
            n_ind = gp_config.get("best_params", {}).get("num_inducing",
                        gp_config.get("num_inducing", "?"))
            log.info("Loaded SGPR Krasserm from %s (%s inducing, features=%s)",
                     model_path, n_ind, self._gp_input_cols)
        else:
            self._gp_model, self._gp_likelihood, self._x_scaler, self._y_scaler, gp_config = \
                load_model_svgp(model_path, device="cpu")
            self._gp_input_cols = gp_config.get("input_cols", None)
            self._gp_use_avg_kappa = gp_config.get("use_avg_kappa", True)
            self._gp_minimal = gp_config.get("minimal_features", False)
            n_ind = gp_config.get("num_inducing") or gp_config.get("best_params", {}).get("num_inducing", "?")
            log.info("Loaded SVGP action prior from %s (%s inducing, features=%s)",
                     model_path, n_ind,
                     self._gp_input_cols or ("minimal" if self._gp_minimal else "default"))

        # --- Cornering stiffness estimation ---
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
        self._local_cx = np.zeros(106)
        self._local_cy = np.zeros(106)
        self._local_heading = np.zeros(106)

        log.info(
            "MCTS DPW SVGP + Action Prior ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon,
        )

    # --- v_ref window with curvature solver ---

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity using forward-backward solver.

        1. Per-point curvature limit:  v_limit[i] = sqrt(a_lat_budget / |kappa[i]|)
        2. Backward pass:  propagate braking constraint from corners
        3. Forward pass:   propagate acceleration constraint from current speed
        """
        v_cap = 20.0
        v_current = state_np[3]
        a_max = self._vehicle_params.a_max

        kappas = self.track.get_curvature_lookahead(
            position=state_np[:2], v=v_current,
            horizon=self.curvature_lookahead, dt=self.dt,
        )

        self._max_kappa_lookahead = float(np.max(np.abs(kappas)))
        self._avg_kappa_lookahead = float(np.mean(np.abs(kappas)))

        # Dynamic horizon
        if self._max_kappa_lookahead < self.kappa_threshold:
            self._dynamic_horizon = self.rollout_horizon_straight
        else:
            self._dynamic_horizon = self.rollout_horizon_curve

        ds = max(abs(v_current), 1.0) * self.dt
        N = len(kappas)

        # Step 1: per-point curvature speed limit
        v_limit = np.minimum(v_cap,
                             np.sqrt(self.a_lat_budget / np.maximum(np.abs(kappas), 1e-6)))

        # Step 2: backward pass — brake in time for corners
        for i in range(N - 2, -1, -1):
            v_limit[i] = min(v_limit[i],
                             np.sqrt(v_limit[i + 1] ** 2 + 2 * a_max * ds))

        window_size = self.rollout_horizon_curve
        v_ref_now = float(v_limit[0])
        v_ref_window = np.full(window_size, v_ref_now)

        self._v_ref_window = v_ref_window.copy()

        # Extract local centerline slice (-5 to +100 points) for heading lookup in C++
        idx, _, _ = self.track.get_nearest_point(state_np[:2])
        N_cl = len(self.track.centerline)
        indices = np.arange(idx - 5, idx + 101) % N_cl
        self._local_cx = self.track.centerline[indices, 0].copy()
        self._local_cy = self.track.centerline[indices, 1].copy()
        self._local_heading = self.track.heading[indices].copy()

        return v_ref_window

    # --- Cornering stiffness ---

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

        self._calpha_buffer.append(cs_clipped)
        if len(self._calpha_buffer) > self._calpha_window_size:
            self._calpha_buffer = self._calpha_buffer[-self._calpha_window_size:]
        return float(np.mean(self._calpha_buffer))

    # --- Action prior inference ---

    _COL_TO_ARG = {
        "S_vel": "v", "S_delta": "delta", "S_omega": "omega", "S_beta": "beta",
        "max_kappa": "max_kappa", "avg_kappa": "avg_kappa", "C_alpha": "c_alpha",
    }

    def _predict_action_prior(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """Run GP inference to get action prior (mean, std) in original scale."""
        if self._model_type == "sgpr_krasserm":
            return self._predict_krasserm(v, delta, omega, beta, max_kappa, avg_kappa, c_alpha)
        return self._predict_svgp(v, delta, omega, beta, max_kappa, avg_kappa, c_alpha)

    def _predict_svgp(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """SVGP multi-output prediction."""
        if self._gp_input_cols is not None:
            local = {"v": v, "delta": delta, "omega": omega, "beta": beta,
                     "max_kappa": max_kappa, "avg_kappa": avg_kappa, "c_alpha": c_alpha}
            vals = [local[self._COL_TO_ARG[c]] for c in self._gp_input_cols]
            x_raw = np.array([vals], dtype=np.float32)
        elif self._gp_minimal:
            x_raw = np.array([[v, max_kappa, c_alpha]], dtype=np.float32)
        elif self._gp_use_avg_kappa:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, avg_kappa, c_alpha]],
                             dtype=np.float32)
        else:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, c_alpha]],
                             dtype=np.float32)
        x_scaled = self._x_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self._gp_likelihood(self._gp_model(x_tensor))

        mean_scaled = preds.mean[0].numpy()
        var_scaled = preds.variance[0].numpy()

        mean_original = mean_scaled * self._y_scaler.scale_ + self._y_scaler.mean_
        std_original = np.sqrt(var_scaled) * self._y_scaler.scale_
        return mean_original, std_original

    def _predict_krasserm(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """SGPR Krasserm independent single-output prediction."""
        local = {"v": v, "delta": delta, "omega": omega, "beta": beta,
                 "max_kappa": max_kappa, "avg_kappa": avg_kappa, "c_alpha": c_alpha}
        vals = [local[self._COL_TO_ARG[c]] for c in self._gp_input_cols]
        x_raw = np.array([vals], dtype=np.float64)
        x_scaled = self._x_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float64)

        mean_scaled = np.zeros(2)
        var_scaled = np.zeros(2)
        with torch.no_grad():
            for i, model in enumerate(self._gp_models):
                f_mean, _, y_var = model.predict(x_tensor)
                mean_scaled[i] = f_mean[0].item()
                var_scaled[i] = y_var[0].item()

        mean_original = mean_scaled * self._y_scaler.scale_ + self._y_scaler.mean_
        std_original = np.sqrt(var_scaled) * self._y_scaler.scale_
        return mean_original, std_original

    # --- Tree creation ---

    def _make_tree(self, state_np, v_ref_window):
        """Create a fresh C++ DPW tree with SVGP dynamics."""
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
            # Occupancy maps
            occ_grid=self._occ_grid,
            occ_x_min=self._occ_x_min, occ_y_min=self._occ_y_min,
            occ_cell_size=self._occ_cell_size,
            occ_nx=self._occ_nx, occ_ny=self._occ_ny,
            occ_grid_real=self._occ_grid_real,
            # Cost params
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            v_ref=self.v_ref,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount, dt=self.dt,
            normalize_reward=self.normalize_reward,
            # SVGP dynamics params
            omega_data=self._omega_data,
            beta_data=self._beta_data,
            svgp_M=self._svgp_M,
            svgp_D_input=self._svgp_D_input,
            svgp_D_hidden=self._svgp_D_hidden,
            base_timestep=self.base_timestep,
            seed=self.seed,
        )

    # --- Main control loop ---

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # 0. Startup: full throttle + zero steer until we reach 1.5 m/s (first 20 steps)
        self._timestep += 1
        if state.v < 1.5 and self._timestep <= 20:
            self._prev_state = state
            a_max = self._vehicle_params.a_max
            log.debug("Startup phase: v=%.2f < 1.5, step=%d, applying a_max=%.1f",
                      state.v, self._timestep, a_max)
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

        # 5. Push action prior, dynamic horizon, and local centerline to C++
        self._tree.set_action_prior(action_mean, action_std)
        self._tree.set_rollout_horizon(self._dynamic_horizon)
        self._tree.set_centerline(self._local_cx, self._local_cy, self._local_heading)

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

    # --- Introspection for logging ---

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
