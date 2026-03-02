"""
MCTS DPW controller + PI-SVGP dynamics — C++ accelerated, with online
cornering stiffness (C_alpha) adaptation + GP action prior + dual boundaries.

Extends the base MCTS PI-SVGP C++ controller with:

  1. Internal C_alpha estimation — rear cornering stiffness is computed from
     consecutive VehicleStateMsg observations (no external input needed).

  2. Rolling-window latent update — the mean C_alpha over the last
     `calpha_window_size` steps is mapped to PI-SVGP task latent parameters
     via linear interpolation, then pushed to C++ via set_hp_mu().

  3. C_alpha-adaptive reference velocity — the v_ref window is reduced when
     the estimated surface grip is low.

  4. GP action prior — learned Gaussian prior for action sampling.

  5. Dual occupancy maps — soft imaginary + hard real track boundary.

  6. Heading-backward check — terminates rollouts facing wrong direction.

  7. Forward-backward velocity solver — curvature-aware v_ref.
"""

import json
import logging
import math
import sys
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import gpytorch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory

try:
    from mcts_f1tenth_pi_svgp_cpp import F1TenthDPWPiSvgp as F1TenthDPW
except ImportError as e:
    raise ImportError(
        "mcts_f1tenth_pi_svgp_cpp C++ extension not found. "
        "Build with: pip install -e src/controller/mcts_pi_svgp_cpp_cs/"
    ) from e

# Add SVGP/SGPR training code to path for load_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "MCTS_SVGP_action_sampler"))
from train import load_model as load_model_svgp
from train_krasserm import load_model as load_model_krasserm

log = logging.getLogger(__name__)

# Task index -> friction mapping (for logging)
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}

# Small value to avoid division by zero in C_alpha formula
_EPSILON = 1e-8

# Nominal high-friction cornering stiffness — used as the initial value and
# as the clip upper bound (matches task-0 anchor in SVGP training data).
_C_ALPHA_NOMINAL = 104.89


# ---------------------------------------------------------------------------
# Helper: load PI-SVGP model from .npz
# ---------------------------------------------------------------------------

def _load_model_and_scalers_npz(path: str):
    """Load PI-SVGP model + scalers from .npz file."""
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
    return model, xscaler


def _pack_pi_svgp_data(model_dict, xscaler, hp_mu, M, D_input, D_hidden, num_outputs):
    """Pack all PI-SVGP parameters into a flat float64 array for C++.

    Layout: theta(D_aug+1) | alpha(M*num_outputs) | Z_aug(M*D_aug) |
            xmean(D_input) | xscale(D_input) | hp_mu(D_hidden) |
            output_bias(num_outputs) | K_mm_inv(M*M) | S_diag(M*num_outputs)
    """
    D_aug = D_input + D_hidden

    theta = np.array(model_dict['theta_opt'], dtype=np.float64)

    # Precompute K_mm_inv @ m in Python
    Zx = np.array(model_dict['Zx_opt'], dtype=np.float64)
    Zh = np.array(model_dict['Zh_opt'], dtype=np.float64)
    Z_aug = np.concatenate([Zx, Zh], axis=1)  # (M, D_aug)

    # Build K_mm and invert
    from src.dynamics.pi_svgp import _rbf_ard_kernel
    theta_t = torch.tensor(theta, dtype=torch.float64)
    Z_aug_t = torch.tensor(Z_aug, dtype=torch.float64)
    K_mm = _rbf_ard_kernel(Z_aug_t, Z_aug_t, theta_t)
    K_mm = K_mm + torch.eye(M, dtype=torch.float64) * 1e-6
    K_mm_inv = torch.linalg.inv(K_mm).numpy()

    m_opt = np.array(model_dict['m_opt'], dtype=np.float64)  # (M, num_outputs)
    alpha = (K_mm_inv @ m_opt).ravel()  # (M * num_outputs), row-major

    Z_aug_flat = Z_aug.ravel()
    K_mm_inv_flat = K_mm_inv.ravel()

    # Variational posterior diagonal variance: S_diag[j, p] = exp(2 * m_logstd[j, p])
    m_logstd = np.array(model_dict['m_logstd_opt'], dtype=np.float64)  # (M, num_outputs)
    S_diag = np.exp(2.0 * m_logstd).ravel()  # (M * num_outputs), row-major

    xmean = np.array(xscaler.mean_, dtype=np.float64)
    xscale = np.array(xscaler.scale_, dtype=np.float64)
    hp = np.array(hp_mu, dtype=np.float64)
    output_bias = np.array(model_dict['output_bias_opt'], dtype=np.float64)

    return np.concatenate([
        theta, alpha, Z_aug_flat, xmean, xscale, hp, output_bias,
        K_mm_inv_flat, S_diag,
    ])


# ---------------------------------------------------------------------------
# Helper: estimate rear cornering stiffness from two consecutive states
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

    return abs(F_yr) / (abs(alpha_r) + _EPSILON)


class MCTSPiSvgpCppCsController:
    """MCTS DPW controller with PI-SVGP dynamics + online C_alpha adaptation
    + GP action prior + dual boundaries + heading check."""

    # Column name -> function argument mapping for krasserm models
    _COL_TO_ARG = {
        "S_vel": "v", "S_delta": "delta", "S_omega": "omega", "S_beta": "beta",
        "max_kappa": "max_kappa", "avg_kappa": "avg_kappa", "C_alpha": "c_alpha",
    }

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
        self.seed = simulation_config.seed if simulation_config is not None else 42
        self.base_timestep = 0.02

        # MCTS hyperparameters.
        self.n_simulations = config.n_simulations
        self.K = config.exploration_constant
        self.alpha = config.alpha_dpw
        self.beta = config.beta_dpw
        self.discount = config.discount
        self.rollout_horizon = config.rollout_horizon
        self.rollout_horizon_straight = getattr(config, 'rollout_horizon_straight', config.rollout_horizon)
        self.rollout_horizon_curve = getattr(config, 'rollout_horizon_curve', config.rollout_horizon)
        self.kappa_threshold = getattr(config, 'kappa_threshold', 0.05)
        self.v_ref = config.v_ref
        self.speed_weight = config.speed_weight
        self.alpha_threshold = config.alpha_threshold
        self.a_lat_budget = getattr(config, 'a_lat_budget', 30.0)
        self.curvature_lookahead = getattr(config, 'curvature_lookahead', 30)
        self.mcts_time_budget = getattr(config, 'mcts_time_budget', 0)
        self.normalize_reward = getattr(config, 'normalize_reward', False)

        # Cornering stiffness velocity-adaptation parameters.
        self._cs_w1 = config.cs_w1
        self._cs_w2 = config.cs_w2
        self._cs_w3 = config.cs_w3
        self._C_alpha_th = config.C_alpha_th
        self._calpha_window_size = config.calpha_window_size

        # Cache vehicle params needed for the C_alpha formula.
        self._lf = vehicle_params.lf
        self._lr = vehicle_params.lr
        self._mass = vehicle_params.m
        self._iz = vehicle_params.I

        # Load track (centerline CSV).
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS PI-SVGP C++ CS controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

        # Occupancy maps: imaginary (soft penalty) + real (hard penalty, done)
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.01,
            lane_width=config.lane_width,
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

        # Reference trajectory for per-step v_ref.
        self.ref_traj = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning(
                    "Could not load reference trajectory: %s. "
                    "Using constant v_ref=%.1f", e, self.v_ref
                )

        # --- Load PI-SVGP dynamics model ---
        project_root = Path(__file__).resolve().parents[3]
        task_index = getattr(config, 'svgp_task_index', 0)

        model_path = str(project_root / config.pi_svgp_model_path)

        log.info("Loading PI-SVGP from: %s", model_path)
        model_dict, xscaler = _load_model_and_scalers_npz(model_path)

        M = int(model_dict['M'])
        D_input = int(model_dict['dx_flat'])
        D_hidden = int(model_dict['Hdim'])
        num_outputs = int(model_dict['num_outputs'])
        self._svgp_M = M
        self._svgp_D_input = D_input
        self._svgp_D_hidden = D_hidden
        self._svgp_num_outputs = num_outputs

        hp_mu = np.array(model_dict['h_mu_opt'][task_index], dtype=np.float64)

        self._svgp_data = _pack_pi_svgp_data(
            model_dict, xscaler, hp_mu, M, D_input, D_hidden, num_outputs,
        )

        # Build C_alpha -> latent h interpolators (linear per dimension).
        self._build_calpha_interpolators(model_dict)

        log.info(
            "PI-SVGP loaded: M=%d D_input=%d D_hidden=%d outputs=%d task=%d (mu~%.4f)",
            M, D_input, D_hidden, num_outputs, task_index,
            TASK_FRICTION_MAP.get(task_index, -1),
        )

        # --- Load GP action prior model ---
        action_prior_path = getattr(config, 'action_prior_model_path', None)
        self._action_prior_model_type = getattr(config, 'action_prior_model_type', 'svgp')
        self._has_action_prior = False

        if action_prior_path:
            try:
                if self._action_prior_model_type == "sgpr_krasserm":
                    self._ap_models, self._ap_x_scaler, self._ap_y_scaler, ap_config = \
                        load_model_krasserm(action_prior_path, device="cpu")
                    self._ap_input_cols = ap_config["input_cols"]
                    log.info("Loaded SGPR Krasserm action prior from %s", action_prior_path)
                else:
                    self._ap_model, self._ap_likelihood, self._ap_x_scaler, self._ap_y_scaler, ap_config = \
                        load_model_svgp(action_prior_path, device="cpu")
                    self._ap_input_cols = ap_config.get("input_cols", None)
                    self._ap_use_avg_kappa = ap_config.get("use_avg_kappa", True)
                    self._ap_minimal = ap_config.get("minimal_features", False)
                    log.info("Loaded SVGP action prior from %s (features=%s)",
                             action_prior_path,
                             self._ap_input_cols or ("minimal" if self._ap_minimal else "default"))
                self._has_action_prior = True
            except Exception as e:
                log.warning("Could not load action prior model: %s", e)

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
            "MCTS DPW PI-SVGP C++ CS ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d  "
            "cs_w1=%.2f cs_w2=%.2f cs_w3=%.2f C_alpha_th=%.2f window=%d",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon,
            self._cs_w1, self._cs_w2, self._cs_w3,
            self._C_alpha_th, self._calpha_window_size,
        )

    # -- C_alpha -> latent interpolation -----------------------------------

    def _build_calpha_interpolators(self, model_dict) -> None:
        """Fit per-dimension linear maps from C_alpha -> latent h."""
        self._c_alpha_anchors = np.array([104.89, 86.42, 49.47])
        h_anchors = np.array(model_dict['h_mu_opt'][:3])  # (3, Hdim)
        Hdim = h_anchors.shape[1]

        self._calpha_slopes = np.zeros(Hdim)
        self._calpha_intercepts = np.zeros(Hdim)
        for d in range(Hdim):
            self._calpha_slopes[d], self._calpha_intercepts[d] = np.polyfit(
                self._c_alpha_anchors, h_anchors[:, d], 1
            )

        self._c_alpha_min = float(self._c_alpha_anchors.min())
        self._c_alpha_max = float(self._c_alpha_anchors.max())
        log.info(
            "C_alpha interpolators built: range [%.2f, %.2f] -> Hdim=%d",
            self._c_alpha_min, self._c_alpha_max, Hdim,
        )

    def _interpolate_hp_mu(self, c_alpha: float) -> np.ndarray:
        """Map cornering stiffness to PI-SVGP latent params."""
        c_alpha_clamped = c_alpha
        return self._calpha_slopes * c_alpha_clamped + self._calpha_intercepts

    # -- v_ref adaptation --------------------------------------------------

    def _adapt_v_ref_window(self, v_ref_window: np.ndarray) -> np.ndarray:
        """Reduce v_ref entries when estimated grip is low."""
        delta_C_alpha = max(0.0, self._C_alpha_th - self._cornering_stiffness)
        if delta_C_alpha <= 0.0:
            return v_ref_window

        log_arg = self._cs_w2 * delta_C_alpha + self._cs_w3
        if log_arg <= 0.0:
            return v_ref_window

        reduction = self._cs_w1 * math.log(log_arg)
        adapted = v_ref_window - reduction
        adapted = np.maximum(adapted, 3.0)
        return adapted

    # -- C_alpha estimation ------------------------------------------------

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

    # -- Forward-backward velocity solver ----------------------------------

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity using forward-backward solver.

        1. Per-point curvature limit:  v_limit[i] = sqrt(a_lat_budget / |kappa[i]|)
        2. Backward pass:  propagate braking constraint from corners
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

        # ds between consecutive kappa samples
        ds = max(abs(v_current), 1.0) * self.dt
        N = len(kappas)

        # Step 1: per-point curvature speed limit
        v_limit = np.minimum(v_cap,
                             np.sqrt(self.a_lat_budget / np.maximum(np.abs(kappas), 1e-6)))

        # Step 2: backward pass — brake in time for corners
        for i in range(N - 2, -1, -1):
            v_limit[i] = min(v_limit[i],
                             np.sqrt(v_limit[i + 1] ** 2 + 2 * a_max * ds))

        # Use v_limit[0] as uniform value for the rollout window
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

    # -- GP action prior prediction ----------------------------------------

    def _predict_action_prior(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """Run GP inference to get action prior (mean, std) in original scale."""
        if not self._has_action_prior:
            return np.zeros(2), np.ones(2) * 5.0  # wide uniform-like fallback
        if self._action_prior_model_type == "sgpr_krasserm":
            return self._predict_krasserm(v, delta, omega, beta, max_kappa, avg_kappa, c_alpha)
        return self._predict_svgp(v, delta, omega, beta, max_kappa, avg_kappa, c_alpha)

    def _predict_svgp(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """SVGP multi-output prediction."""
        if self._ap_input_cols is not None:
            local = {"v": v, "delta": delta, "omega": omega, "beta": beta,
                     "max_kappa": max_kappa, "avg_kappa": avg_kappa, "c_alpha": c_alpha}
            vals = [local[self._COL_TO_ARG[c]] for c in self._ap_input_cols]
            x_raw = np.array([vals], dtype=np.float32)
        elif self._ap_minimal:
            x_raw = np.array([[v, max_kappa, c_alpha]], dtype=np.float32)
        elif self._ap_use_avg_kappa:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, avg_kappa, c_alpha]],
                             dtype=np.float32)
        else:
            x_raw = np.array([[v, delta, omega, beta, max_kappa, c_alpha]],
                             dtype=np.float32)
        x_scaled = self._ap_x_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self._ap_likelihood(self._ap_model(x_tensor))

        mean_scaled = preds.mean[0].numpy()
        var_scaled = preds.variance[0].numpy()

        mean_original = mean_scaled * self._ap_y_scaler.scale_ + self._ap_y_scaler.mean_
        std_original = np.sqrt(var_scaled) * self._ap_y_scaler.scale_
        return mean_original, std_original

    def _predict_krasserm(self, v, delta, omega, beta, max_kappa, avg_kappa, c_alpha):
        """SGPR Krasserm independent single-output prediction."""
        local = {"v": v, "delta": delta, "omega": omega, "beta": beta,
                 "max_kappa": max_kappa, "avg_kappa": avg_kappa, "c_alpha": c_alpha}
        vals = [local[self._COL_TO_ARG[c]] for c in self._ap_input_cols]
        x_raw = np.array([vals], dtype=np.float64)
        x_scaled = self._ap_x_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float64)

        mean_scaled = np.zeros(2)
        var_scaled = np.zeros(2)
        with torch.no_grad():
            for i, model in enumerate(self._ap_models):
                f_mean, _, y_var = model.predict(x_tensor)
                mean_scaled[i] = f_mean[0].item()
                var_scaled[i] = y_var[0].item()

        mean_original = mean_scaled * self._ap_y_scaler.scale_ + self._ap_y_scaler.mean_
        std_original = np.sqrt(var_scaled) * self._ap_y_scaler.scale_
        return mean_original, std_original

    # -- Tree management ---------------------------------------------------

    def _make_tree(self, state_np, v_ref_window):
        """Create a fresh C++ DPW tree at the given state."""
        p = self._vehicle_params
        return F1TenthDPW(
            initial_obs=state_np,
            alpha=self.alpha,
            beta=self.beta,
            K=self.K,
            # Vehicle params
            mu=p.mu, C_Sf=p.C_Sf, C_Sr=p.C_Sr,
            lf=p.lf, lr=p.lr, h=p.h,
            m=p.m, I=p.I,
            s_min=p.s_min, s_max=p.s_max,
            sv_min=p.sv_min, sv_max=p.sv_max,
            v_switch=p.v_switch, a_max=p.a_max,
            v_min=p.v_min, v_max=p.v_max,
            width=p.width, length=p.length,
            # Occupancy map (imaginary)
            occ_grid=self._occ_grid,
            occ_x_min=self._occ_x_min,
            occ_y_min=self._occ_y_min,
            occ_cell_size=self._occ_cell_size,
            occ_nx=self._occ_nx,
            occ_ny=self._occ_ny,
            # Occupancy map (real)
            occ_grid_real=self._occ_grid_real,
            # Cost params
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            v_ref=self.v_ref,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount,
            dt=self.dt,
            normalize_reward=self.normalize_reward,
            # PI-SVGP params
            svgp_data=self._svgp_data,
            svgp_M=self._svgp_M,
            svgp_D_input=self._svgp_D_input,
            svgp_D_hidden=self._svgp_D_hidden,
            svgp_num_outputs=self._svgp_num_outputs,
            base_timestep=self.base_timestep,
            seed=self.seed,
        )

    # -- Public interface --------------------------------------------------

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # 0. Startup: full throttle + zero steer until we reach 1.5 m/s
        self._timestep += 1
        if abs(state.v) < 1.5 and self._timestep <= 20:
            self._prev_state = state
            self._tree = None
            self._last_action = None
            return ControlCmdMsg(
                acceleration=self._vehicle_params.a_max,
                steering_vel=0.0,
            )

        # 1. Estimate cornering stiffness + smooth
        c_alpha_mean = self._estimate_cornering_stiffness(state)

        # 2. Interpolate smoothed C_alpha -> new hp_mu latent params
        hp_mu_new = self._interpolate_hp_mu(c_alpha_mean)

        # 3. Compute curvature + v_ref window (forward-backward solver)
        v_ref_window = self._get_v_ref_window(state_np)

        # 4. Adapt v_ref for grip
        v_ref_window = self._adapt_v_ref_window(v_ref_window)

        # 5. GP action prior inference
        action_mean, action_std = self._predict_action_prior(
            state.v, state.delta, state.omega, state.beta,
            self._max_kappa_lookahead, self._avg_kappa_lookahead, c_alpha_mean,
        )
        self._last_action_mean = action_mean
        self._last_action_std = action_std

        # 6. Cache state
        self._prev_state = state

        # 7. Tree reuse or creation
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.set_state(state_np)
                self._tree.set_hp_mu(hp_mu_new)
                self._tree.set_v_ref_window(v_ref_window)
            except RuntimeError:
                self._tree = None

        if self._tree is None:
            self._tree = self._make_tree(state_np, v_ref_window)
            self._tree.set_hp_mu(hp_mu_new)
            self._tree.set_v_ref_window(v_ref_window)

        # 8. Push action prior, dynamic horizon, and local centerline to C++
        self._tree.set_action_prior(action_mean, action_std)
        self._tree.set_rollout_horizon(self._dynamic_horizon)
        self._tree.set_centerline(self._local_cx, self._local_cy, self._local_heading)

        # 9. Run MCTS
        if self.mcts_time_budget > 0:
            n_sims = self._tree.learn_timed(self.mcts_time_budget)
        else:
            self._tree.learn(self.n_simulations)
            n_sims = self.n_simulations
        log.debug("MCTS ran %d simulations", n_sims)

        # 10. Extract best trajectory from tree (for visualization)
        try:
            self._last_best_trajectory = self._tree.get_best_trajectory(
                self._dynamic_horizon)
        except Exception:
            self._last_best_trajectory = []

        # 11. Select best action
        action = self._tree.best_action()
        self._last_action = action

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset the tree and C_alpha state (e.g. at episode start)."""
        self._tree = None
        self._last_action = None
        self._prev_state = None
        self._cornering_stiffness = _C_ALPHA_NOMINAL
        self._calpha_buffer = []
        self._timestep = 0

    # -- Introspection helpers ---------------------------------------------

    def get_cornering_stiffness(self) -> float:
        return self._cornering_stiffness

    def get_smoothed_calpha(self) -> float:
        if not self._calpha_buffer:
            return _C_ALPHA_NOMINAL
        return float(np.mean(self._calpha_buffer))

    def get_max_kappa(self) -> float:
        return self._max_kappa_lookahead

    def get_avg_kappa(self) -> float:
        return self._avg_kappa_lookahead

    def get_action_prior_mean(self) -> np.ndarray:
        return self._last_action_mean

    def get_action_prior_std(self) -> np.ndarray:
        return self._last_action_std

    def get_best_trajectory(self) -> list:
        return getattr(self, '_last_best_trajectory', [])
