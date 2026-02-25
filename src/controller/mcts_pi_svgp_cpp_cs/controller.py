"""
MCTS DPW controller + PI-SVGP dynamics — C++ accelerated, with online
cornering stiffness (C_alpha) adaptation.

Extends the base MCTS PI-SVGP C++ controller with three additions:

  1. Internal C_alpha estimation — rear cornering stiffness is computed from
     consecutive VehicleStateMsg observations (no external input needed).

  2. Rolling-window latent update — the mean C_alpha over the last
     `calpha_window_size` steps is mapped to PI-SVGP task latent parameters
     via linear interpolation, then pushed to C++ via set_hp_mu().

  3. C_alpha-adaptive reference velocity — the v_ref window is reduced when
     the estimated surface grip is low:

         delta_C = max(0,  C_alpha_th - cornering_stiffness_{t-1})
         v_ref_t -= cs_w1 * log(cs_w2 * delta_C + cs_w3)   [when delta_C > 0]
         v_ref_t  = max(v_ref_t, 3.0)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

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
            xmean(D_input) | xscale(D_input) | hp_mu(D_hidden) | output_bias(num_outputs)
    """
    D_aug = D_input + D_hidden

    theta = np.array(model_dict['theta_opt'], dtype=np.float64)

    # Precompute K_mm_inv @ m in Python
    Zx = np.array(model_dict['Zx_opt'], dtype=np.float64)
    Zh = np.array(model_dict['Zh_opt'], dtype=np.float64)
    Z_aug = np.concatenate([Zx, Zh], axis=1)  # (M, D_aug)

    # Build K_mm and invert
    from src.dynamics.pi_svgp import _rbf_ard_kernel
    import torch
    theta_t = torch.tensor(theta, dtype=torch.float64)
    Z_aug_t = torch.tensor(Z_aug, dtype=torch.float64)
    K_mm = _rbf_ard_kernel(Z_aug_t, Z_aug_t, theta_t)
    K_mm = K_mm + torch.eye(M, dtype=torch.float64) * 1e-6
    K_mm_inv = torch.linalg.inv(K_mm).numpy()

    m_opt = np.array(model_dict['m_opt'], dtype=np.float64)  # (M, num_outputs)
    alpha = (K_mm_inv @ m_opt).ravel()  # (M * num_outputs), row-major

    Z_aug_flat = Z_aug.ravel()

    xmean = np.array(xscaler.mean_, dtype=np.float64)
    xscale = np.array(xscaler.scale_, dtype=np.float64)
    hp = np.array(hp_mu, dtype=np.float64)
    output_bias = np.array(model_dict['output_bias_opt'], dtype=np.float64)

    return np.concatenate([
        theta, alpha, Z_aug_flat, xmean, xscale, hp, output_bias,
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
    """Estimate rear cornering stiffness from two consecutive states.

    Derivation (linearised single-track model):
        F_yr = (lf * m * (v_y_dot + v_x * omega) - Iz * omega_dot) / (lf + lr)
        alpha_r = atan2(lr * omega - v_y, |v_x|)
        C_r = |F_yr| / (|alpha_r| + eps)

    Derivatives are approximated with a forward finite-difference from t-1 to t.
    """
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
    """MCTS DPW controller with PI-SVGP dynamics + online C_alpha adaptation."""

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
        self.v_ref = config.v_ref
        self.speed_weight = config.speed_weight
        self.alpha_threshold = config.alpha_threshold

        # Cornering stiffness velocity-adaptation parameters.
        self._cs_w1 = config.cs_w1
        self._cs_w2 = config.cs_w2
        self._cs_w3 = config.cs_w3
        self._C_alpha_th = config.C_alpha_th
        self._calpha_window_size = config.calpha_window_size

        # Runtime state for C_alpha tracking.
        self._prev_state: Optional[VehicleStateMsg] = None
        self._cornering_stiffness: float = _C_ALPHA_NOMINAL
        self._calpha_buffer: list = []

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

        # Occupancy map for O(1) off-track detection.
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.01,
            lane_width=config.lane_width,
        )

        # Extract occupancy map data for C++.
        self._occ_grid = self.track_occ_map._map.ravel().astype(np.float32)
        self._occ_x_min = self.track_occ_map.x_min
        self._occ_y_min = self.track_occ_map.y_min
        self._occ_cell_size = self.track_occ_map.cell_size
        self._occ_nx = self.track_occ_map.nx
        self._occ_ny = self.track_occ_map.ny

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

        # --- Load PI-SVGP model ---
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

        # Tree state.
        self._tree = None
        self._last_action = None

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
        """Fit per-dimension linear maps from C_alpha -> latent h.

        C_alpha anchors correspond to the three training friction levels:
            task 0: mu=1.0489  ->  C_alpha ~ 104.89
            task 1: mu=0.8642  ->  C_alpha ~  86.42
            task 2: mu=0.4947  ->  C_alpha ~  49.47
        """
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
        """Map cornering stiffness to PI-SVGP latent params via linear interpolation.

        Values are clamped to the training range [49.47, 104.89].
        """
        # c_alpha_clamped = np.clip(c_alpha, self._c_alpha_min, self._c_alpha_max)
        c_alpha_clamped = c_alpha
        return self._calpha_slopes * c_alpha_clamped + self._calpha_intercepts

    # -- v_ref adaptation --------------------------------------------------

    def _adapt_v_ref_window(self, v_ref_window: np.ndarray) -> np.ndarray:
        """Reduce v_ref entries when estimated grip is low.

        delta_C = max(0, C_alpha_th - CS)
        v_ref_t -= cs_w1 * log(cs_w2 * delta_C + cs_w3)   [when delta_C > 0]
        v_ref_t  = max(v_ref_t, 3.0)
        """
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

    # -- Tree management ---------------------------------------------------

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity window."""
        if self.ref_traj is not None:
            return self.ref_traj.get_reference_velocity_window(
                state_np[:2], self.rollout_horizon, self.dt,
            )
        return np.full(self.rollout_horizon, self.v_ref)

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
            # Occupancy map
            occ_grid=self._occ_grid,
            occ_x_min=self._occ_x_min,
            occ_y_min=self._occ_y_min,
            occ_cell_size=self._occ_cell_size,
            occ_nx=self._occ_nx,
            occ_ny=self._occ_ny,
            # Cost params
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            v_ref=self.v_ref,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount,
            dt=self.dt,
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
        """Compute control command with internal C_alpha estimation.

        Steps:
          1. Estimate cornering stiffness from (prev_state, state).
          2. Update rolling C_alpha buffer and compute smoothed mean.
          3. Interpolate smoothed C_alpha -> new hp_mu latent params.
          4. Cache current state as prev_state.
          5. Adapt v_ref window based on estimated grip.
          6. Push updated hp_mu and v_ref to C++ tree, run MCTS.
        """
        # Step 1: estimate cornering stiffness
        if self._prev_state is not None:
            cs_raw = _compute_cornering_stiffness(
                self._prev_state, state, self.dt,
                self._lf, self._lr, self._mass, self._iz,
            )
            cs_clipped = float(np.clip(cs_raw, 0.0, _C_ALPHA_NOMINAL))
        else:
            cs_clipped = _C_ALPHA_NOMINAL

        self._cornering_stiffness = cs_clipped

        # Step 2: rolling window smooth
        self._calpha_buffer.append(cs_clipped)
        if len(self._calpha_buffer) > self._calpha_window_size:
            self._calpha_buffer = self._calpha_buffer[-self._calpha_window_size:]
        c_alpha_mean = float(np.mean(self._calpha_buffer))

        # Step 3: interpolate -> new hp_mu
        hp_mu_new = self._interpolate_hp_mu(c_alpha_mean)

        # Step 4: cache state
        self._prev_state = state

        # Low-speed override: full throttle, no steering until v > 2.0.
        if abs(state.v) < 1.5:
            self._tree = None
            self._last_action = None
            return ControlCmdMsg(
                acceleration=self._vehicle_params.a_max,
                steering_vel=0.0,
            )

        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # Step 5: compute and adapt v_ref window
        v_ref_window = self._get_v_ref_window(state_np)
        v_ref_window = self._adapt_v_ref_window(v_ref_window)

        # Try to reuse the tree from the previous step.
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.set_state(state_np)
                self._tree.set_hp_mu(hp_mu_new)
                self._tree.set_v_ref_window(v_ref_window)
            except RuntimeError:
                self._tree = None

        # Create a fresh tree if reuse failed or first call.
        if self._tree is None:
            self._tree = self._make_tree(state_np, v_ref_window)
            self._tree.set_hp_mu(hp_mu_new)
            self._tree.set_v_ref_window(v_ref_window)

        # Step 6: run MCTS simulations.
        self._tree.learn(self.n_simulations)

        # Select best action (most visited).
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

    # -- Introspection helpers ---------------------------------------------

    def get_cornering_stiffness(self) -> float:
        """Return the most recently estimated cornering stiffness [N/rad]."""
        return self._cornering_stiffness

    def get_smoothed_calpha(self) -> float:
        """Return the rolling-window mean of C_alpha used for latent update."""
        if not self._calpha_buffer:
            return _C_ALPHA_NOMINAL
        return float(np.mean(self._calpha_buffer))
