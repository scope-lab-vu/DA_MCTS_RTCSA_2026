"""
MCTS DPW controller + SVGP dynamics — C++ accelerated.

Uses Extended Kinematic base model with SVGP residual corrections for
omega (yaw rate) and beta (slip angle), all running in C++.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
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

log = logging.getLogger(__name__)

# Task index → friction mapping (for logging)
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}


# ---------------------------------------------------------------------------
# Helper: load SVGP model from .npz (same loader as src/dynamics/svgp.py)
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
            xmean(D_input) | xscale(D_input) | ymean(1) | yscale(1) | hp_mu(D_hidden)
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


class MCTSDPWSvgpCppController:
    """MCTS DPW controller with SVGP dynamics (C++ accelerated)."""

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

        # Load track (centerline CSV).
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS DPW SVGP C++ controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

        # Occupancy map for O(1) off-track detection.
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.01,
            lane_width=config.lane_width,
        )

        # Extract occupancy map data for C++ (flat numpy array).
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

        # --- Load SVGP models ---
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
            "SVGP loaded: M=%d D_input=%d D_hidden=%d task=%d (mu≈%.4f)",
            M, D_input, D_hidden, task_index, TASK_FRICTION_MAP.get(task_index, -1),
        )

        # Tree state.
        self._tree = None
        self._last_action = None

        log.info(
            "MCTS DPW SVGP C++ ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon,
        )

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
            # SVGP params
            omega_data=self._omega_data,
            beta_data=self._beta_data,
            svgp_M=self._svgp_M,
            svgp_D_input=self._svgp_D_input,
            svgp_D_hidden=self._svgp_D_hidden,
            base_timestep=self.base_timestep,
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

        v_ref_window = self._get_v_ref_window(state_np)

        # Try to reuse the tree from the previous step.
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.set_state(state_np)
                self._tree.set_v_ref_window(v_ref_window)
            except RuntimeError:
                self._tree = None

        # Create a fresh tree if reuse failed or first call.
        if self._tree is None:
            self._tree = self._make_tree(state_np, v_ref_window)
            self._tree.set_v_ref_window(v_ref_window)

        # Run MCTS simulations.
        t0 = time.monotonic()
        self._tree.learn(self.n_simulations)
        mcts_ms = (time.monotonic() - t0) * 1000.0

        # Select best action (most visited).
        action = self._tree.best_action()
        self._last_action = action

        # log.info("mcts_time=%.1fms", mcts_ms)

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset the tree (e.g. at episode start)."""
        self._tree = None
        self._last_action = None
