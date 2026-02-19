"""
MCTS DPW controller + PI-SVGP dynamics — C++ accelerated.

Uses a multi-output GP to predict 9 Pacejka tire parameters,
then applies Pacejka physics for dynamics rollouts, all in C++.
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
    from mcts_f1tenth_pi_svgp_cpp import F1TenthDPWPiSvgp as F1TenthDPW
except ImportError as e:
    raise ImportError(
        "mcts_f1tenth_pi_svgp_cpp C++ extension not found. "
        "Build with: pip install -e src/controller/mcts_pi_svgp_cpp/"
    ) from e

log = logging.getLogger(__name__)

# Task index -> friction mapping (for logging)
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}


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


class MCTSPiSvgpCppController:
    """MCTS DPW controller with PI-SVGP dynamics (C++ accelerated)."""

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

        # Load track (centerline CSV).
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS PI-SVGP C++ controller requires centerline_path in map config."
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

        log.info(
            "PI-SVGP loaded: M=%d D_input=%d D_hidden=%d outputs=%d task=%d (mu~%.4f)",
            M, D_input, D_hidden, num_outputs, task_index,
            TASK_FRICTION_MAP.get(task_index, -1),
        )

        # Tree state.
        self._tree = None
        self._last_action = None

        log.info(
            "MCTS DPW PI-SVGP C++ ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
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
            # PI-SVGP params
            svgp_data=self._svgp_data,
            svgp_M=self._svgp_M,
            svgp_D_input=self._svgp_D_input,
            svgp_D_hidden=self._svgp_D_hidden,
            svgp_num_outputs=self._svgp_num_outputs,
            base_timestep=self.base_timestep,
            seed=self.seed,
        )

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        # Low-speed override: full throttle, no steering until v > 2.0.
        if abs(state.v) < 2.0:
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
        self._tree.learn(self.n_simulations)

        # Select best action (most visited).
        action = self._tree.best_action()
        self._last_action = action

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset the tree (e.g. at episode start)."""
        self._tree = None
        self._last_action = None
