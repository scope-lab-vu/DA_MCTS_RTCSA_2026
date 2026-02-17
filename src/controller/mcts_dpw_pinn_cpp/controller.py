"""
MCTS DPW controller for F1TENTH racing — C++ accelerated, PINN dynamics.

Same MCTS tree search as the analytical-dynamics version, but replaces the
single-track model with a traced TorchScript PINN loaded via LibTorch in C++.
"""

import logging
from pathlib import Path

import numpy as np

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory

try:
    from mcts_f1tenth_pinn_cpp import F1TenthDPW
except ImportError as e:
    raise ImportError(
        "mcts_f1tenth_pinn_cpp C++ extension not found. "
        "Build it with: pip install -e src/controller/mcts_dpw_pinn_cpp/"
    ) from e

log = logging.getLogger(__name__)

# PINN constants (must match PINNConfig in f1tenth_simulator.hpp)
PINN_HORIZON = 10
PINN_NUM_FEATURES = 6
PINN_BASE_TIMESTEP = 0.02


class MCTSDPWPinnCppController:
    """MCTS DPW controller with PINN dynamics (C++ accelerated)."""

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

        # MCTS hyperparameters
        self.n_simulations = config.n_simulations
        self.K = config.exploration_constant
        self.alpha = config.alpha_dpw
        self.beta = config.beta_dpw
        self.discount = config.discount
        self.rollout_horizon = config.rollout_horizon
        self.v_ref = config.v_ref
        self.speed_weight = config.speed_weight
        self.alpha_threshold = config.alpha_threshold

        # ── PINN model paths ────────────────────────────────────────────────
        project_root = Path(__file__).resolve().parents[3]
        self._model_path = str(
            project_root / "models" / "pinn" / "weights" / "pinn_traced.pt"
        )

        scaler_path = project_root / "models" / "pinn" / "weights" / "scaler_safe.npz"
        scaler_data = np.load(scaler_path, allow_pickle=True)
        self._scaler_mean = scaler_data["mean"].astype(np.float32)
        self._scaler_scale = scaler_data["scale"].astype(np.float32)

        # ── Feature history (maintained in Python, passed to C++) ────────────
        self._n_substeps = max(1, round(self.dt / PINN_BASE_TIMESTEP))
        self._feature_history = np.zeros(
            (PINN_HORIZON, PINN_NUM_FEATURES), dtype=np.float32
        )
        self._prev_action = None  # previous real action (for feature extraction)

        # ── Track + occupancy map ────────────────────────────────────────────
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS DPW PINN C++ controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

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

        # ── Reference trajectory ─────────────────────────────────────────────
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

        # Tree state
        self._tree = None
        self._last_action = None

        log.info(
            "MCTS DPW PINN C++ ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d dt=%.3f n_substeps=%d "
            "model=%s",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon, self.dt, self._n_substeps,
            self._model_path,
        )

    # ── Feature history management ───────────────────────────────────────────

    def _update_feature_history(self, state_np, prev_action):
        """Update sliding-window feature history with a real observation.

        Mirrors PINNModel._extract_features + observe() in pinn.py.
        """
        if prev_action is None:
            return

        v = state_np[3]
        beta = state_np[6]
        omega = state_np[5]
        delta = state_np[2]
        accl = prev_action[0]
        steer_vel = prev_action[1]

        vx = v * np.cos(beta)
        vy = v * np.sin(beta)

        # Feature timestep matches chaining logic
        ts = PINN_BASE_TIMESTEP if self._n_substeps > 1 else self.dt

        features = np.array([
            vx, vy, omega, delta,
            accl * ts, steer_vel * ts,
        ], dtype=np.float32)

        # Slide window: drop oldest, append new
        self._feature_history = np.roll(self._feature_history, -1, axis=0)
        self._feature_history[-1] = features

    # ── Reference velocity ───────────────────────────────────────────────────

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity window."""
        if self.ref_traj is not None:
            return self.ref_traj.get_reference_velocity_window(
                state_np[:2], self.rollout_horizon, self.dt,
            )
        return np.full(self.rollout_horizon, self.v_ref)

    # ── Tree creation ────────────────────────────────────────────────────────

    def _make_tree(self, state_np, v_ref_window):
        """Create a fresh C++ DPW tree at the given state."""
        p = self._vehicle_params
        tree = F1TenthDPW(
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
            # PINN params
            model_path=self._model_path,
            scaler_mean=self._scaler_mean,
            scaler_scale=self._scaler_scale,
            feature_history=self._feature_history.ravel(),
        )
        tree.set_v_ref_window(v_ref_window)
        return tree

    # ── Main control loop entry point ────────────────────────────────────────

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # Update feature history with the new real observation
        self._update_feature_history(state_np, self._prev_action)

        v_ref_window = self._get_v_ref_window(state_np)

        # Try to reuse the tree from the previous step
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.set_state(state_np)
                self._tree.set_feature_history(self._feature_history.ravel())
                self._tree.set_v_ref_window(v_ref_window)
            except RuntimeError:
                self._tree = None

        # Create a fresh tree if reuse failed or first call
        if self._tree is None:
            self._tree = self._make_tree(state_np, v_ref_window)

        # Run MCTS simulations
        self._tree.learn(self.n_simulations)

        # Select best action (most visited)
        action = self._tree.best_action()
        self._last_action = action
        self._prev_action = action  # for next feature history update

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset the tree and feature history (e.g. at episode start)."""
        self._tree = None
        self._last_action = None
        self._prev_action = None
        self._feature_history = np.zeros(
            (PINN_HORIZON, PINN_NUM_FEATURES), dtype=np.float32
        )
