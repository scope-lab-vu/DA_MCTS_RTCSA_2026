"""
MCTS DPW controller for F1TENTH racing — C++ accelerated.

Same interface as the Python version (mcts_dpw_py) but delegates tree search,
dynamics evaluation, and rollouts to a compiled C++ extension for speed.
"""

import logging

import numpy as np

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

log = logging.getLogger(__name__)


class MCTSDPWCppController:
    """MCTS DPW controller for F1TENTH racing (C++ accelerated)."""

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

        # MCTS hyperparameters.
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
        # Load track (centerline CSV).
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS DPW C++ controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

        # Occupancy map for O(1) off-track detection.
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.001,
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

        # Tree state.
        self._tree = None
        self._last_action = None
        self._max_kappa_lookahead = 0.0
        self._dynamic_horizon = self.rollout_horizon

        log.info(
            "MCTS DPW C++ ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon,
        )

    def _get_v_ref_window(self, state_np):
        """Compute per-step reference velocity from curvature lookahead.

        For each rollout step i, v_ref[i] is based on the max curvature
        from step i onward within the lookahead window. This ensures the
        car starts braking before a turn, not when it's already in it.

        Also computes the dynamic rollout horizon based on max kappa:
        straight (low kappa) → short horizon, curve → long horizon.
        """
        # v_max = self._vehicle_params.v_max
        v_max = 12.0
        kappas = self.track.get_curvature_lookahead(
            position=state_np[:2], v=state_np[3],
            horizon=self.curvature_lookahead, dt=self.dt,
        )

        self._max_kappa_lookahead = float(np.max(kappas))

        # Dynamic horizon: short on straights, long near curves.
        if self._max_kappa_lookahead < self.kappa_threshold:
            self._dynamic_horizon = self.rollout_horizon_straight
        else:
            self._dynamic_horizon = self.rollout_horizon_curve

        # Always size window to the max possible horizon (curve) so C++
        # has enough entries regardless of dynamic horizon.
        window_size = self.rollout_horizon_curve
        v_ref_window = np.zeros(window_size)
        for i in range(window_size):
            max_kappa = np.max(kappas[i:])
            v_ref_window[i] = min(v_max, np.sqrt(self.a_lat_budget / max(max_kappa, 1e-6)))

        min_v_ref = min(v_ref_window)
        for i in range(window_size):
            v_ref_window[i] = min_v_ref

        self._v_ref_window = v_ref_window.copy()
        self._min_v_ref_curv = float(v_ref_window[0])
        return v_ref_window

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

        # Set dynamic rollout horizon and curve flag based on curvature.
        self._tree.set_rollout_horizon(self._dynamic_horizon)
        self._tree.set_is_curve(self._max_kappa_lookahead >= self.kappa_threshold)

        # Run MCTS: fixed count for determinism, time budget for real-time.
        if self.mcts_time_budget > 0:
            n_sims = self._tree.learn_timed(self.mcts_time_budget)
        else:
            self._tree.learn(self.n_simulations)
            n_sims = self.n_simulations
        log.debug("MCTS ran %d simulations", n_sims)

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
