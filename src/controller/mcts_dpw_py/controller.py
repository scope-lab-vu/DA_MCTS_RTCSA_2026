"""
MCTS DPW controller for F1TENTH racing.

Implements the Controller interface using Monte Carlo Tree Search
with Double Progressive Widening for action selection.
"""

import logging

import numpy as np

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory
from src.controller.mcts_dpw_py.tree import DPW
from src.controller.mcts_dpw_py.simulator import F1TenthSimulator

log = logging.getLogger(__name__)


class MCTSDPWController:
    """MCTS DPW controller for F1TENTH racing."""

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
                "MCTS DPW controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

        # Occupancy map for O(1) off-track detection.
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.01,
            lane_width=config.lane_width,
        )

        # Reference trajectory for per-step v_ref (same as MPPI).
        self.ref_traj = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning("Could not load reference trajectory: %s. Using constant v_ref=%.1f", e, self.v_ref)

        # Tree state.
        self._tree = None
        self._last_action = None

        log.info(
            "MCTS DPW ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d dynamics=%s",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon,
            type(dynamics_model).__name__,
        )

    def _make_simulator(self, state_np):
        """Create a fresh F1TenthSimulator at the given state."""
        # Compute per-step reference velocity window (same as MPPI).
        if self.ref_traj is not None:
            v_ref_window = self.ref_traj.get_reference_velocity_window(
                state_np[:2], self.rollout_horizon, self.dt,
            )
        else:
            v_ref_window = np.full(self.rollout_horizon, self.v_ref)

        return F1TenthSimulator(
            state_array=state_np,
            dynamics_model=self._dynamics_model,
            vehicle_params=self._vehicle_params,
            track=self.track,
            track_occ_map=self.track_occ_map,
            dt=self.dt,
            v_ref=self.v_ref,
            v_ref_window=v_ref_window,
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount,
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

        sim = self._make_simulator(state_np)

        # Try to reuse the tree from the previous step.
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.env = sim
            except RuntimeWarning:
                self._tree = None

        # Create a fresh tree if reuse failed or first call.
        if self._tree is None:
            self._tree = DPW(
                alpha=self.alpha,
                beta=self.beta,
                initial_obs=state_np,
                env=sim,
                K=self.K,
            )

        # Run MCTS simulations.
        self._tree.learn(self.n_simulations)

        # Select best action (most visited).
        action = self._tree.best_action()
        self._last_action = action

        # Log the best action's mean reward from the tree.
        action_key = tuple(action)
        if action_key in self._tree.root.children:
            rn = self._tree.root.children[action_key]
            mean_rew = rn.cumulative_reward / max(rn.visits, 1)
            log.info("best_action=[%.3f, %.3f] visits=%d mean_reward=%.4f",
                     action[0], action[1], rn.visits, mean_rew)

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset the tree (e.g. at episode start)."""
        self._tree = None
        self._last_action = None
