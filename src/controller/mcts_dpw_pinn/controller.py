"""
MCTS DPW controller with PINN dynamics and batched GPU evaluation.

Uses known F1TENTH dynamics for fast tree traversal (selection/expansion)
and batched PINN inference on GPU for leaf rollout evaluation.  Feature
histories are built during selection and passed to the PINN evaluator.
"""

import logging

import numpy as np
import torch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import Track, TrackOccupancyMap, ReferenceTrajectory
from src.dynamics.f1tenth_dynamics import F1TenthDynamicsModel

from src.controller.mcts_dpw_pinn.simulator import F1TenthSelectionSimulator
from src.controller.mcts_dpw_pinn.evaluator import BatchedPINNEvaluator
from src.controller.mcts_dpw_pinn.tree import BatchedDPW

log = logging.getLogger(__name__)


class MCTSDPWPINNController:
    """MCTS DPW controller with PINN dynamics and batched GPU evaluation."""

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig,
        simulation_config: SimulationConfig = None,
    ) -> None:
        self._pinn_model = dynamics_model  # PINNModel
        self._vehicle_params = vehicle_params
        self._config = config

        self.dt = simulation_config.dt_s if simulation_config is not None else 0.04

        # Create known dynamics model for fast tree traversal.
        self._known_dynamics = F1TenthDynamicsModel(vehicle_params)

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
        self.batch_size = getattr(config, "batch_size", 8)
        self.eval_K = getattr(config, "eval_K", 4)

        # Determine evaluation device.
        eval_device_str = getattr(config, "eval_device", "cuda")
        if eval_device_str == "cuda" and torch.cuda.is_available():
            self._eval_device = "cuda"
        else:
            self._eval_device = "cpu"
        log.info("MCTS DPW PINN eval device: %s", self._eval_device)

        # Load track centerline.
        if map_config.centerline_path is None:
            raise ValueError(
                "MCTS DPW PINN controller requires centerline_path in map config."
            )
        self.track = Track(map_config.centerline_path)

        # Occupancy map for off-track detection.
        self.track_occ_map = TrackOccupancyMap(
            track=self.track,
            cell_size=0.01,
            lane_width=config.lane_width,
        )

        # Reference trajectory for per-step v_ref.
        self.ref_traj = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning(
                    "Could not load reference trajectory: %s. Using constant v_ref=%.1f",
                    e, self.v_ref,
                )

        # Create batched PINN evaluator.
        self._evaluator = BatchedPINNEvaluator(
            pinn_model=self._pinn_model,
            vehicle_params=vehicle_params,
            track_occ_map=self.track_occ_map,
            v_ref=self.v_ref,
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount,
            dt=self.dt,
            device=self._eval_device,
            K=self.eval_K,
        )

        # PINN base timestep for feature extraction.
        self._pinn_base_dt = self._pinn_model._base_timestep

        # Tree state.
        self._tree = None
        self._last_action = None
        self._last_action_tensor = None

        log.info(
            "MCTS DPW PINN ready: n_sim=%d K=%.2f alpha=%.2f beta=%.2f "
            "discount=%.2f rollout_horizon=%d batch_size=%d eval_K=%d device=%s",
            self.n_simulations, self.K, self.alpha, self.beta,
            self.discount, self.rollout_horizon, self.batch_size,
            self.eval_K, self._eval_device,
        )

    def _get_feature_history(self):
        """Get current real feature history from PINN model."""
        if self._pinn_model._feature_history is None:
            pinn_horizon = self._pinn_model._pinn_horizon
            return torch.zeros(
                pinn_horizon, 6, dtype=torch.float32,
            )
        return self._pinn_model._feature_history.cpu().clone()

    def _make_simulator(self, state_np, feature_history):
        """Create a selection simulator at the given state."""
        if self.ref_traj is not None:
            v_ref_window = self.ref_traj.get_reference_velocity_window(
                state_np[:2], self.rollout_horizon, self.dt,
            )
        else:
            v_ref_window = np.full(self.rollout_horizon, self.v_ref)

        return F1TenthSelectionSimulator(
            state_array=state_np,
            known_dynamics_model=self._known_dynamics,
            vehicle_params=self._vehicle_params,
            track_occ_map=self.track_occ_map,
            dt=self.dt,
            v_ref=self.v_ref,
            v_ref_window=v_ref_window,
            speed_weight=self.speed_weight,
            alpha_threshold=self.alpha_threshold,
            rollout_horizon=self.rollout_horizon,
            discount=self.discount,
            feature_history=feature_history,
            pinn_base_dt=self._pinn_base_dt,
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
        state_tensor = torch.tensor(state_np, dtype=torch.float32)

        # Update PINN real feature history with actual observation.
        self._pinn_model.observe(state_tensor, self._last_action_tensor)

        # Get feature history for tree search.
        feature_history = self._get_feature_history()

        # Create selection simulator.
        sim = self._make_simulator(state_np, feature_history)

        # Try tree reuse.
        if self._tree is not None and self._last_action is not None:
            try:
                self._tree.forward(self._last_action, state_np)
                self._tree.env = sim
            except RuntimeWarning:
                self._tree = None

        # Create fresh tree if needed.
        if self._tree is None:
            self._tree = BatchedDPW(
                alpha=self.alpha,
                beta=self.beta,
                initial_obs=state_np,
                env=sim,
                K=self.K,
                evaluator=self._evaluator,
                batch_size=self.batch_size,
            )

        # Run MCTS simulations.
        self._tree.learn(self.n_simulations)

        # Select best action.
        action = self._tree.best_action()
        self._last_action = action
        self._last_action_tensor = torch.tensor(
            [float(action[0]), float(action[1])], dtype=torch.float32,
        )

        return ControlCmdMsg(
            acceleration=float(action[0]),
            steering_vel=float(action[1]),
        )

    def reset(self):
        """Reset tree and PINN state."""
        self._tree = None
        self._last_action = None
        self._last_action_tensor = None
        self._pinn_model.reset()
