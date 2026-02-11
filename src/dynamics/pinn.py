"""
PINN (Physics-Informed Neural Network) dynamics model.

Uses a pretrained DeepPacejka PINN to predict [vx, vy, yaw_rate] from a
10-timestep feature history, from which we derive:
    omega = yaw_rate   (directly)
    beta  = atan2(vy, vx)

Kinematic equations handle [x, y, delta, v, psi].

State:  [x, y, delta, v, psi, omega, beta]  (7-dim)
Action: [acceleration, steering_velocity]    (2-dim)
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import DynamicsConfig, SimulationConfig, VehicleParams

log = logging.getLogger(__name__)


class PINNModel:
    """PINN-based dynamics model for MPPI."""

    def __init__(
        self,
        vehicle_params: VehicleParams,
        dynamics_config: DynamicsConfig = None,
        simulation_config: SimulationConfig = None,
    ) -> None:
        self._p = vehicle_params
        self._g = 9.81

        # Torch batched dynamics state (set via setup_torch).
        self._dt_torch: float = None
        self._device: str = None
        self._dtype: torch.dtype = None

        # ------------------------------------------------------------------
        # Resolve PINN asset paths (relative to project root)
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]

        if dynamics_config is None:
            raise ValueError("PINNModel requires dynamics_config with model paths.")

        weights_path = project_root / dynamics_config.model_path
        cfg_path = project_root / dynamics_config.model_cfg_path
        scaler_path = project_root / dynamics_config.scaler_path
        device_str = dynamics_config.inference_device

        # ------------------------------------------------------------------
        # Timestep / chaining
        # ------------------------------------------------------------------
        self._timestep = simulation_config.dt_s if simulation_config else 0.02
        self._base_timestep = 0.02
        self._n_substeps = max(1, round(self._timestep / self._base_timestep))
        self._use_chaining = self._n_substeps > 1 and self._timestep != 0.05

        if self._use_chaining:
            log.info(
                "PINN chaining: dt=%.3f → %d sub-steps of %.3f",
                self._timestep, self._n_substeps, self._base_timestep,
            )

        # ------------------------------------------------------------------
        # Load PINN architecture config
        # ------------------------------------------------------------------
        with open(cfg_path, "r") as f:
            self._param_dict = yaml.safe_load(f)
        self._pinn_horizon = self._param_dict["MODEL"]["HORIZON"]

        # ------------------------------------------------------------------
        # Build and load PINN model
        # ------------------------------------------------------------------
        from models.pinn.models import string_to_model

        self._pinn = string_to_model[self._param_dict["MODEL"]["NAME"]](
            self._param_dict, eval=True,
        )
        self._pinn_device = torch.device(device_str)
        self._pinn.to(self._pinn_device)
        self._pinn.load_state_dict(
            torch.load(weights_path, map_location=self._pinn_device),
        )
        self._pinn.eval()
        for param in self._pinn.parameters():
            param.requires_grad = False

        self._is_rnn = self._pinn.is_rnn
        log.info("PINN loaded: %s on %s", self._param_dict["MODEL"]["NAME"], device_str)

        # ------------------------------------------------------------------
        # Load feature scaler
        # ------------------------------------------------------------------
        scaler_data = np.load(scaler_path, allow_pickle=True)
        self._scaler_mean = torch.tensor(
            scaler_data["mean"], device=self._pinn_device, dtype=torch.float32,
        )
        self._scaler_scale = torch.tensor(
            scaler_data["scale"], device=self._pinn_device, dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # Feature history buffer — (pinn_horizon, 6)
        # Features: [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd]
        # ------------------------------------------------------------------
        self._num_features = 6
        self._feature_history: Optional[torch.Tensor] = None
        self._rollout_feature_histories: Optional[torch.Tensor] = None

    # -- Observation update (called by controller each real step) -----------

    def observe(self, state: torch.Tensor, prev_action: Optional[torch.Tensor]) -> None:
        """Update feature history with a real (state, action) observation.

        Args:
            state: (7,) tensor [x, y, delta, v, psi, omega, beta]
            prev_action: (2,) tensor [accel, steer_vel] or None (first step)
        """
        if self._feature_history is None:
            self._feature_history = torch.zeros(
                self._pinn_horizon, self._num_features,
                device=self._pinn_device, dtype=torch.float32,
            )

        if prev_action is None:
            return  # first step — history stays zero-padded

        features = self._extract_features(
            state.unsqueeze(0).to(self._pinn_device),
            prev_action.unsqueeze(0).to(self._pinn_device),
        ).squeeze(0)  # (6,)

        self._feature_history = torch.cat([
            self._feature_history[1:],
            features.unsqueeze(0),
        ], dim=0)

    def reset(self) -> None:
        """Clear all internal state."""
        self._feature_history = None
        self._rollout_feature_histories = None

    # -- Feature extraction -------------------------------------------------

    def _extract_features(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """(K, 7) state + (K, 2) action → (K, 6) features."""
        v = state[:, 3]
        beta = state[:, 6]
        omega = state[:, 5]
        delta = state[:, 2]
        accel = action[:, 0]
        steer_vel = action[:, 1]

        vx = v * torch.cos(beta)
        vy = v * torch.sin(beta)
        yaw_rate = omega
        steering_fb = delta

        ts = self._base_timestep if self._use_chaining else self._timestep
        throttle_cmd = accel * ts
        steering_cmd = steer_vel * ts

        return torch.stack(
            [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd], dim=1,
        )

    # -- Batched PINN prediction --------------------------------------------

    def _predict_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        feature_histories: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict omega, beta for a batch using the PINN.

        Returns: (omega_pred, beta_pred, updated_histories)
        """
        new_features = self._extract_features(states, actions)  # (K, 6)

        input_features = torch.cat([
            feature_histories[:, 1:, :],
            new_features.unsqueeze(1),
        ], dim=1)  # (K, pinn_horizon, 6)

        input_features = input_features.to(self._pinn_device)
        features_norm = (input_features - self._scaler_mean) / self._scaler_scale

        with torch.no_grad():
            if self._is_rnn:
                K = states.shape[0]
                h = self._pinn.init_hidden(K)
                output, _, _ = self._pinn(input_features, features_norm, h)
            else:
                output, _, _ = self._pinn(input_features, features_norm)

        vx_pred = output[:, 0]
        vy_pred = output[:, 1]
        yaw_rate_pred = output[:, 2]

        omega_pred = yaw_rate_pred
        beta_pred = torch.atan2(vy_pred, vx_pred + 1e-8)

        return omega_pred, beta_pred, input_features

    # -- Scalar predict (DynamicsModel contract) ----------------------------

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        raise NotImplementedError(
            "PINNModel.predict_next_state is not supported for scalar calls. "
            "Use batched_dynamics via MPPI instead."
        )

    # -- Batched torch dynamics (for MPPI) ----------------------------------

    def setup_torch(self, dt: float, device: str, dtype: torch.dtype) -> None:
        """Configure for batched torch dynamics."""
        self._dt_torch = dt
        self._device = device
        self._dtype = dtype

    def batched_dynamics(
        self, state: torch.Tensor, action: torch.Tensor, t: int,
    ) -> torch.Tensor:
        """Batched PINN dynamics. State: (K,7), Action: (K,2) → (K,7)."""
        p = self._p
        dt = self._dt_torch
        K = state.shape[0]

        x = state[:, 0]
        y = state[:, 1]
        delta = state[:, 2]
        v = state[:, 3]
        psi = state[:, 4]
        omega = state[:, 5]
        beta = state[:, 6]

        accl = action[:, 0]
        delta_dot = action[:, 1]

        # Apply constraints
        u_delta_dot = self._steering_constraint_batch(delta, delta_dot)
        u_accl = self._accl_constraints_batch(v, accl)

        # Initialize per-sample feature histories at t=0
        if t == 0:
            if self._feature_history is None:
                self._feature_history = torch.zeros(
                    self._pinn_horizon, self._num_features,
                    device=self._pinn_device, dtype=torch.float32,
                )
            self._rollout_feature_histories = (
                self._feature_history.unsqueeze(0).expand(K, -1, -1).clone()
            )

        n_substeps = self._n_substeps
        base_dt = self._base_timestep

        if n_substeps == 1:
            # Single-step prediction
            omega_pred, beta_pred, updated_histories = self._predict_batch(
                state, action, self._rollout_feature_histories,
            )
            self._rollout_feature_histories = updated_histories

            next_x = x + dt * v * torch.cos(beta_pred + psi)
            next_y = y + dt * v * torch.sin(beta_pred + psi)
            next_delta = delta + dt * u_delta_dot
            next_v = v + dt * u_accl
            next_psi = psi + dt * omega_pred
            next_omega = omega_pred
            next_beta = beta_pred
        else:
            # Chaining: n sub-steps of base_dt
            constrained_action = torch.stack([u_accl, u_delta_dot], dim=1)
            cur_x, cur_y, cur_delta, cur_v, cur_psi = x, y, delta, v, psi
            cur_omega, cur_beta = omega, beta

            for _ in range(n_substeps):
                current_state = torch.stack([
                    cur_x, cur_y, cur_delta, cur_v, cur_psi, cur_omega, cur_beta,
                ], dim=1)

                omega_pred, beta_pred, updated_histories = self._predict_batch(
                    current_state, constrained_action, self._rollout_feature_histories,
                )
                self._rollout_feature_histories = updated_histories

                cur_x = cur_x + base_dt * cur_v * torch.cos(beta_pred + cur_psi)
                cur_y = cur_y + base_dt * cur_v * torch.sin(beta_pred + cur_psi)
                cur_delta = cur_delta + base_dt * u_delta_dot
                cur_v = cur_v + base_dt * u_accl
                cur_psi = cur_psi + base_dt * omega_pred
                cur_omega = omega_pred
                cur_beta = beta_pred

            next_x, next_y, next_delta, next_v, next_psi = (
                cur_x, cur_y, cur_delta, cur_v, cur_psi,
            )
            next_omega, next_beta = cur_omega, cur_beta

        # Normalize heading
        next_psi = torch.where(next_psi > 2 * np.pi, next_psi - 2 * np.pi, next_psi)
        next_psi = torch.where(next_psi < 0, next_psi + 2 * np.pi, next_psi)

        return torch.stack([
            next_x, next_y, next_delta, next_v, next_psi, next_omega, next_beta,
        ], dim=1)

    # -- Batched constraint helpers (same as F1TenthDynamicsModel) ----------

    def _accl_constraints_batch(
        self, vel: torch.Tensor, accl: torch.Tensor,
    ) -> torch.Tensor:
        p = self._p
        pos_limit = torch.where(
            vel > p.v_switch,
            p.a_max * p.v_switch / (vel + 1e-8),
            torch.full_like(vel, p.a_max),
        )
        result = accl.clone()
        at_min = (vel <= p.v_min) & (accl <= 0)
        at_max = (vel >= p.v_max) & (accl >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(accl), result)
        result = torch.maximum(result, torch.full_like(result, -p.a_max))
        result = torch.minimum(result, pos_limit)
        return result

    def _steering_constraint_batch(
        self, steering_angle: torch.Tensor, steering_velocity: torch.Tensor,
    ) -> torch.Tensor:
        p = self._p
        result = steering_velocity.clone()
        at_min = (steering_angle <= p.s_min) & (steering_velocity <= 0)
        at_max = (steering_angle >= p.s_max) & (steering_velocity >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(steering_velocity), result)
        result = torch.clamp(result, p.sv_min, p.sv_max)
        return result
