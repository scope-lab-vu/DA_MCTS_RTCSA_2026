"""
SVGP (Sparse Variational Gaussian Process) dynamics model.

Uses an Extended Kinematic (Ekin) base model with two SVGPs providing
residual error corrections for omega (yaw rate) and beta (slip angle).

    st+1 = Ekin(st, ut) + SVGP(st, ut)

Each SVGP takes 6 inputs: [delta, v, omega, beta, accel, steer_vel]
and predicts a scalar correction. Multi-task learning via task-specific
hidden parameters selects the friction condition.

State:  [x, y, delta, v, psi, omega, beta]  (7-dim)
Action: [acceleration, steering_velocity]    (2-dim)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import DynamicsConfig, SimulationConfig, VehicleParams

log = logging.getLogger(__name__)

# Task index → friction mapping (for reference / logging only)
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}


# ---------------------------------------------------------------------------
# Helper: RBF ARD kernel
# ---------------------------------------------------------------------------

def _rbf_ard_kernel(
    X: torch.Tensor, Z: torch.Tensor, theta: torch.Tensor,
) -> torch.Tensor:
    """Batched RBF ARD kernel.

    Args:
        X: (K, D) test points
        Z: (M, D) inducing points
        theta: (D+1,) — [length_scale_1, ..., length_scale_D, sigma]

    Returns:
        K_xz: (K, M) kernel matrix
    """
    length_scales = theta[:-1]  # (D,)
    sigma = theta[-1]

    X_s = X / length_scales.unsqueeze(0)  # (K, D)
    Z_s = Z / length_scales.unsqueeze(0)  # (M, D)

    X_sq = (X_s ** 2).sum(dim=1, keepdim=True)  # (K, 1)
    Z_sq = (Z_s ** 2).sum(dim=1, keepdim=True)  # (M, 1)
    dist = X_sq + Z_sq.T - 2.0 * X_s @ Z_s.T    # (K, M)

    return (sigma ** 2) * torch.exp(-0.5 * dist)


# ---------------------------------------------------------------------------
# Helper: load SVGP model from .npz
# ---------------------------------------------------------------------------

def _load_model_and_scalers_npz(path: str):
    """Load SVGP model, scalers, and metadata from a .npz file."""
    data = np.load(path, allow_pickle=False)

    meta = json.loads(str(data["meta_json"]))
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
    return model, xscaler, yscaler, meta


class SVGPModel:
    """SVGP-based dynamics model for MPPI (Ekin + SVGP residuals)."""

    def __init__(
        self,
        vehicle_params: VehicleParams,
        dynamics_config: Optional[DynamicsConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
    ) -> None:
        self._p = vehicle_params
        self._g = 9.81

        # Torch batched dynamics state (set via setup_torch).
        self._dt_torch: Optional[float] = None
        self._device: Optional[str] = None
        self._dtype: Optional[torch.dtype] = None

        if dynamics_config is None:
            raise ValueError("SVGPModel requires dynamics_config with model paths.")

        # ------------------------------------------------------------------
        # Resolve paths
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]
        omega_path = str(project_root / dynamics_config.omega_model_path)
        beta_path = str(project_root / dynamics_config.beta_model_path)
        device_str = dynamics_config.inference_device
        task_index = dynamics_config.task_index

        # ------------------------------------------------------------------
        # Timestep / chaining
        # ------------------------------------------------------------------
        self._timestep = simulation_config.dt_s if simulation_config else 0.02
        self._base_timestep = 0.02
        self._n_substeps = max(1, round(self._timestep / self._base_timestep))
        self._use_chaining = self._n_substeps > 1

        if self._use_chaining:
            log.info(
                "SVGP chaining: dt=%.3f → %d sub-steps of %.3f",
                self._timestep, self._n_substeps, self._base_timestep,
            )

        # ------------------------------------------------------------------
        # SVGP internal precision: float64 for numerical stability
        # ------------------------------------------------------------------
        self._svgp_device = torch.device(device_str)
        self._svgp_dtype = torch.float64
        self._output_dtype = torch.float32

        # ------------------------------------------------------------------
        # Load omega SVGP
        # ------------------------------------------------------------------
        log.info("Loading omega SVGP from: %s", omega_path)
        omega_dict, omega_xs, omega_ys, _ = _load_model_and_scalers_npz(omega_path)

        self._omega_theta = torch.tensor(
            np.array(omega_dict['theta_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._omega_mu_m = torch.tensor(
            np.array(omega_dict['mu_m_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._omega_K_mm_inv = torch.tensor(
            np.array(omega_dict['K_mm_inv']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._omega_Zx = torch.tensor(
            np.array(omega_dict['Zx_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._omega_Zh = torch.tensor(
            np.array(omega_dict['Zh_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._omega_Z_aug = torch.cat([self._omega_Zx, self._omega_Zh], dim=1)
        self._omega_alpha = self._omega_K_mm_inv @ self._omega_mu_m

        self._omega_xmean = torch.tensor(omega_xs.mean_, device=self._svgp_device, dtype=self._svgp_dtype)
        self._omega_xscale = torch.tensor(omega_xs.scale_, device=self._svgp_device, dtype=self._svgp_dtype)
        self._omega_ymean = torch.tensor(np.array(omega_ys.mean_), device=self._svgp_device, dtype=self._svgp_dtype).squeeze()
        self._omega_yscale = torch.tensor(np.array(omega_ys.scale_), device=self._svgp_device, dtype=self._svgp_dtype).squeeze()

        # Store raw h_mu_opt for task switching
        self._omega_h_mu_all = omega_dict['h_mu_opt']

        # ------------------------------------------------------------------
        # Load beta SVGP
        # ------------------------------------------------------------------
        log.info("Loading beta SVGP from: %s", beta_path)
        beta_dict, beta_xs, beta_ys, _ = _load_model_and_scalers_npz(beta_path)

        self._beta_theta = torch.tensor(
            np.array(beta_dict['theta_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_mu_m = torch.tensor(
            np.array(beta_dict['mu_m_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_K_mm_inv = torch.tensor(
            np.array(beta_dict['K_mm_inv']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_Zx = torch.tensor(
            np.array(beta_dict['Zx_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_Zh = torch.tensor(
            np.array(beta_dict['Zh_opt']), device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_Z_aug = torch.cat([self._beta_Zx, self._beta_Zh], dim=1)
        self._beta_alpha = self._beta_K_mm_inv @ self._beta_mu_m

        self._beta_xmean = torch.tensor(beta_xs.mean_, device=self._svgp_device, dtype=self._svgp_dtype)
        self._beta_xscale = torch.tensor(beta_xs.scale_, device=self._svgp_device, dtype=self._svgp_dtype)
        self._beta_ymean = torch.tensor(np.array(beta_ys.mean_), device=self._svgp_device, dtype=self._svgp_dtype).squeeze()
        self._beta_yscale = torch.tensor(np.array(beta_ys.scale_), device=self._svgp_device, dtype=self._svgp_dtype).squeeze()

        self._beta_h_mu_all = beta_dict['h_mu_opt']

        # ------------------------------------------------------------------
        # Set task-specific hidden parameters
        # ------------------------------------------------------------------
        self.set_task_index(task_index)

        M_omega = self._omega_Zx.shape[0]
        M_beta = self._beta_Zx.shape[0]
        D = self._omega_Zx.shape[1]
        log.info("SVGP loaded: D=%d, M_omega=%d, M_beta=%d, task=%d", D, M_omega, M_beta, task_index)

    # -- Task switching -----------------------------------------------------

    def set_task_index(self, task_index: int) -> None:
        """Set the task-specific hidden parameters for both GPs."""
        self._task_index = task_index
        self._omega_hp_mu = torch.tensor(
            np.array(self._omega_h_mu_all[task_index]),
            device=self._svgp_device, dtype=self._svgp_dtype,
        )
        self._beta_hp_mu = torch.tensor(
            np.array(self._beta_h_mu_all[task_index]),
            device=self._svgp_device, dtype=self._svgp_dtype,
        )
        log.info("SVGP task_index=%d (mu≈%.4f)", task_index, TASK_FRICTION_MAP.get(task_index, -1))

    # -- Core SVGP prediction -----------------------------------------------

    def _svgp_predict(
        self,
        gp_inputs: torch.Tensor,
        xmean: torch.Tensor,
        xscale: torch.Tensor,
        ymean: torch.Tensor,
        yscale: torch.Tensor,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        Z_aug: torch.Tensor,
        hp_mu: torch.Tensor,
    ) -> torch.Tensor:
        """Core SVGP prediction: normalize → augment → kernel → predict → unnormalize.

        All computation in float64; output cast to output_dtype.
        """
        K = gp_inputs.shape[0]
        gp_inputs = gp_inputs.to(self._svgp_dtype)

        gp_norm = (gp_inputs - xmean.unsqueeze(0)) / xscale.unsqueeze(0)
        hp_tiled = hp_mu.unsqueeze(0).expand(K, -1)
        X_aug = torch.cat([gp_norm, hp_tiled], dim=1)

        K_sm = _rbf_ard_kernel(X_aug, Z_aug, theta)
        f_mean = K_sm @ alpha
        y_pred = f_mean.squeeze(1) * yscale + ymean

        return y_pred.to(self._output_dtype)

    def _predict_omega_batch(self, gp_inputs: torch.Tensor) -> torch.Tensor:
        return self._svgp_predict(
            gp_inputs,
            self._omega_xmean, self._omega_xscale,
            self._omega_ymean, self._omega_yscale,
            self._omega_theta, self._omega_alpha, self._omega_Z_aug,
            self._omega_hp_mu,
        )

    def _predict_beta_batch(self, gp_inputs: torch.Tensor) -> torch.Tensor:
        return self._svgp_predict(
            gp_inputs,
            self._beta_xmean, self._beta_xscale,
            self._beta_ymean, self._beta_yscale,
            self._beta_theta, self._beta_alpha, self._beta_Z_aug,
            self._beta_hp_mu,
        )

    # -- Single-step Ekin + SVGP --------------------------------------------

    def _single_step_dynamics(
        self,
        state: torch.Tensor,
        u_accl: torch.Tensor,
        u_delta_dot: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single Euler step: Ekin base + SVGP residual corrections.

        Args:
            state: (K, 7) tensor
            u_accl: (K,) constrained acceleration
            u_delta_dot: (K,) constrained steering rate
            dt: sub-step timestep
        """
        p = self._p
        L = p.lf + p.lr

        x = state[:, 0]
        y = state[:, 1]
        delta = state[:, 2]
        v = state[:, 3]
        psi = state[:, 4]
        omega = state[:, 5]
        beta = state[:, 6]

        # Extended Kinematic model derivatives
        x_dot = v * torch.cos(beta + psi)
        y_dot = v * torch.sin(beta + psi)
        v_dot = u_accl
        psi_dot = omega
        omega_dot = ((u_delta_dot * v) + (delta * u_accl)) / L
        beta_dot = (p.lf * ((u_delta_dot * v) + (delta * u_accl))) / L

        # Euler integration (Ekin baseline)
        next_x = x + dt * x_dot
        next_y = y + dt * y_dot
        next_delta = delta + dt * u_delta_dot
        next_v = v + dt * v_dot
        next_psi = psi + dt * psi_dot
        next_omega = omega + dt * omega_dot
        next_beta = beta + dt * beta_dot

        # SVGP residual corrections
        gp_inputs = torch.stack([delta, v, omega, beta, u_accl, u_delta_dot], dim=1)

        with torch.no_grad():
            err_omega = self._predict_omega_batch(gp_inputs)
            err_beta = self._predict_beta_batch(gp_inputs)

        next_omega = next_omega + err_omega
        next_beta = next_beta + err_beta

        return torch.stack([
            next_x, next_y, next_delta, next_v, next_psi, next_omega, next_beta,
        ], dim=1)

    # -- Scalar predict (DynamicsModel contract) ----------------------------

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        raise NotImplementedError(
            "SVGPModel.predict_next_state is not supported for scalar calls. "
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
        """Batched SVGP dynamics: Ekin + SVGP residuals.

        State: (K,7), Action: (K,2) → (K,7).
        """
        accl = action[:, 0]
        delta_dot = action[:, 1]

        # Apply constraints
        u_delta_dot = self._steering_constraint_batch(state[:, 2], delta_dot)
        u_accl = self._accl_constraints_batch(state[:, 3], accl)

        n_substeps = self._n_substeps
        base_dt = self._base_timestep

        if n_substeps == 1:
            next_state = self._single_step_dynamics(state, u_accl, u_delta_dot, self._dt_torch)
        else:
            cur_state = state
            for _ in range(n_substeps):
                cur_state = self._single_step_dynamics(cur_state, u_accl, u_delta_dot, base_dt)
            next_state = cur_state

        # Normalize heading
        next_psi = next_state[:, 4]
        next_psi = torch.where(next_psi > 2 * np.pi, next_psi - 2 * np.pi, next_psi)
        next_psi = torch.where(next_psi < 0, next_psi + 2 * np.pi, next_psi)
        next_state = next_state.clone()
        next_state[:, 4] = next_psi

        return next_state

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
