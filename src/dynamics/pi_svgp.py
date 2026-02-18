"""
PI-SVGP (Physics-Informed SVGP) dynamics model.

Uses a multi-output Sparse Variational GP to predict 9 Pacejka tire
parameters, then applies Pacejka tire physics to compute next-state
predictions [VX, VY, YAW_RATE], from which we derive:
    omega = yaw_rate   (directly)
    beta  = atan2(vy, vx)

Kinematic equations handle [x, y, delta, v, psi].

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

# Task index -> friction mapping
TASK_FRICTION_MAP = {0: 1.0489, 1: 0.8642, 2: 0.4947}

# Pacejka parameter ordering and positivity constraints
PACEJKA_PARAM_NAMES = ["Bf", "Cf", "Df", "Ef", "Br", "Cr", "Dr", "Er", "Frx"]
# Indices that must be positive (softplus): Bf=0, Cf=1, Df=2, Br=4, Cr=5, Dr=6
POSITIVE_INDICES = [0, 1, 2, 4, 5, 6]


# ---------------------------------------------------------------------------
# Helper: load model from .npz
# ---------------------------------------------------------------------------

def _load_model_and_scalers_npz(path: str):
    """Load pi_svgp model + scalers from .npz file."""
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


# ---------------------------------------------------------------------------
# Helper: RBF ARD kernel (same as svgp.py)
# ---------------------------------------------------------------------------

def _rbf_ard_kernel(
    X: torch.Tensor, Z: torch.Tensor, theta: torch.Tensor,
) -> torch.Tensor:
    """Batched RBF ARD kernel.

    Args:
        X: (K, D) test points
        Z: (M, D) inducing points
        theta: (D+1,) [length_scale_1, ..., length_scale_D, sigma]

    Returns:
        K_xz: (K, M) kernel matrix
    """
    length_scales = theta[:-1]
    sigma = theta[-1]

    X_s = X / length_scales.unsqueeze(0)
    Z_s = Z / length_scales.unsqueeze(0)

    X_sq = (X_s ** 2).sum(dim=1, keepdim=True)
    Z_sq = (Z_s ** 2).sum(dim=1, keepdim=True)
    dist = X_sq + Z_sq.T - 2.0 * X_s @ Z_s.T

    return (sigma ** 2) * torch.exp(-0.5 * dist)


class PiSVGPModel:
    """Physics-Informed SVGP dynamics model for MPPI.

    Multi-output GP predicts 9 Pacejka tire parameters, then Pacejka
    physics computes next [VX, VY, YAW_RATE].
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        dynamics_config: Optional[DynamicsConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
    ) -> None:
        self._p = vehicle_params

        self._dt_torch: Optional[float] = None
        self._device: Optional[str] = None
        self._dtype: Optional[torch.dtype] = None

        if dynamics_config is None:
            raise ValueError("PiSVGPModel requires dynamics_config with model_path.")

        # ------------------------------------------------------------------
        # Resolve model path
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]
        model_path = str(project_root / dynamics_config.model_path)
        device_str = dynamics_config.inference_device
        task_index = dynamics_config.task_index

        # ------------------------------------------------------------------
        # Timestep / chaining
        # ------------------------------------------------------------------
        self._timestep = simulation_config.dt_s if simulation_config else 0.02
        self._base_timestep = 0.02
        self._n_substeps = max(1, round(self._timestep / self._base_timestep))

        if self._n_substeps > 1:
            log.info(
                "PI-SVGP chaining: dt=%.3f -> %d sub-steps of %.3f",
                self._timestep, self._n_substeps, self._base_timestep,
            )

        # ------------------------------------------------------------------
        # Internal precision: float64 for GP, float32 output
        # ------------------------------------------------------------------
        self._gp_device = torch.device(device_str)
        self._gp_dtype = torch.float64
        self._output_dtype = torch.float32

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        log.info("Loading PI-SVGP from: %s", model_path)
        model_dict, xscaler = _load_model_and_scalers_npz(model_path)

        M = int(model_dict['M'])
        Hdim = int(model_dict['Hdim'])
        dx_flat = int(model_dict['dx_flat'])
        num_outputs = int(model_dict['num_outputs'])

        # Kernel hyperparameters
        self._theta = torch.tensor(
            np.array(model_dict['theta_opt']),
            device=self._gp_device, dtype=self._gp_dtype,
        )

        # Inducing points (augmented)
        Zx = torch.tensor(
            np.array(model_dict['Zx_opt']),
            device=self._gp_device, dtype=self._gp_dtype,
        )
        Zh = torch.tensor(
            np.array(model_dict['Zh_opt']),
            device=self._gp_device, dtype=self._gp_dtype,
        )
        self._Z_aug = torch.cat([Zx, Zh], dim=1)  # (M, dx_flat + Hdim)

        # Variational inducing means
        m_opt = torch.tensor(
            np.array(model_dict['m_opt']),
            device=self._gp_device, dtype=self._gp_dtype,
        )  # (M, 9)

        # Output bias (pre-constraint space)
        self._output_bias = torch.tensor(
            np.array(model_dict['output_bias_opt']),
            device=self._gp_device, dtype=self._gp_dtype,
        )  # (9,)

        # Precompute K_mm_inv and alpha = K_mm_inv @ m for efficient prediction
        K_mm = _rbf_ard_kernel(self._Z_aug, self._Z_aug, self._theta)
        K_mm = K_mm + torch.eye(M, device=self._gp_device, dtype=self._gp_dtype) * 1e-6
        K_mm_inv = torch.linalg.inv(K_mm)
        self._alpha = K_mm_inv @ m_opt  # (M, 9)

        # Input scaler
        self._xmean = torch.tensor(
            xscaler.mean_, device=self._gp_device, dtype=self._gp_dtype,
        )
        self._xscale = torch.tensor(
            xscaler.scale_, device=self._gp_device, dtype=self._gp_dtype,
        )

        # Task-specific latent parameters
        self._h_mu_all = model_dict['h_mu_opt']  # (T, Hdim) numpy
        self.set_task_index(task_index)

        # Build linear interpolators: C_alpha -> latent h
        self._build_calpha_interpolators()

        # Positive-constraint mask for Pacejka parameters
        self._positive_mask = torch.zeros(
            num_outputs, device=self._gp_device, dtype=torch.bool,
        )
        for idx in POSITIVE_INDICES:
            self._positive_mask[idx] = True

        log.info(
            "PI-SVGP loaded: M=%d dx=%d Hdim=%d outputs=%d task=%d (mu~%.4f)",
            M, dx_flat, Hdim, num_outputs, task_index,
            TASK_FRICTION_MAP.get(task_index, -1),
        )

    # -- Task switching -----------------------------------------------------

    def set_task_index(self, task_index: int) -> None:
        """Set task-specific latent parameters."""
        self._task_index = task_index
        self._hp_mu = torch.tensor(
            np.array(self._h_mu_all[task_index]),
            device=self._gp_device, dtype=self._gp_dtype,
        )
        log.info(
            "PI-SVGP task=%d (mu~%.4f)",
            task_index, TASK_FRICTION_MAP.get(task_index, -1),
        )

    # -- C_alpha latent interpolation ---------------------------------------

    def _build_calpha_interpolators(self) -> None:
        """Fit per-dimension linear maps from C_alpha -> latent h.

        C_alpha anchors correspond to the three training friction levels:
            task 0: mu=1.0489  ->  C_alpha ≈ 104.89
            task 1: mu=0.8642  ->  C_alpha ≈  86.42
            task 2: mu=0.4947  ->  C_alpha ≈  49.47
        """
        self._c_alpha_anchors = np.array([104.89, 86.42, 49.47])
        h_anchors = np.array(self._h_mu_all[:3])  # (3, Hdim)
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
            "PI-SVGP C_alpha interpolators built: range [%.2f, %.2f] -> Hdim=%d",
            self._c_alpha_min, self._c_alpha_max, Hdim,
        )

    def set_latents_from_calpha(self, c_alpha: float) -> None:
        """Interpolate and set task latent parameters from cornering stiffness.

        Args:
            c_alpha: Estimated rear cornering stiffness [N/rad]. Values are
                     clamped to the training range [49.47, 104.89].
        """
        c_alpha_clamped = np.clip(c_alpha, self._c_alpha_min, self._c_alpha_max)
        h = self._calpha_slopes * c_alpha_clamped + self._calpha_intercepts  # (Hdim,)
        self._hp_mu = torch.tensor(h, device=self._gp_device, dtype=self._gp_dtype)
        log.debug("PI-SVGP latents updated from c_alpha=%.2f (clamped=%.2f)", c_alpha, c_alpha_clamped)

    # -- Feature extraction -------------------------------------------------

    def _extract_features(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Extract GP input features from state and action.

        Args:
            state: (K, 7) [x, y, delta, v, psi, omega, beta]
            action: (K, 2) [accel, steer_vel]

        Returns:
            features: (K, 6) [VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD]
        """
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
        throttle_cmd = accel * self._base_timestep
        steering_cmd = steer_vel * self._base_timestep

        return torch.stack(
            [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd], dim=1,
        )

    # -- GP prediction ------------------------------------------------------

    def _gp_predict(self, features: torch.Tensor) -> torch.Tensor:
        """Multi-output GP prediction -> 9 constrained Pacejka params.

        Args:
            features: (K, 6) raw features (any dtype)

        Returns:
            pacejka_params: (K, 9) constrained Pacejka parameters (float64)
        """
        K = features.shape[0]
        features = features.to(self._gp_dtype)

        # Normalize
        feat_norm = (features - self._xmean.unsqueeze(0)) / self._xscale.unsqueeze(0)

        # Augment with task latent params
        hp_tiled = self._hp_mu.unsqueeze(0).expand(K, -1)
        X_aug = torch.cat([feat_norm, hp_tiled], dim=1)  # (K, dx + Hdim)

        # GP posterior mean
        K_nm = _rbf_ard_kernel(X_aug, self._Z_aug, self._theta)  # (K, M)
        raw_gp = K_nm @ self._alpha  # (K, 9)

        # Add output bias
        raw_with_bias = raw_gp + self._output_bias.unsqueeze(0)

        # Apply Pacejka constraints (softplus on B, C, D params)
        constrained = raw_with_bias.clone()
        constrained[:, self._positive_mask] = torch.nn.functional.softplus(
            raw_with_bias[:, self._positive_mask]
        )

        return constrained

    # -- Pacejka tire physics -----------------------------------------------

    def _pacejka_physics(
        self,
        features: torch.Tensor,
        pacejka_params: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply Pacejka tire physics to compute next [VX, VY, YAW_RATE].

        Ported from model.py pacejka_physics_jax.

        Args:
            features: (K, 6) [VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD]
            pacejka_params: (K, 9) Pacejka parameters
            dt: integration timestep

        Returns:
            next_state: (K, 3) [VX_next, VY_next, YAW_RATE_next]
        """
        features = features.to(self._gp_dtype)

        VX = features[:, 0]
        VY = features[:, 1]
        YAW_RATE = features[:, 2]
        STEERING_FB = features[:, 3]
        STEERING_CMD = features[:, 5]

        Bf = pacejka_params[:, 0]
        Cf = pacejka_params[:, 1]
        Df = pacejka_params[:, 2]
        Ef = pacejka_params[:, 3]
        Br = pacejka_params[:, 4]
        Cr = pacejka_params[:, 5]
        Dr = pacejka_params[:, 6]
        Er = pacejka_params[:, 7]
        Frx = pacejka_params[:, 8]

        lf = self._p.lf
        lr = self._p.lr
        mass = self._p.m
        Iz = self._p.I

        # Steering angle
        steering = STEERING_FB + STEERING_CMD

        # Slip angles
        alphaf = steering - torch.atan2(lf * YAW_RATE + VY, torch.abs(VX))
        alphar = torch.atan2(lr * YAW_RATE - VY, torch.abs(VX))

        # Pacejka tire forces (front and rear lateral)
        Ffy = Df * torch.sin(
            Cf * torch.atan(Bf * alphaf - Ef * (Bf * alphaf - torch.atan(Bf * alphaf)))
        )
        Fry = Dr * torch.sin(
            Cr * torch.atan(Br * alphar - Er * (Br * alphar - torch.atan(Br * alphar)))
        )

        # Vehicle dynamics
        dVX = (1.0 / mass) * (Frx - Ffy * torch.sin(steering)) + VY * YAW_RATE
        dVY = (1.0 / mass) * (Fry + Ffy * torch.cos(steering)) - VX * YAW_RATE
        dYAW_RATE = (1.0 / Iz) * (Ffy * lf * torch.cos(steering) - Fry * lr)

        # Euler integration
        next_VX = VX + dVX * dt
        next_VY = VY + dVY * dt
        next_YAW_RATE = YAW_RATE + dYAW_RATE * dt

        return torch.stack([next_VX, next_VY, next_YAW_RATE], dim=1)

    # -- Single-step dynamics -----------------------------------------------

    def _single_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        u_accl: torch.Tensor,
        u_delta_dot: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single step: GP -> Pacejka physics -> kinematic update.

        Args:
            state: (K, 7)
            action: (K, 2) constrained [accl, steer_vel]
            u_accl: (K,) constrained acceleration
            u_delta_dot: (K,) constrained steering velocity
            dt: substep timestep

        Returns:
            next_state: (K, 7)
        """
        x = state[:, 0]
        y = state[:, 1]
        delta = state[:, 2]
        v = state[:, 3]
        psi = state[:, 4]

        # Extract features
        features = self._extract_features(state, action)

        # GP predict Pacejka params + apply physics
        with torch.no_grad():
            pacejka_params = self._gp_predict(features)

        next_body = self._pacejka_physics(features, pacejka_params, dt)

        # Map body-frame predictions to state
        omega_pred = next_body[:, 2].to(self._output_dtype)
        beta_pred = torch.atan2(
            next_body[:, 1], next_body[:, 0] + 1e-8,
        ).to(self._output_dtype)

        # Kinematic updates (same as PINN)
        next_x = x + dt * v * torch.cos(beta_pred + psi)
        next_y = y + dt * v * torch.sin(beta_pred + psi)
        next_delta = delta + dt * u_delta_dot
        next_v = v + dt * u_accl
        next_psi = psi + dt * omega_pred

        return torch.stack([
            next_x, next_y, next_delta, next_v, next_psi, omega_pred, beta_pred,
        ], dim=1)

    # -- Scalar predict (DynamicsModel contract) ----------------------------

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        raise NotImplementedError(
            "PiSVGPModel.predict_next_state is not supported for scalar calls. "
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
        """Batched PI-SVGP dynamics: GP -> Pacejka -> next state.

        State: (K,7), Action: (K,2) -> (K,7).
        """
        accl = action[:, 0]
        delta_dot = action[:, 1]

        # Apply constraints
        u_delta_dot = self._steering_constraint_batch(state[:, 2], delta_dot)
        u_accl = self._accl_constraints_batch(state[:, 3], accl)
        constrained_action = torch.stack([u_accl, u_delta_dot], dim=1)

        n_substeps = self._n_substeps
        base_dt = self._base_timestep

        if n_substeps == 1:
            next_state = self._single_step(
                state, constrained_action, u_accl, u_delta_dot, self._dt_torch,
            )
        else:
            cur_state = state
            for _ in range(n_substeps):
                cur_state = self._single_step(
                    cur_state, constrained_action, u_accl, u_delta_dot, base_dt,
                )
            next_state = cur_state

        # Normalize heading
        next_psi = next_state[:, 4]
        next_psi = torch.where(next_psi > 2 * np.pi, next_psi - 2 * np.pi, next_psi)
        next_psi = torch.where(next_psi < 0, next_psi + 2 * np.pi, next_psi)
        next_state = next_state.clone()
        next_state[:, 4] = next_psi

        return next_state

    # -- Batched constraint helpers (same as other dynamics models) ----------

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
        result = torch.where(
            at_min | at_max, torch.zeros_like(steering_velocity), result,
        )
        result = torch.clamp(result, p.sv_min, p.sv_max)
        return result
