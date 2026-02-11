"""
F1TENTH Single-Track dynamics — known dynamics model.

Same equations used inside the f1tenth_gym simulator.
State: [x, y, delta, v, psi, omega, beta]
Inputs: [acceleration, steering_velocity]

Uses kinematic model for |v| < 0.5 m/s, dynamic model otherwise.
This model is used by MPPI as a "cheating" dynamics model for testing,
since it matches the simulator exactly.
"""

import math

import numpy as np
import torch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class F1TenthDynamicsModel:
    """Known single-track dynamics identical to f1tenth_gym."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._p = vehicle_params
        self._g = 9.81

        # Torch batched dynamics state (set via setup_torch).
        self._dt_torch: float = None
        self._device: str = None
        self._dtype: torch.dtype = None

    # -- Scalar predict (DynamicsModel contract) ----------------------------

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        p = self._p
        g = self._g

        x, y, delta, v, psi, omega, beta = (
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        )
        accl = action.acceleration
        sv = action.steering_vel

        # Steering constraint
        if (delta <= p.s_min and sv <= 0) or (delta >= p.s_max and sv >= 0):
            sv = 0.0
        sv = max(p.sv_min, min(p.sv_max, sv))

        # Acceleration constraint
        if v > p.v_switch:
            pos_limit = p.a_max * p.v_switch / (v + 1e-8)
        else:
            pos_limit = p.a_max
        if (v <= p.v_min and accl <= 0) or (v >= p.v_max and accl >= 0):
            accl = 0.0
        accl = max(-p.a_max, min(pos_limit, accl))

        L = p.lf + p.lr

        if abs(v) < 0.5:
            # Kinematic model
            f_x = v * math.cos(psi)
            f_y = v * math.sin(psi)
            f_delta = sv
            f_v = accl
            f_psi = v / L * math.tan(delta)
            f_omega = accl / L * math.tan(delta) + v / (L * math.cos(delta) ** 2) * sv
            f_beta = 0.0
        else:
            # Dynamic model
            f_x = v * math.cos(beta + psi)
            f_y = v * math.sin(beta + psi)
            f_delta = sv
            f_v = accl
            f_psi = omega

            v_safe = max(abs(v), 0.5)
            f_omega = (
                -p.mu * p.m / (v_safe * p.I * L) * (
                    p.lf ** 2 * p.C_Sf * (g * p.lr - accl * p.h)
                    + p.lr ** 2 * p.C_Sr * (g * p.lf + accl * p.h)
                ) * omega
                + p.mu * p.m / (p.I * L) * (
                    p.lr * p.C_Sr * (g * p.lf + accl * p.h)
                    - p.lf * p.C_Sf * (g * p.lr - accl * p.h)
                ) * beta
                + p.mu * p.m / (p.I * L)
                * p.lf * p.C_Sf * (g * p.lr - accl * p.h) * delta
            )
            f_beta = (
                (p.mu / (v_safe ** 2 * L) * (
                    p.C_Sr * (g * p.lf + accl * p.h) * p.lr
                    - p.C_Sf * (g * p.lr - accl * p.h) * p.lf
                ) - 1) * omega
                - p.mu / (v_safe * L) * (
                    p.C_Sr * (g * p.lf + accl * p.h)
                    + p.C_Sf * (g * p.lr - accl * p.h)
                ) * beta
                + p.mu / (v_safe * L)
                * p.C_Sf * (g * p.lr - accl * p.h) * delta
            )

        # Euler integration
        x_n = x + dt * f_x
        y_n = y + dt * f_y
        delta_n = delta + dt * f_delta
        v_n = v + dt * f_v
        psi_n = psi + dt * f_psi
        omega_n = omega + dt * f_omega
        beta_n = beta + dt * f_beta

        # Normalize heading
        psi_n = psi_n % (2 * math.pi)

        return VehicleStateMsg(
            x=x_n, y=y_n, delta=delta_n, v=v_n,
            psi=psi_n, omega=omega_n, beta=beta_n,
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
        """Batched single-track dynamics.  State: (K,7), Action: (K,2)."""
        p = self._p
        dt = self._dt_torch

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

        # Mask for kinematic vs dynamic model
        use_ks = torch.abs(v) < 0.5
        L = p.lf + p.lr
        g = self._g

        # Kinematic model derivatives
        f_ks_x = v * torch.cos(psi)
        f_ks_y = v * torch.sin(psi)
        f_ks_delta = u_delta_dot
        f_ks_v = u_accl
        f_ks_psi = v / L * torch.tan(delta)
        f_ks_omega = u_accl / L * torch.tan(delta) + v / (L * torch.cos(delta) ** 2) * u_delta_dot
        f_ks_beta = torch.zeros_like(beta)

        # Dynamic model derivatives
        f_dyn_x = v * torch.cos(beta + psi)
        f_dyn_y = v * torch.sin(beta + psi)
        f_dyn_delta = u_delta_dot
        f_dyn_v = u_accl
        f_dyn_psi = omega

        v_safe = torch.clamp(torch.abs(v), min=0.5)
        f_dyn_omega = (
            -p.mu * p.m / (v_safe * p.I * L) * (
                p.lf ** 2 * p.C_Sf * (g * p.lr - u_accl * p.h)
                + p.lr ** 2 * p.C_Sr * (g * p.lf + u_accl * p.h)
            ) * omega
            + p.mu * p.m / (p.I * L) * (
                p.lr * p.C_Sr * (g * p.lf + u_accl * p.h)
                - p.lf * p.C_Sf * (g * p.lr - u_accl * p.h)
            ) * beta
            + p.mu * p.m / (p.I * L)
            * p.lf * p.C_Sf * (g * p.lr - u_accl * p.h) * delta
        )
        f_dyn_beta = (
            (p.mu / (v_safe ** 2 * L) * (
                p.C_Sr * (g * p.lf + u_accl * p.h) * p.lr
                - p.C_Sf * (g * p.lr - u_accl * p.h) * p.lf
            ) - 1) * omega
            - p.mu / (v_safe * L) * (
                p.C_Sr * (g * p.lf + u_accl * p.h)
                + p.C_Sf * (g * p.lr - u_accl * p.h)
            ) * beta
            + p.mu / (v_safe * L)
            * p.C_Sf * (g * p.lr - u_accl * p.h) * delta
        )

        # Select derivatives based on velocity
        f_x = torch.where(use_ks, f_ks_x, f_dyn_x)
        f_y = torch.where(use_ks, f_ks_y, f_dyn_y)
        f_delta = torch.where(use_ks, f_ks_delta, f_dyn_delta)
        f_v = torch.where(use_ks, f_ks_v, f_dyn_v)
        f_psi = torch.where(use_ks, f_ks_psi, f_dyn_psi)
        f_omega = torch.where(use_ks, f_ks_omega, f_dyn_omega)
        f_beta = torch.where(use_ks, f_ks_beta, f_dyn_beta)

        f = torch.stack([f_x, f_y, f_delta, f_v, f_psi, f_omega, f_beta], dim=1)

        # Euler integration
        next_state = state + dt * f

        # Normalize heading
        psi_new = next_state[:, 4]
        psi_new = torch.where(psi_new > 2 * np.pi, psi_new - 2 * np.pi, psi_new)
        psi_new = torch.where(psi_new < 0, psi_new + 2 * np.pi, psi_new)
        next_state = torch.cat([next_state[:, :4], psi_new.unsqueeze(1), next_state[:, 5:]], dim=1)

        return next_state

    # -- Batched constraint helpers -----------------------------------------

    def _accl_constraints_batch(self, vel: torch.Tensor, accl: torch.Tensor) -> torch.Tensor:
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

    def _steering_constraint_batch(self, steering_angle: torch.Tensor, steering_velocity: torch.Tensor) -> torch.Tensor:
        p = self._p
        result = steering_velocity.clone()
        at_min = (steering_angle <= p.s_min) & (steering_velocity <= 0)
        at_max = (steering_angle >= p.s_max) & (steering_velocity >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(steering_velocity), result)
        result = torch.clamp(result, p.sv_min, p.sv_max)
        return result
