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

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class F1TenthDynamicsModel:
    """Known single-track dynamics identical to f1tenth_gym."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._p = vehicle_params
        self._g = 9.81

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
