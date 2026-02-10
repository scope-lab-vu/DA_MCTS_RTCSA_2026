"""
Kinematic bicycle model — simple, runnable placeholder.

State: [x, y, psi, v, beta, omega]
Inputs: [acceleration, steering_vel]

Uses the standard kinematic bicycle equations with no tire slip.
"""

import math

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class KinematicBicycleModel:
    """Single-track kinematic bicycle (no lateral slip)."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._lf = vehicle_params.lf
        self._lr = vehicle_params.lr
        self._wheelbase = vehicle_params.lf + vehicle_params.lr
        self._max_steer = vehicle_params.s_max
        self._max_speed = vehicle_params.v_max
        self._max_accel = vehicle_params.a_max

        # Internal steering angle state (integrated from steering_vel).
        self._steer_angle: float = 0.0

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        # Integrate steering angle from steering velocity.
        self._steer_angle += action.steering_vel * dt
        self._steer_angle = max(-self._max_steer, min(self._max_steer, self._steer_angle))

        # Clamp acceleration.
        accel = max(-self._max_accel, min(self._max_accel, action.acceleration))

        # Current state unpacking.
        x, y, psi, v = state.x, state.y, state.psi, state.v
        delta = self._steer_angle

        # Slip angle.
        beta = math.atan2(self._lr * math.tan(delta), self._wheelbase)

        # Kinematic equations.
        x_next = x + v * math.cos(psi + beta) * dt
        y_next = y + v * math.sin(psi + beta) * dt
        psi_next = psi + (v / self._wheelbase) * math.sin(beta) * dt  # simplified
        v_next = v + accel * dt
        v_next = max(0.0, min(self._max_speed, v_next))
        omega_next = (v_next / self._wheelbase) * math.sin(beta)

        return VehicleStateMsg(
            x=x_next,
            y=y_next,
            psi=psi_next,
            v=v_next,
            beta=beta,
            omega=omega_next,
        )
