"""
Gym-based vehicle interface.

Uses an internal dynamics model to simulate the vehicle.
In the future, replace with a real f1tenth_gym environment call.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class GymVehicleInterface:
    """Simulated vehicle interface using a dynamics model for state rollout."""

    def __init__(
        self,
        vehicle_params: VehicleParams,
        dynamics_model: "DynamicsModel",
        dt: float,
    ) -> None:
        self._vehicle_params = vehicle_params
        self._dynamics_model = dynamics_model
        self._dt = dt
        self._state = VehicleStateMsg()  # start at origin
        self._last_action = ControlCmdMsg()

    # -- VehicleInterface contract ------------------------------------------

    def get_state(self) -> VehicleStateMsg:
        return self._state

    def get_last_applied_action(self) -> ControlCmdMsg:
        return self._last_action

    def apply_action(self, cmd: ControlCmdMsg) -> None:
        self._last_action = cmd
        # Roll forward the state using the injected dynamics model.
        self._state = self._dynamics_model.predict_next_state(
            state=self._state,
            action=cmd,
            dt=self._dt,
        )
