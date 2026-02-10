"""
MPC (Model Predictive Control) controller — stub.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, VehicleParams


class MPCController:
    """TODO: Implement MPC controller."""

    def __init__(
        self,
        dynamics_model: "DynamicsModel",
        vehicle_params: VehicleParams,
        config: ControllerConfig,
    ) -> None:
        self._dynamics_model = dynamics_model
        self._vehicle_params = vehicle_params
        self._config = config

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        raise NotImplementedError(
            "MPCController.compute_action is not yet implemented. "
            "Set controller.type to 'simple' to use the placeholder."
        )
