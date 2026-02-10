"""
PINN (Physics-Informed Neural Network) dynamics model — stub.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class PINNModel:
    """TODO: Implement PINN-based learned dynamics."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._vehicle_params = vehicle_params

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        raise NotImplementedError(
            "PINNModel.predict_next_state is not yet implemented. "
            "Set dynamics.type to 'kinematic_bicycle' to use the placeholder."
        )
