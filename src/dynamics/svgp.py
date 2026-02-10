"""
SVGP (Sparse Variational Gaussian Process) dynamics model — stub.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class SVGPModel:
    """TODO: Implement SVGP-based learned dynamics."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._vehicle_params = vehicle_params

    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        raise NotImplementedError(
            "SVGPModel.predict_next_state is not yet implemented. "
            "Set dynamics.type to 'kinematic_bicycle' to use the placeholder."
        )
