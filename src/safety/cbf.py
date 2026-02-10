"""
CBF (Control Barrier Function) safety monitor — stub.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class CBFSafetyMonitor:
    """TODO: Implement CBF-based safety filtering."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._vehicle_params = vehicle_params

    def filter_action(
        self,
        state: VehicleStateMsg,
        proposed_cmd: ControlCmdMsg,
    ) -> ControlCmdMsg:
        raise NotImplementedError(
            "CBFSafetyMonitor.filter_action is not yet implemented. "
            "Set safety.type to 'noop' to use the pass-through placeholder."
        )
