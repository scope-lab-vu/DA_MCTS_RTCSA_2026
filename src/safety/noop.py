"""
No-op safety monitor — passes commands through unchanged.

Use as the default when no safety filtering is needed.
"""

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams


class NoOpSafetyMonitor:
    """Pass-through: returns the proposed command unmodified."""

    def __init__(self, vehicle_params: VehicleParams) -> None:
        self._vehicle_params = vehicle_params

    def filter_action(
        self,
        state: VehicleStateMsg,
        proposed_cmd: ControlCmdMsg,
    ) -> ControlCmdMsg:
        return proposed_cmd
