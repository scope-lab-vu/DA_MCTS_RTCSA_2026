"""
Safety Monitor — filters or overrides control commands for safety.

ABC:  SafetyMonitor
Factory: create_safety_monitor(type_str, vehicle_params)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams

from src.safety.noop import NoOpSafetyMonitor
from src.safety.cbf import CBFSafetyMonitor


class SafetyMonitor(ABC):
    """Contract for all safety monitors."""

    @abstractmethod
    def filter_action(
        self,
        state: VehicleStateMsg,
        proposed_cmd: ControlCmdMsg,
    ) -> ControlCmdMsg:
        """Return a (possibly modified) safe command."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAFETY_REGISTRY: Dict[str, Type[SafetyMonitor]] = {
    "noop": NoOpSafetyMonitor,
    "cbf": CBFSafetyMonitor,
}


def create_safety_monitor(
    type_str: str,
    vehicle_params: VehicleParams,
) -> SafetyMonitor:
    cls = SAFETY_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown safety monitor '{type_str}'. "
            f"Available: {list(SAFETY_REGISTRY.keys())}"
        )
    return cls(vehicle_params=vehicle_params)
