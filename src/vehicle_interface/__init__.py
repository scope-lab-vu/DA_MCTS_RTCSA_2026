"""
Vehicle Interface — abstracts the data source (sim, hardware, replay).

ABC:  VehicleInterface
Factory: create_vehicle_interface(type_str, vehicle_params, dynamics_model, dt)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams

# Imported here so they register; concrete files also usable standalone.
from src.vehicle_interface.gym_interface import GymVehicleInterface


class VehicleInterface(ABC):
    """Contract for all vehicle data sources."""

    @abstractmethod
    def get_state(self) -> VehicleStateMsg:
        """Return the current vehicle state."""
        ...

    @abstractmethod
    def get_last_applied_action(self) -> ControlCmdMsg:
        """Return the most recently applied control command."""
        ...

    @abstractmethod
    def apply_action(self, cmd: ControlCmdMsg) -> None:
        """Send a control command to the vehicle / simulator."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VEHICLE_INTERFACE_REGISTRY: Dict[str, Type[VehicleInterface]] = {
    "gym": GymVehicleInterface,
    # TODO: "ros2": ROS2VehicleInterface,
    # TODO: "telemetry_replay": TelemetryReplayInterface,
}


def create_vehicle_interface(
    type_str: str,
    vehicle_params: VehicleParams,
    dynamics_model: "DynamicsModel",  # forward ref to avoid circular import
    dt: float,
) -> VehicleInterface:
    cls = VEHICLE_INTERFACE_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown vehicle interface '{type_str}'. "
            f"Available: {list(VEHICLE_INTERFACE_REGISTRY.keys())}"
        )
    return cls(vehicle_params=vehicle_params, dynamics_model=dynamics_model, dt=dt)
