"""
Vehicle Interface — abstracts the data source (sim, hardware, replay).

ABC:  VehicleInterface
Factory: create_vehicle_interface(type_str, vehicle_params, simulation_config)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import MapConfig, SimulationConfig, VehicleParams

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

    def setup_rendering(self, waypoints: WaypointsMsg) -> None:
        """Register render callback for drawing raceline, etc. No-op by default."""
        pass

    def set_planned_trajectory(self, xy_points) -> None:
        """Set planned trajectory for visualization. No-op by default."""
        pass

    def render(self, mode: str = "human") -> None:
        """Render the environment. No-op by default."""
        pass


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
    simulation_config: SimulationConfig,
    map_config: MapConfig,
) -> VehicleInterface:
    cls = VEHICLE_INTERFACE_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown vehicle interface '{type_str}'. "
            f"Available: {list(VEHICLE_INTERFACE_REGISTRY.keys())}"
        )
    return cls(
        vehicle_params=vehicle_params,
        simulation_config=simulation_config,
        map_config=map_config,
    )
