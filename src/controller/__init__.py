"""
Controller — computes a control command given vehicle state + waypoints.

ABC:  Controller
Factory: create_controller(type_str, dynamics_model, vehicle_params, controller_config)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, VehicleParams

from src.controller.simple_controller import SimpleController
from src.controller.mpc import MPCController
from src.controller.mppi import MPPIController


class Controller(ABC):
    """Contract for all controllers."""

    @abstractmethod
    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONTROLLER_REGISTRY: Dict[str, Type[Controller]] = {
    "simple": SimpleController,
    "mpc": MPCController,
    "mppi": MPPIController,
}


def create_controller(
    type_str: str,
    dynamics_model: "DynamicsModel",
    vehicle_params: VehicleParams,
    controller_config: ControllerConfig,
) -> Controller:
    cls = CONTROLLER_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown controller '{type_str}'. "
            f"Available: {list(CONTROLLER_REGISTRY.keys())}"
        )
    return cls(
        dynamics_model=dynamics_model,
        vehicle_params=vehicle_params,
        config=controller_config,
    )
