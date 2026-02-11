"""
Dynamics Model — predicts next vehicle state given current state + action.

ABC:  DynamicsModel
Factory: create_dynamics_model(type_str, vehicle_params)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import VehicleParams

from src.dynamics.kinematic_bicycle import KinematicBicycleModel
from src.dynamics.f1tenth_dynamics import F1TenthDynamicsModel
from src.dynamics.svgp import SVGPModel
from src.dynamics.pinn import PINNModel


class DynamicsModel(ABC):
    """Contract for all dynamics models."""

    @abstractmethod
    def predict_next_state(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        dt: float,
    ) -> VehicleStateMsg:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DYNAMICS_REGISTRY: Dict[str, Type[DynamicsModel]] = {
    "kinematic_bicycle": KinematicBicycleModel,
    "f1tenth_dynamics": F1TenthDynamicsModel,
    "svgp": SVGPModel,
    "pinn": PINNModel,
}


def create_dynamics_model(
    type_str: str,
    vehicle_params: VehicleParams,
) -> DynamicsModel:
    cls = DYNAMICS_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown dynamics model '{type_str}'. "
            f"Available: {list(DYNAMICS_REGISTRY.keys())}"
        )
    return cls(vehicle_params=vehicle_params)
