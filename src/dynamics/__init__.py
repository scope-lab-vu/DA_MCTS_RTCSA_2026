"""
Dynamics Model — predicts next vehicle state given current state + action.

ABC:  DynamicsModel
Factory: create_dynamics_model(type_str, vehicle_params)

Models used with MPPI must also implement:
    setup_torch(dt, device, dtype)   — configure for batched torch inference
    batched_dynamics(state, action, t) — (K,7) x (K,2) → (K,7) torch tensors
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg
from src.params import DynamicsConfig, SimulationConfig, VehicleParams

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

    # -- Optional batched-torch interface (required for MPPI) ---------------

    def setup_torch(self, dt: float, device: str, dtype: torch.dtype) -> None:
        """Configure for batched torch dynamics. Override in MPPI-compatible models."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support batched torch dynamics."
        )

    def batched_dynamics(
        self, state: torch.Tensor, action: torch.Tensor, t: int,
    ) -> torch.Tensor:
        """Batched dynamics: (K,7) state x (K,2) action → (K,7) next_state."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support batched torch dynamics."
        )


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
    dynamics_config: Optional[DynamicsConfig] = None,
    simulation_config: Optional[SimulationConfig] = None,
) -> DynamicsModel:
    cls = DYNAMICS_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown dynamics model '{type_str}'. "
            f"Available: {list(DYNAMICS_REGISTRY.keys())}"
        )
    if type_str == "pinn":
        return cls(
            vehicle_params=vehicle_params,
            dynamics_config=dynamics_config,
            simulation_config=simulation_config,
        )
    return cls(vehicle_params=vehicle_params)
