"""
Controller — computes a control command given vehicle state + waypoints.

ABC:  Controller
Factory: create_controller(type_str, dynamics_model, vehicle_params, controller_config, map_config)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams

from src.controller.simple_controller import SimpleController
from src.controller.mpc import MPCController
from src.controller.mppi import MPPIController
from src.controller.mcts_dpw_py import MCTSDPWController
from src.controller.mcts_dpw_cpp import MCTSDPWCppController
from src.controller.mcts_dpw_pinn import MCTSDPWPINNController
from src.controller.mcts_dpw_pinn_cpp import MCTSDPWPinnCppController
from src.controller.mcts_dpw_svgp_cpp import MCTSDPWSvgpCppController
from src.controller.mcts_pi_svgp_cpp import MCTSPiSvgpCppController

# Controllers that need extra kwargs in their constructor.
_NEEDS_MAP_CONFIG = {"mppi", "mcts_dpw_py", "mcts_dpw_cpp", "mcts_dpw_pinn", "mcts_dpw_pinn_cpp", "mcts_dpw_svgp_cpp", "mcts_pi_svgp_cpp"}
_NEEDS_SIMULATION_CONFIG = {"mppi", "mcts_dpw_py", "mcts_dpw_cpp", "mcts_dpw_pinn", "mcts_dpw_pinn_cpp", "mcts_dpw_svgp_cpp", "mcts_pi_svgp_cpp"}


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
    "mcts_dpw_py": MCTSDPWController,
    "mcts_dpw_cpp": MCTSDPWCppController,
    "mcts_dpw_pinn": MCTSDPWPINNController,
    "mcts_dpw_pinn_cpp": MCTSDPWPinnCppController,
    "mcts_dpw_svgp_cpp": MCTSDPWSvgpCppController,
    "mcts_pi_svgp_cpp": MCTSPiSvgpCppController,
}


def create_controller(
    type_str: str,
    dynamics_model: "DynamicsModel",
    vehicle_params: VehicleParams,
    controller_config: ControllerConfig,
    map_config: Optional[MapConfig] = None,
    simulation_config: Optional[SimulationConfig] = None,
) -> Controller:
    cls = CONTROLLER_REGISTRY.get(type_str)
    if cls is None:
        raise ValueError(
            f"Unknown controller '{type_str}'. "
            f"Available: {list(CONTROLLER_REGISTRY.keys())}"
        )
    kwargs = dict(
        dynamics_model=dynamics_model,
        vehicle_params=vehicle_params,
        config=controller_config,
    )
    if type_str in _NEEDS_MAP_CONFIG:
        kwargs["map_config"] = map_config
    if type_str in _NEEDS_SIMULATION_CONFIG:
        kwargs["simulation_config"] = simulation_config
    return cls(**kwargs)
