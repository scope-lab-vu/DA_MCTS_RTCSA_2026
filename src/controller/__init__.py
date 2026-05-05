"""
Controller — computes a control command given vehicle state + waypoints.

ABC:  Controller
Factory: create_controller(type_str, dynamics_model, vehicle_params, controller_config, map_config)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams

import logging as _log


def _try_import(dotted_path, attr):
    """Import *attr* from *dotted_path*; return None if a C++ ext is missing."""
    try:
        mod = __import__(dotted_path, fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError) as exc:
        _log.getLogger(__name__).warning("Skipping %s (%s)", dotted_path, exc)
        return None


MCTSBudgetAwareChanceProbPredCppController = _try_import("src.controller.mcts_svgp_cpp_budget_aware_chance_prob_pred", "MCTSDPWSvgpCppController")
DAMctsNewRewardController = _try_import("src.controller.da_mcts_new_reward", "MCTSDPWSvgpCppController")
DAMctsTrainValueFunctionController = _try_import("src.controller.da_mcts_train_value_function", "MCTSDPWSvgpCppController")
StanleyMethodController = _try_import("src.controller.stanley_method", "StanleyMethodController")

_NEEDS_MAP_CONFIG = {"mcts_svgp_cpp_budget_aware_chance_prob_pred", "da_mcts_new_reward", "da_mcts_train_value_function", "stanley_method"}
_NEEDS_SIMULATION_CONFIG = {"mcts_svgp_cpp_budget_aware_chance_prob_pred", "da_mcts_new_reward", "da_mcts_train_value_function", "stanley_method"}


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

_ALL_CONTROLLERS = {
    "mcts_svgp_cpp_budget_aware_chance_prob_pred": MCTSBudgetAwareChanceProbPredCppController,
    "da_mcts_new_reward": DAMctsNewRewardController,
    "da_mcts_train_value_function": DAMctsTrainValueFunctionController,
    "stanley_method": StanleyMethodController,
}
# Drop entries whose C++ extension failed to import (value is None)
CONTROLLER_REGISTRY: Dict[str, Type[Controller]] = {
    k: v for k, v in _ALL_CONTROLLERS.items() if v is not None
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
