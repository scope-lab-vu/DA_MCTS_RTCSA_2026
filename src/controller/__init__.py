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

from src.controller.simple_controller import SimpleController
from src.controller.mpc import MPCController
from src.controller.mppi import MPPIController
from src.controller.mppi_cornering_stiffness import MPPICorneringStiffnessController
from src.controller.mcts_dpw_py import MCTSDPWController


def _try_import(dotted_path, attr):
    """Import *attr* from *dotted_path*; return None if a C++ ext is missing."""
    try:
        mod = __import__(dotted_path, fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError) as exc:
        _log.getLogger(__name__).warning("Skipping %s (%s)", dotted_path, exc)
        return None


# MCTSDPWCppController             = _try_import("src.controller.mcts_dpw_cpp",                                          "MCTSDPWCppController")
# MCTSDPWPINNController            = _try_import("src.controller.mcts_dpw_pinn",                                         "MCTSDPWPINNController")
# MCTSDPWPinnCppController         = _try_import("src.controller.mcts_dpw_pinn_cpp",                                     "MCTSDPWPinnCppController")
# MCTSDPWSvgpCppController         = _try_import("src.controller.mcts_dpw_svgp_cpp",                                     "MCTSDPWSvgpCppController")
# MCTSPiSvgpCppController          = _try_import("src.controller.mcts_pi_svgp_cpp",                                      "MCTSPiSvgpCppController")
# MCTSPiSvgpCppCsController        = _try_import("src.controller.mcts_pi_svgp_cpp_cs",                                   "MCTSPiSvgpCppCsController")
# MCTSPiSvgpCppCsExtractController = _try_import("src.controller.mcts_pi_svgp_cpp_cs_extract_actions",                   "MCTSPiSvgpCppCsExtractController")
# MCTSKnownDynActionPriorCppController = _try_import("src.controller.mcts_known_dyn_action_prior_cpp",                   "MCTSKnownDynActionPriorCppController")
# MCTSPiSvgpCppCsCAlphaController  = _try_import("src.controller.mcts_pi_svgp_cpp_cs_c_alpha_input",                     "MCTSPiSvgpCppCsCAlphaController")
# MCTSDPWSvgpCAlphaCppController   = _try_import("src.controller.mcts_dpw_svgp_cpp_calpha_model",                        "MCTSDPWSvgpCppController")
# MCTSBudgetAwareCppController     = _try_import("src.controller.mcts_dpw_svgp_cpp_budget_aware_mcts",                   "MCTSDPWSvgpCppController")
# MCTSChanceNodeCppController      = _try_import("src.controller.mcts_dpw_svgp_cpp_calpha_model_sample_chance_node",     "MCTSDPWSvgpCppController")
# MCTSBudgetAwareChanceNodeCppController = _try_import("src.controller.mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node", "MCTSDPWSvgpCppController")
MCTSBudgetAwareChanceProbPredCppController = _try_import("src.controller.mcts_svgp_cpp_budget_aware_chance_prob_pred", "MCTSDPWSvgpCppController")

# Controllers that need extra kwargs in their constructor.
_NEEDS_MAP_CONFIG = {"mppi", "mppi_cs", "mcts_dpw_py", "mcts_dpw_cpp", "mcts_dpw_pinn", "mcts_dpw_pinn_cpp", "mcts_dpw_svgp_cpp", "mcts_pi_svgp_cpp", "mcts_pi_svgp_cpp_cs", "mcts_pi_svgp_cpp_cs_extract", "mcts_known_dyn_action_prior_cpp", "mcts_pi_svgp_cpp_cs_c_alpha_input", "mcts_dpw_svgp_cpp_calpha_model", "mcts_dpw_svgp_cpp_budget_aware_mcts", "mcts_dpw_svgp_cpp_calpha_model_sample_chance_node", "mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node", "mcts_svgp_cpp_budget_aware_chance_prob_pred"}
_NEEDS_SIMULATION_CONFIG = {"mppi", "mppi_cs", "mcts_dpw_py", "mcts_dpw_cpp", "mcts_dpw_pinn", "mcts_dpw_pinn_cpp", "mcts_dpw_svgp_cpp", "mcts_pi_svgp_cpp", "mcts_pi_svgp_cpp_cs", "mcts_pi_svgp_cpp_cs_extract", "mcts_known_dyn_action_prior_cpp", "mcts_pi_svgp_cpp_cs_c_alpha_input", "mcts_dpw_svgp_cpp_calpha_model", "mcts_dpw_svgp_cpp_budget_aware_mcts", "mcts_dpw_svgp_cpp_calpha_model_sample_chance_node", "mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node", "mcts_svgp_cpp_budget_aware_chance_prob_pred"}


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
    "simple": SimpleController,
    "mpc": MPCController,
    "mppi": MPPIController,
    "mppi_cs": MPPICorneringStiffnessController,
    # "mcts_dpw_py": MCTSDPWController,
    # "mcts_dpw_cpp": MCTSDPWCppController,
    # "mcts_dpw_pinn": MCTSDPWPINNController,
    # "mcts_dpw_pinn_cpp": MCTSDPWPinnCppController,
    # "mcts_dpw_svgp_cpp": MCTSDPWSvgpCppController,
    # "mcts_pi_svgp_cpp": MCTSPiSvgpCppController,
    # "mcts_pi_svgp_cpp_cs": MCTSPiSvgpCppCsController,
    # "mcts_pi_svgp_cpp_cs_extract": MCTSPiSvgpCppCsExtractController,
    # "mcts_known_dyn_action_prior_cpp": MCTSKnownDynActionPriorCppController,
    # "mcts_pi_svgp_cpp_cs_c_alpha_input": MCTSPiSvgpCppCsCAlphaController,
    # "mcts_dpw_svgp_cpp_calpha_model": MCTSDPWSvgpCAlphaCppController,
    # "mcts_dpw_svgp_cpp_budget_aware_mcts": MCTSBudgetAwareCppController,
    # "mcts_dpw_svgp_cpp_calpha_model_sample_chance_node": MCTSChanceNodeCppController,
    # "mcts_dpw_svgp_cpp_budget_aware_mcts_chance_node": MCTSBudgetAwareChanceNodeCppController,
    "mcts_svgp_cpp_budget_aware_chance_prob_pred": MCTSBudgetAwareChanceProbPredCppController,
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
