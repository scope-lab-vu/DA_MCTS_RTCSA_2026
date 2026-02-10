"""
F1TenthAPI — single entry-point that wires all components together.

main.py only talks to this class.  All factory calls and component
interactions are hidden behind simple method calls.
"""

import logging
from pathlib import Path

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import (
    AlgoConfig,
    VehicleParams,
    load_algo_config,
    load_vehicle_params,
)
from src.dynamics import create_dynamics_model
from src.controller import create_controller
from src.safety import create_safety_monitor
from src.vehicle_interface import create_vehicle_interface

log = logging.getLogger(__name__)


class F1TenthAPI:
    """Orchestrates vehicle interface, dynamics, controller, and safety."""

    def __init__(
        self,
        vehicle_params_path: Path,
        algo_config_path: Path,
    ) -> None:
        # Load configuration.
        self.vehicle_params: VehicleParams = load_vehicle_params(vehicle_params_path)
        self.algo_config: AlgoConfig = load_algo_config(algo_config_path)

        log.info(
            "Building pipeline: interface=%s  dynamics=%s  controller=%s  safety=%s",
            self.algo_config.simulation.vehicle_interface,
            self.algo_config.dynamics.type,
            self.algo_config.controller.type,
            self.algo_config.safety.type,
        )

        # Construct components via factories.
        self._dynamics = create_dynamics_model(
            type_str=self.algo_config.dynamics.type,
            vehicle_params=self.vehicle_params,
        )
        self._controller = create_controller(
            type_str=self.algo_config.controller.type,
            dynamics_model=self._dynamics,
            vehicle_params=self.vehicle_params,
            controller_config=self.algo_config.controller,
        )
        self._safety = create_safety_monitor(
            type_str=self.algo_config.safety.type,
            vehicle_params=self.vehicle_params,
        )
        self._vehicle = create_vehicle_interface(
            type_str=self.algo_config.simulation.vehicle_interface,
            vehicle_params=self.vehicle_params,
            dynamics_model=self._dynamics,
            dt=self.algo_config.simulation.dt_s,
        )

        log.info("Pipeline ready.")

    # -- Public API (what main.py calls) ------------------------------------

    def get_state(self) -> VehicleStateMsg:
        return self._vehicle.get_state()

    def get_last_applied_action(self) -> ControlCmdMsg:
        return self._vehicle.get_last_applied_action()

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        raw_cmd = self._controller.compute_action(state, waypoints)
        safe_cmd = self._safety.filter_action(state, raw_cmd)
        return safe_cmd

    def apply_action(self, cmd: ControlCmdMsg) -> None:
        self._vehicle.apply_action(cmd)
