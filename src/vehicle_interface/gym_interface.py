"""
F1TENTH Gym vehicle interface.

Wraps the f1tenth_gym OpenAI Gym environment.
Translates between our IDL messages (ControlCmdMsg with acceleration/steering_vel)
and the gym's action format ([steering_angle, speed]).
"""

import logging
from typing import List

import gym
import numpy as np
from f110_gym.envs.base_classes import Integrator

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import MapConfig, SimulationConfig, VehicleParams

log = logging.getLogger(__name__)


class GymVehicleInterface:
    """Vehicle interface backed by the f1tenth_gym simulator."""

    def __init__(
        self,
        vehicle_params: VehicleParams,
        simulation_config: SimulationConfig,
        map_config: MapConfig,
    ) -> None:
        self._vehicle_params = vehicle_params
        self._map_config = map_config
        self._dt = simulation_config.dt_s

        # Internal commanded values (integrated from acceleration / steering_vel).
        self._cmd_speed: float = 0.0
        self._cmd_steer: float = 0.0

        self._last_action = ControlCmdMsg()
        self._state = VehicleStateMsg()

        # Build gym environment with vehicle params.
        self._env = gym.make(
            "f110_gym:f110-v0",
            map=map_config.map_path,
            map_ext=map_config.map_ext,
            num_agents=1,
            timestep=simulation_config.dt_s,
            integrator=Integrator.Euler,
            params=vehicle_params.to_gym_dict(),
        )

        # Reset at starting pose.
        self._start_pose = np.array([[
            map_config.sx,
            map_config.sy,
            map_config.stheta,
        ]])
        obs, _, _, _ = self._env.reset(self._start_pose)
        self._state = self._obs_to_state(obs)

        log.info(
            "Gym env ready: map=%s  start=(%.2f, %.2f, %.2f)",
            map_config.map_path,
            map_config.sx,
            map_config.sy,
            map_config.stheta,
        )

    # -- VehicleInterface contract ------------------------------------------

    def get_state(self) -> VehicleStateMsg:
        return self._state

    def get_last_applied_action(self) -> ControlCmdMsg:
        return self._last_action

    def apply_action(self, cmd: ControlCmdMsg) -> None:
        self._last_action = cmd

        # Integrate acceleration → speed, steering_vel → steering_angle.
        self._cmd_speed += cmd.acceleration * self._dt
        self._cmd_speed = max(self._vehicle_params.v_min,
                              min(self._vehicle_params.v_max, self._cmd_speed))

        self._cmd_steer += cmd.steering_vel * self._dt
        self._cmd_steer = max(self._vehicle_params.s_min,
                              min(self._vehicle_params.s_max, self._cmd_steer))

        # Gym expects [[steer_angle, speed]].
        action = np.array([[self._cmd_steer, self._cmd_speed]])
        obs, _, done, _ = self._env.step(action)

        self._state = self._obs_to_state(obs)

        if done:
            log.warning("Gym env done (collision or out of bounds). Resetting.")
            obs, _, _, _ = self._env.reset(self._start_pose)
            self._state = self._obs_to_state(obs)
            self._cmd_speed = 0.0
            self._cmd_steer = 0.0

    def setup_rendering(self, waypoints: WaypointsMsg) -> None:
        """Register render callback that draws the raceline and follows the car."""
        from pyglet.gl import GL_POINTS

        # Pre-compute scaled waypoint vertices for drawing.
        points = np.column_stack([waypoints.x, waypoints.y])
        scaled_points = 50.0 * points
        self._drawn_waypoints: List = []

        def render_callback(env_renderer):
            e = env_renderer

            # Update camera to follow car.
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left = left - 800
            e.right = right + 800
            e.top = top + 800
            e.bottom = bottom - 800

            # Draw waypoints as points.
            for i in range(scaled_points.shape[0]):
                if len(self._drawn_waypoints) < scaled_points.shape[0]:
                    b = e.batch.add(
                        1, GL_POINTS, None,
                        ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                        ('c3B/stream', [183, 193, 222]),
                    )
                    self._drawn_waypoints.append(b)
                else:
                    self._drawn_waypoints[i].vertices = [
                        scaled_points[i, 0], scaled_points[i, 1], 0.
                    ]

        self._env.add_render_callback(render_callback)
        log.info("Render callback registered with %d waypoints.", len(waypoints.x))

    def render(self, mode: str = "human") -> None:
        self._env.render(mode=mode)

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _obs_to_state(obs: dict) -> VehicleStateMsg:
        return VehicleStateMsg(
            x=float(obs["poses_x"][0]),
            y=float(obs["poses_y"][0]),
            psi=float(obs["poses_theta"][0]),
            v=float(obs["linear_vels_x"][0]),
            beta=0.0,  # not directly in gym obs
            omega=float(obs["ang_vels_z"][0]),
        )
