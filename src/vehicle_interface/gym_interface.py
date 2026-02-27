"""
F1TENTH Gym vehicle interface.

Wraps the f1tenth_gym OpenAI Gym environment.
Translates between our IDL messages (ControlCmdMsg with acceleration/steering_vel)
and the gym's action format ([[steering_velocity, acceleration]]).

The gym has been patched (base_classes.py) to accept raw acceleration and
steering velocity directly, bypassing the internal PID.
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
        self._state = self._extract_state()

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

    def apply_action(self, cmd: ControlCmdMsg) -> bool:
        """Apply action and return True if the episode is done."""
        self._last_action = cmd

        # Gym (patched) expects [[steering_velocity, acceleration]].
        action = np.array([[cmd.steering_vel, cmd.acceleration]])
        obs, _, done, _ = self._env.step(action)

        self._state = self._extract_state()

        # Track collision flag from the gym environment.
        self._collision = bool(self._env.collisions[self._env.ego_idx])

        if done:
            reason = "collision" if self._collision else "lap completed"
            log.warning("Gym env done (%s).", reason)

        return bool(done)

    @property
    def collision(self) -> bool:
        """True if the last step resulted in a collision."""
        return getattr(self, '_collision', False)

    def set_planned_trajectory(self, xy_points: np.ndarray) -> None:
        """Set MCTS planned trajectory for visualization. xy_points: (N, 2)."""
        self._planned_trajectory = xy_points

    def setup_rendering(self, waypoints: WaypointsMsg) -> None:
        """Register render callback that draws the raceline and follows the car."""
        from pyglet.gl import GL_POINTS, GL_QUADS

        # Pre-compute scaled waypoint vertices for drawing.
        points = np.column_stack([waypoints.x, waypoints.y])
        scaled_points = 50.0 * points
        self._drawn_waypoints: List = []
        self._drawn_trajectory: List = []
        self._planned_trajectory: np.ndarray = np.empty((0, 2))

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

            # Draw waypoints once (they never move, so no need to update vertices).
            if len(self._drawn_waypoints) < scaled_points.shape[0]:
                for i in range(scaled_points.shape[0]):
                    b = e.batch.add(
                        1, GL_POINTS, None,
                        ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                        ('c3B/stream', [183, 193, 222]),
                    )
                    self._drawn_waypoints.append(b)

            # Draw MCTS planned trajectory (updated each step).
            # Remove old trajectory points.
            for b in self._drawn_trajectory:
                b.delete()
            self._drawn_trajectory.clear()

            traj = self._planned_trajectory
            if traj is not None and len(traj) > 0:
                sz = 8.0  # half-size of each quad in screen units
                for i in range(len(traj)):
                    # Color gradient: green (start) -> red (end)
                    t = i / max(len(traj) - 1, 1)
                    r = int(255 * t)
                    g = int(255 * (1 - t))
                    b_color = 0
                    sx, sy = 50.0 * traj[i, 0], 50.0 * traj[i, 1]
                    b = e.batch.add(
                        4, GL_QUADS, None,
                        ('v3f/stream', [
                            sx - sz, sy - sz, 0.,
                            sx + sz, sy - sz, 0.,
                            sx + sz, sy + sz, 0.,
                            sx - sz, sy + sz, 0.,
                        ]),
                        ('c3B/stream', [r, g, b_color] * 4),
                    )
                    self._drawn_trajectory.append(b)

        self._env.add_render_callback(render_callback)
        log.info("Render callback registered with %d waypoints.", len(waypoints.x))

    def render(self, mode: str = "human") -> None:
        self._env.render(mode=mode)

    # -- Internal helpers ---------------------------------------------------

    def _extract_state(self) -> VehicleStateMsg:
        """Extract full 7-state from the gym simulator's internal agent state.

        Gym internal state vector: [x, y, delta, v, psi, omega, beta]
        """
        agent_state = self._env.sim.agents[0].state
        return VehicleStateMsg(
            x=float(agent_state[0]),
            y=float(agent_state[1]),
            delta=float(agent_state[2]),
            v=float(agent_state[3]),
            psi=float(agent_state[4]),
            omega=float(agent_state[5]),
            beta=float(agent_state[6]),
        )
