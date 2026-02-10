"""
Simple controller — working placeholder.

Holds a target speed and steers proportionally toward the nearest waypoint.
Good enough to verify the full pipeline runs end-to-end.
"""

import math

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, VehicleParams


class SimpleController:
    """Proportional speed + steer-to-nearest-waypoint controller."""

    def __init__(
        self,
        dynamics_model: "DynamicsModel",
        vehicle_params: VehicleParams,
        config: ControllerConfig,
    ) -> None:
        self._dynamics_model = dynamics_model  # available for future use
        self._vehicle_params = vehicle_params
        self._target_speed = config.target_speed_mps
        self._kp_steer = 1.0   # proportional gain for steering
        self._kp_speed = 2.0   # proportional gain for speed

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        # --- Speed control: proportional to error ---
        speed_error = self._target_speed - state.v
        acceleration = self._kp_speed * speed_error

        # --- Steering: aim toward nearest waypoint ---
        steering_vel = 0.0
        if len(waypoints.x) > 0:
            # Find nearest waypoint.
            best_dist = float("inf")
            target_x, target_y = waypoints.x[0], waypoints.y[0]
            for wx, wy in zip(waypoints.x, waypoints.y):
                dist = math.hypot(wx - state.x, wy - state.y)
                if dist < best_dist:
                    best_dist = dist
                    target_x, target_y = wx, wy

            # Desired heading toward that waypoint.
            desired_heading = math.atan2(target_y - state.y, target_x - state.x)
            heading_error = desired_heading - state.psi
            # Wrap to [-pi, pi].
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            steering_vel = self._kp_steer * heading_error

        return ControlCmdMsg(
            acceleration=acceleration,
            steering_vel=steering_vel,
        )
