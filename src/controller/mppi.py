"""
MPPI (Model Predictive Path Integral) controller for F1TENTH racing.

Uses pytorch_mppi library for the core MPPI algorithm.
Delegates vehicle dynamics to the injected dynamics_model (swappable).

Cost function:
    c(s, t) = (0.9)^t * Track(s) + 5 * Speed(s) + Slip(s)
where:
    Track(s) = 10000 if off-track, else 0
    Speed(s) = (v - v_ref)^2
    Slip(s)  = alpha^2 + 10000 * I(|alpha| > 3.5)
"""

import logging

import numpy as np
import torch
from pytorch_mppi import mppi

from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams
from src.track import (
    ReferenceTrajectory,
    Track,
    TrackOccupancyMap,
    TrackTensor,
)

log = logging.getLogger(__name__)


class MPPIController:
    """MPPI controller for F1TENTH racing."""

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig,
        simulation_config: SimulationConfig = None,
    ) -> None:
        self._vehicle_params = vehicle_params
        self._config = config

        self.dt = simulation_config.dt_s if simulation_config is not None else 0.02
        self.horizon = config.horizon
        self.v_ref = config.v_ref
        self.device = config.device
        self.dtype = torch.float32
        self.v_max_cost = config.v_max_cost
        self.alpha_threshold = config.alpha_threshold
        self.speed_weight = config.speed_weight

        self.nx = 7  # [x, y, delta, v, psi, omega, beta]
        self.nu = 2  # [acceleration, steering_rate]

        # Configure the dynamics model for batched torch inference.
        self._dynamics_model = dynamics_model
        dynamics_model.setup_torch(self.dt, self.device, self.dtype)

        # Load track (centerline CSV)
        if map_config.centerline_path is None:
            raise ValueError(
                "MPPI controller requires centerline_path in map config. "
                "Add 'centerline_path' to your map YAML."
            )
        self.track = Track(map_config.centerline_path)
        self.track_tensor = TrackTensor(self.track, device=self.device, dtype=self.dtype)

        # Occupancy map for fast off-track detection
        self.use_occupancy_map = config.use_occupancy_map
        if self.use_occupancy_map:
            self.track_map = TrackOccupancyMap(
                track=self.track,
                cell_size=0.01,
                lane_width=config.lane_width,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.track_map = None

        # Reference trajectory (raceline) for velocity lookup
        self.ref_traj = None
        self.v_ref_window = None
        if map_config.wpt_path:
            try:
                self.ref_traj = ReferenceTrajectory(map_config.wpt_path)
                log.info("Loaded reference trajectory from %s", map_config.wpt_path)
            except Exception as e:
                log.warning("Could not load reference trajectory: %s. Using constant v_ref=%.1f", e, self.v_ref)

        # Noise covariance matrix
        noise_sigma = torch.tensor(
            [[config.noise_sigma_accel ** 2, 0],
             [0, config.noise_sigma_steer ** 2]],
            device=self.device, dtype=self.dtype,
        )

        # Action bounds (from vehicle params)
        p = vehicle_params
        u_min = torch.tensor([-p.a_max, p.sv_min], device=self.device, dtype=self.dtype)
        u_max = torch.tensor([p.a_max, p.sv_max], device=self.device, dtype=self.dtype)

        # Create MPPI controller — dynamics delegated to dynamics_model
        self.mppi = mppi.MPPI(
            dynamics=self._dynamics_model.batched_dynamics,
            running_cost=self._running_cost,
            nx=self.nx,
            noise_sigma=noise_sigma,
            num_samples=config.num_samples,
            horizon=self.horizon,
            device=self.device,
            lambda_=config.lambda_,
            u_min=u_min,
            u_max=u_max,
            step_dependent_dynamics=True,
        )

        log.info(
            "MPPI ready: horizon=%d samples=%d lambda=%.1f device=%s dynamics=%s",
            self.horizon, config.num_samples, config.lambda_, self.device,
            type(dynamics_model).__name__,
        )

    # -- Running cost -------------------------------------------------------

    def _running_cost(self, state: torch.Tensor, action: torch.Tensor, t: int) -> torch.Tensor:
        """c(s,t) = 0.9^t * Track(s) + 5*Speed(s) + Slip(s)"""
        positions = state[:, :2]
        velocities = state[:, 3]
        beta = state[:, 6]

        # Track cost
        discount = 0.9 ** t
        if self.use_occupancy_map and self.track_map is not None:
            off_track = self.track_map.compute_cost(positions)
        else:
            off_track = self.track_tensor.is_off_track_batch(positions, self._vehicle_params.width).float()
        track_cost = discount * off_track * 10000.0

        # Speed cost
        if self.v_ref_window is not None and t < len(self.v_ref_window):
            v_ref_t = self.v_ref_window[t]
        else:
            v_ref_t = self.v_ref
        v_ref_t = min(v_ref_t, self.v_max_cost)
        speed_cost = (velocities - v_ref_t) ** 2

        # Slip cost
        v_x = velocities * torch.cos(beta)
        v_y = velocities * torch.sin(beta)
        alpha = -torch.atan2(v_y, torch.abs(v_x) + 1e-8)
        slip_cost = alpha ** 2 + 10000.0 * (torch.abs(alpha) > self.alpha_threshold).float()

        return track_cost + self.speed_weight * speed_cost + slip_cost

    # -- Public interface (Controller contract) -----------------------------

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        # Convert VehicleStateMsg → numpy [x, y, delta, v, psi, omega, beta]
        state_np = np.array([
            state.x, state.y, state.delta, state.v,
            state.psi, state.omega, state.beta,
        ])

        # Update reference velocity window from raceline
        if self.ref_traj is not None:
            position = state_np[:2]
            v_ref_window_np = self.ref_traj.get_reference_velocity_window(
                position, self.horizon, self.dt,
            )
            self.v_ref_window = torch.tensor(v_ref_window_np, device=self.device, dtype=self.dtype)
        else:
            self.v_ref_window = None

        state_tensor = torch.tensor(state_np, device=self.device, dtype=self.dtype)
        action = self.mppi.command(state_tensor)
        action_np = action.cpu().numpy()

        return ControlCmdMsg(
            acceleration=float(action_np[0]),
            steering_vel=float(action_np[1]),
        )

    def reset(self):
        self.mppi.reset()
