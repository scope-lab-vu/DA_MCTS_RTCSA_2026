"""
MPPI (Model Predictive Path Integral) controller for F1TENTH racing.

Uses pytorch_mppi library for the core MPPI algorithm.
Implements F1TENTH single-track vehicle dynamics (batched torch).

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
from src.params import ControllerConfig, MapConfig, VehicleParams
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
    ) -> None:
        self._vehicle_params = vehicle_params
        self._config = config

        p = vehicle_params
        self.lf = p.lf
        self.lr = p.lr
        self.L = p.lf + p.lr
        self.mu = p.mu
        self.C_Sf = p.C_Sf
        self.C_Sr = p.C_Sr
        self.h = p.h
        self.m = p.m
        self.I = p.I
        self.v_switch = p.v_switch
        self.g = 9.81

        self.a_max = p.a_max
        self.sv_min = p.sv_min
        self.sv_max = p.sv_max
        self.s_min = p.s_min
        self.s_max = p.s_max
        self.v_min = p.v_min
        self.v_max = p.v_max

        self.dt = 0.02  # internal MPPI dynamics dt (can differ from sim dt)
        self.horizon = config.horizon
        self.v_ref = config.v_ref
        self.device = config.device
        self.dtype = torch.float32

        self.nx = 7  # [x, y, delta, v, psi, omega, beta]
        self.nu = 2  # [acceleration, steering_rate]

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

        # Action bounds
        u_min = torch.tensor([-self.a_max, self.sv_min], device=self.device, dtype=self.dtype)
        u_max = torch.tensor([self.a_max, self.sv_max], device=self.device, dtype=self.dtype)

        # Create MPPI controller
        self.mppi = mppi.MPPI(
            dynamics=self._dynamics,
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
            "MPPI ready: horizon=%d samples=%d lambda=%.1f device=%s",
            self.horizon, config.num_samples, config.lambda_, self.device,
        )

    # -- Batched dynamics (torch) -------------------------------------------

    def _accl_constraints_batch(self, vel: torch.Tensor, accl: torch.Tensor) -> torch.Tensor:
        pos_limit = torch.where(
            vel > self.v_switch,
            self.a_max * self.v_switch / (vel + 1e-8),
            torch.full_like(vel, self.a_max),
        )
        result = accl.clone()
        at_min = (vel <= self.v_min) & (accl <= 0)
        at_max = (vel >= self.v_max) & (accl >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(accl), result)
        result = torch.maximum(result, torch.full_like(result, -self.a_max))
        result = torch.minimum(result, pos_limit)
        return result

    def _steering_constraint_batch(self, steering_angle: torch.Tensor, steering_velocity: torch.Tensor) -> torch.Tensor:
        result = steering_velocity.clone()
        at_min = (steering_angle <= self.s_min) & (steering_velocity <= 0)
        at_max = (steering_angle >= self.s_max) & (steering_velocity >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(steering_velocity), result)
        result = torch.clamp(result, self.sv_min, self.sv_max)
        return result

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor, t: int) -> torch.Tensor:
        """Batched single-track dynamics. State: (K,7), Action: (K,2)."""
        x = state[:, 0]
        y = state[:, 1]
        delta = state[:, 2]
        v = state[:, 3]
        psi = state[:, 4]
        omega = state[:, 5]
        beta = state[:, 6]

        accl = action[:, 0]
        delta_dot = action[:, 1]

        u_delta_dot = self._steering_constraint_batch(delta, delta_dot)
        u_accl = self._accl_constraints_batch(v, accl)

        use_ks = torch.abs(v) < 0.5

        # Kinematic model
        f_ks_x = v * torch.cos(psi)
        f_ks_y = v * torch.sin(psi)
        f_ks_delta = u_delta_dot
        f_ks_v = u_accl
        f_ks_psi = v / self.L * torch.tan(delta)
        f_ks_omega = u_accl / self.L * torch.tan(delta) + v / (self.L * torch.cos(delta) ** 2) * u_delta_dot
        f_ks_beta = torch.zeros_like(beta)

        # Dynamic model
        f_dyn_x = v * torch.cos(beta + psi)
        f_dyn_y = v * torch.sin(beta + psi)
        f_dyn_delta = u_delta_dot
        f_dyn_v = u_accl
        f_dyn_psi = omega

        v_safe = torch.clamp(torch.abs(v), min=0.5)
        f_dyn_omega = (
            -self.mu * self.m / (v_safe * self.I * self.L) * (
                self.lf ** 2 * self.C_Sf * (self.g * self.lr - u_accl * self.h)
                + self.lr ** 2 * self.C_Sr * (self.g * self.lf + u_accl * self.h)
            ) * omega
            + self.mu * self.m / (self.I * self.L) * (
                self.lr * self.C_Sr * (self.g * self.lf + u_accl * self.h)
                - self.lf * self.C_Sf * (self.g * self.lr - u_accl * self.h)
            ) * beta
            + self.mu * self.m / (self.I * self.L)
            * self.lf * self.C_Sf * (self.g * self.lr - u_accl * self.h) * delta
        )
        f_dyn_beta = (
            (self.mu / (v_safe ** 2 * self.L) * (
                self.C_Sr * (self.g * self.lf + u_accl * self.h) * self.lr
                - self.C_Sf * (self.g * self.lr - u_accl * self.h) * self.lf
            ) - 1) * omega
            - self.mu / (v_safe * self.L) * (
                self.C_Sr * (self.g * self.lf + u_accl * self.h)
                + self.C_Sf * (self.g * self.lr - u_accl * self.h)
            ) * beta
            + self.mu / (v_safe * self.L)
            * self.C_Sf * (self.g * self.lr - u_accl * self.h) * delta
        )

        f_x = torch.where(use_ks, f_ks_x, f_dyn_x)
        f_y = torch.where(use_ks, f_ks_y, f_dyn_y)
        f_delta = torch.where(use_ks, f_ks_delta, f_dyn_delta)
        f_v = torch.where(use_ks, f_ks_v, f_dyn_v)
        f_psi = torch.where(use_ks, f_ks_psi, f_dyn_psi)
        f_omega = torch.where(use_ks, f_ks_omega, f_dyn_omega)
        f_beta = torch.where(use_ks, f_ks_beta, f_dyn_beta)

        f = torch.stack([f_x, f_y, f_delta, f_v, f_psi, f_omega, f_beta], dim=1)
        next_state = state + self.dt * f

        # Normalize heading
        psi_new = next_state[:, 4]
        psi_new = torch.where(psi_new > 2 * np.pi, psi_new - 2 * np.pi, psi_new)
        psi_new = torch.where(psi_new < 0, psi_new + 2 * np.pi, psi_new)
        next_state = torch.cat([next_state[:, :4], psi_new.unsqueeze(1), next_state[:, 5:]], dim=1)

        return next_state

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
        v_ref_t = min(v_ref_t, 11.15)
        speed_cost = (velocities - v_ref_t) ** 2

        # Slip cost
        v_x = velocities * torch.cos(beta)
        v_y = velocities * torch.sin(beta)
        alpha = -torch.atan2(v_y, torch.abs(v_x) + 1e-8)
        slip_cost = alpha ** 2 + 10000.0 * (torch.abs(alpha) > 3.5).float()

        return track_cost + 5 * speed_cost + slip_cost

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
