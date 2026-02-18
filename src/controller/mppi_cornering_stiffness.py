"""
MPPI Controller with PI-SVGP dynamics and online cornering stiffness adaptation.

Extends MPPIController with three additions:

  1. Internal C_alpha estimation — rear cornering stiffness is computed from
     consecutive VehicleStateMsg observations (no external input needed).

  2. Rolling-window latent update — the mean C_alpha over the last
     `calpha_window_size` steps is used to call
     `dynamics_model.set_latents_from_calpha()`, smoothly updating the
     PI-SVGP task latent parameters on-the-fly.

  3. C_alpha-adaptive reference velocity — the running cost penalises
     deviations from a v_ref that is reduced when the estimated surface
     grip is low:

         delta_C = max(0,  C_alpha_th - cornering_stiffness_{t-1})
         v_ref_t -= cs_w1 * log(cs_w2 * delta_C + cs_w3)   [when delta_C > 0]
         v_ref_t  = max(v_ref_t, 3.0)

Cost function (full):
    c(s, t) = 0.9^t * Track(s) + speed_weight * Speed(s) + Slip(s)
where:
    Track(s) = 10000 if off-track, else occupancy-map cost
    Speed(s) = (v - v_ref_adapted)^2
    Slip(s)  = alpha^2 + 10000 * I(|alpha| > alpha_threshold)
"""

import logging
import math
from typing import Optional

import numpy as np
import torch

from src.controller.mppi import MPPIController
from src.idl_messages import ControlCmdMsg, VehicleStateMsg, WaypointsMsg
from src.params import ControllerConfig, MapConfig, SimulationConfig, VehicleParams

log = logging.getLogger(__name__)

# Small value to avoid division by zero in C_alpha formula
_EPSILON = 1e-8

# Nominal high-friction cornering stiffness — used as the initial value and
# as the clip upper bound (matches task-0 anchor in SVGP training data).
_C_ALPHA_NOMINAL = 104.89


def _compute_cornering_stiffness(
    prev: VehicleStateMsg,
    curr: VehicleStateMsg,
    dt: float,
    lf: float,
    lr: float,
    mass: float,
    iz: float,
) -> float:
    """Estimate rear cornering stiffness from two consecutive states.

    Derivation (linearised single-track model):
        F_yr = (lf * m * (v_y_dot + v_x * omega) - Iz * omega_dot) / (lf + lr)
        alpha_r = atan2(lr * omega - v_y, |v_x|)
        C_r = |F_yr| / (|alpha_r| + eps)

    Derivatives are approximated with a forward finite-difference from t-1 to t.

    Args:
        prev: State at t-1  [x, y, delta, v, psi, omega, beta]
        curr: State at t    [x, y, delta, v, psi, omega, beta]
        dt:   Timestep [s]
        lf, lr: CoG-to-axle distances [m]
        mass: Vehicle mass [kg]
        iz:   Yaw moment of inertia [kg·m²]

    Returns:
        Estimated rear cornering stiffness [N/rad], always non-negative.
    """
    v_x_prev = prev.v * math.cos(prev.beta)
    v_y_prev = prev.v * math.sin(prev.beta)
    v_y_curr = curr.v * math.sin(curr.beta)

    v_y_dot = (v_y_curr - v_y_prev) / dt
    omega_dot = (curr.omega - prev.omega) / dt

    F_yr = (
        lf * mass * (v_y_dot + v_x_prev * prev.omega) - iz * omega_dot
    ) / (lf + lr)

    alpha_r = math.atan2(lr * prev.omega - v_y_prev, abs(v_x_prev) + _EPSILON)

    return abs(F_yr) / (abs(alpha_r) + _EPSILON)


class MPPICorneringStiffnessController(MPPIController):
    """MPPI + PI-SVGP with online cornering stiffness adaptation.

    Inherits all MPPI, track-cost, and occupancy-map logic from
    ``MPPIController``.  The only behavioural additions are the C_alpha
    estimation, rolling-window latent update, and the adaptive speed target
    in ``_running_cost``.

    Requires that the injected ``dynamics_model`` exposes
    ``set_latents_from_calpha(c_alpha: float)``, which ``PiSVGPModel`` does.
    """

    def __init__(
        self,
        dynamics_model,
        vehicle_params: VehicleParams,
        config: ControllerConfig,
        map_config: MapConfig,
        simulation_config: SimulationConfig = None,
    ) -> None:
        super().__init__(
            dynamics_model=dynamics_model,
            vehicle_params=vehicle_params,
            config=config,
            map_config=map_config,
            simulation_config=simulation_config,
        )

        # Cornering stiffness velocity-adaptation parameters
        self._cs_w1 = config.cs_w1
        self._cs_w2 = config.cs_w2
        self._cs_w3 = config.cs_w3
        self._C_alpha_th = config.C_alpha_th
        self._calpha_window_size = config.calpha_window_size

        # Runtime state for C_alpha tracking
        self._prev_state: Optional[VehicleStateMsg] = None
        self._cornering_stiffness: float = _C_ALPHA_NOMINAL
        self._calpha_buffer: list = []

        # Cache vehicle params needed for the C_alpha formula
        self._lf = vehicle_params.lf
        self._lr = vehicle_params.lr
        self._mass = vehicle_params.m
        self._iz = vehicle_params.I

        log.info(
            "MPPICorneringStiffnessController ready: "
            "cs_w1=%.2f  cs_w2=%.2f  cs_w3=%.2f  "
            "C_alpha_th=%.2f  window=%d",
            self._cs_w1, self._cs_w2, self._cs_w3,
            self._C_alpha_th, self._calpha_window_size,
        )

    # -- Running cost with C_alpha-adaptive speed target --------------------

    def _running_cost(
        self, state: torch.Tensor, action: torch.Tensor, t: int,
    ) -> torch.Tensor:
        """c(s,t) = 0.9^t * Track + speed_weight * Speed_cs + Slip

        Speed_cs uses a v_ref that is reduced whenever estimated cornering
        stiffness falls below C_alpha_th:

            delta_C = max(0, C_alpha_th - CS_{t-1})
            v_ref_t -= cs_w1 * log(cs_w2 * delta_C + cs_w3)   [if applicable]
        """
        positions = state[:, :2]
        velocities = state[:, 3]
        beta = state[:, 6]

        # -- Track cost -----------------------------------------------------
        discount = 0.9 ** t
        if self.use_occupancy_map and self.track_map is not None:
            off_track = self.track_map.compute_cost(positions)
        else:
            off_track = self.track_tensor.is_off_track_batch(
                positions, self._vehicle_params.width
            ).float()
        track_cost = discount * off_track * 10000.0

        # -- Speed cost with CS adaptation ----------------------------------
        if self.v_ref_window is not None and t < len(self.v_ref_window):
            v_ref_t = float(self.v_ref_window[t])
        else:
            v_ref_t = self.v_ref

        # Reduce v_ref when grip is estimated to be low
        delta_C_alpha = max(0.0, self._C_alpha_th - self._cornering_stiffness)
        if delta_C_alpha > 0.0:
            log_arg = self._cs_w2 * delta_C_alpha + self._cs_w3
            if log_arg > 0.0:
                v_ref_t = v_ref_t - self._cs_w1 * math.log(log_arg)
                v_ref_t = max(v_ref_t, 3.0)

        v_ref_t = min(v_ref_t, self.v_max_cost)
        speed_cost = (velocities - v_ref_t) ** 2

        # -- Slip cost ------------------------------------------------------
        v_x = velocities * torch.cos(beta)
        v_y = velocities * torch.sin(beta)
        alpha = -torch.atan2(v_y, torch.abs(v_x) + 1e-8)
        slip_cost = alpha ** 2 + 10000.0 * (torch.abs(alpha) > self.alpha_threshold).float()

        return track_cost + self.speed_weight * speed_cost + slip_cost

    # -- Public interface ---------------------------------------------------

    def compute_action(
        self,
        state: VehicleStateMsg,
        waypoints: WaypointsMsg,
    ) -> ControlCmdMsg:
        """Compute control command with internal C_alpha estimation.

        Steps:
          1. Estimate cornering stiffness from (prev_state, state).
          2. Update rolling C_alpha buffer and compute smoothed mean.
          3. Push smoothed C_alpha to PI-SVGP latent parameters.
          4. Cache current state as prev_state.
          5. Delegate to base MPPIController.compute_action().
        """
        # Step 1: estimate cornering stiffness
        if self._prev_state is not None:
            cs_raw = _compute_cornering_stiffness(
                self._prev_state, state, self.dt,
                self._lf, self._lr, self._mass, self._iz,
            )
            cs_clipped = float(np.clip(cs_raw, 0.0, _C_ALPHA_NOMINAL))
        else:
            cs_clipped = _C_ALPHA_NOMINAL

        self._cornering_stiffness = cs_clipped

        # Step 2: rolling window smooth
        self._calpha_buffer.append(cs_clipped)
        if len(self._calpha_buffer) > self._calpha_window_size:
            self._calpha_buffer = self._calpha_buffer[-self._calpha_window_size:]
        c_alpha_mean = float(np.mean(self._calpha_buffer))

        # Step 3: update PI-SVGP latents
        if hasattr(self._dynamics_model, 'set_latents_from_calpha'):
            self._dynamics_model.set_latents_from_calpha(c_alpha_mean)

        # Step 4: cache state
        self._prev_state = state

        # Step 5: startup phase — below 0.5 m/s apply full throttle, zero steer
        if state.v < 0.5:
            return ControlCmdMsg(
                acceleration=self._vehicle_params.a_max,
                steering_vel=0.0,
            )

        # Step 6: standard MPPI command
        return super().compute_action(state, waypoints)

    def reset(self) -> None:
        super().reset()
        self._prev_state = None
        self._cornering_stiffness = _C_ALPHA_NOMINAL
        self._calpha_buffer = []

    # -- Introspection helpers ----------------------------------------------

    def get_cornering_stiffness(self) -> float:
        """Return the most recently estimated cornering stiffness [N/rad]."""
        return self._cornering_stiffness

    def get_smoothed_calpha(self) -> float:
        """Return the rolling-window mean of C_alpha used for latent update."""
        if not self._calpha_buffer:
            return _C_ALPHA_NOMINAL
        return float(np.mean(self._calpha_buffer))
