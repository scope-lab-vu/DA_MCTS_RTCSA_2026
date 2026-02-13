"""
F1TENTH simulator wrapper for MCTS tree search.

Wraps the dynamics model and track utilities into the simulator interface
expected by the MCTS tree (step, sample_action, copy).
"""

import copy
import math

import numpy as np
import torch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg


class F1TenthSimulator:
    """Lightweight simulator for MCTS rollouts.

    Wraps a dynamics model and computes rewards using the same cost
    function as MPPI (negated, since MCTS maximises reward):
        reward = -(0.9^t * track_cost + speed_weight * speed_cost + slip_cost)

    :param state_array: np.array [x, y, delta, v, psi, omega, beta]
    :param dynamics_model: DynamicsModel with predict_next_state()
    :param vehicle_params: VehicleParams
    :param track: Track object (for progress computation)
    :param track_occ_map: TrackOccupancyMap for O(1) off-track detection
    :param dt: simulation timestep
    :param v_ref: fallback reference velocity for speed cost
    :param v_ref_window: np.array of per-step reference velocities (length >= rollout_horizon)
    :param speed_weight: weight on speed cost term
    :param alpha_threshold: slip angle threshold for slip penalty
    :param rollout_horizon: max steps in random rollout
    :param discount: discount factor for rollout returns
    """

    def __init__(
        self,
        state_array,
        dynamics_model,
        vehicle_params,
        track,
        track_occ_map,
        dt,
        v_ref,
        v_ref_window,
        speed_weight,
        alpha_threshold,
        rollout_horizon,
        discount,
    ):
        self.state = state_array.copy()
        self.dynamics_model = dynamics_model
        self.vehicle_params = vehicle_params
        self.track = track
        self.track_occ_map = track_occ_map
        self.dt = dt
        self.v_ref = v_ref
        self.v_ref_window = v_ref_window
        self.speed_weight = speed_weight
        self.alpha_threshold = alpha_threshold
        self.rollout_horizon = rollout_horizon
        self.discount = discount
        self._step_count = 0  # tracks t for 0.9^t discount on track cost

        # Action bounds.
        p = vehicle_params
        self._a_min = np.array([-p.a_max, p.sv_min])
        self._a_max = np.array([p.a_max, p.sv_max])

        # --- Discrete action set (uncomment to use) ---
        # self.is_action_discrete = True
        # accels = np.linspace(-p.a_max, p.a_max, 3)
        # steers = np.linspace(p.sv_min, p.sv_max, 10)
        # grid = np.meshgrid(accels, steers)
        # self._discrete_actions = np.column_stack([g.ravel() for g in grid])

    def step(self, action_array):
        """Advance one step.

        :param action_array: np.array [acceleration, steering_vel]
        :return: (next_state, reward, done, info)
        """
        s = self.state

        # Build messages for dynamics model.
        state_msg = VehicleStateMsg(
            x=float(s[0]), y=float(s[1]), delta=float(s[2]),
            v=float(s[3]), psi=float(s[4]), omega=float(s[5]),
            beta=float(s[6]),
        )
        cmd_msg = ControlCmdMsg(
            acceleration=float(action_array[0]),
            steering_vel=float(action_array[1]),
        )

        # Predict next state via dynamics model.
        next_msg = self.dynamics_model.predict_next_state(state_msg, cmd_msg, self.dt)
        next_state = np.array([
            next_msg.x, next_msg.y, next_msg.delta, next_msg.v,
            next_msg.psi, next_msg.omega, next_msg.beta,
        ])

        self.state = next_state
        t = self._step_count
        self._step_count += 1

        # --- Cost computation (same as MPPI, negated → reward) ---
        v = next_state[3]
        beta_val = next_state[6]

        # Track cost: 0.9^t * off_track * 10000
        pos_tensor = torch.tensor(
            [[next_state[0], next_state[1]]],
            dtype=torch.float32,
        )
        off_track = float(self.track_occ_map.compute_cost(pos_tensor)[0])
        track_cost = (0.9 ** t) * off_track * 10000.0

        # Speed cost: (v - v_ref_t)^2  (per-step reference, same as MPPI)
        if self.v_ref_window is not None and t < len(self.v_ref_window):
            v_ref_t = self.v_ref_window[t]
        else:
            v_ref_t = self.v_ref
        speed_cost = (v - v_ref_t) ** 2

        # Slip cost: alpha^2 + 10000 * I(|alpha| > threshold)
        v_x = v * math.cos(beta_val)
        v_y = v * math.sin(beta_val)
        alpha = -math.atan2(v_y, abs(v_x) + 1e-8)
        slip_cost = alpha ** 2 + 10000.0 * (abs(alpha) > self.alpha_threshold)

        # Reward = negated MPPI cost (no termination — soft penalties only).
        reward = -(track_cost + self.speed_weight * speed_cost + slip_cost)
        done = False

        return next_state, reward, done, {}

    def sample_action(self):
        """Sample a random action uniformly from the continuous action space."""
        return np.random.uniform(self._a_min, self._a_max)
        # --- Discrete sampling (uncomment to use, comment out line above) ---
        # idx = np.random.randint(len(self._discrete_actions))
        # return self._discrete_actions[idx]

    def copy(self):
        """Create a shallow copy (dynamics model shared, state/counter copied)."""
        new_sim = copy.copy(self)
        new_sim.state = self.state.copy()
        new_sim._step_count = self._step_count
        return new_sim
