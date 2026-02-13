"""
F1TENTH selection simulator for MCTS tree traversal with PINN feature history.

Uses fast known dynamics (F1TenthDynamicsModel) for tree selection/expansion,
while maintaining a PINN-compatible feature history buffer at each node.
The feature history is passed to the batched PINN evaluator at leaf nodes.
"""

import copy
import math

import numpy as np
import torch

from src.idl_messages import ControlCmdMsg, VehicleStateMsg


class F1TenthSelectionSimulator:
    """Simulator for MCTS tree traversal using known dynamics + feature history.

    Uses F1TenthDynamicsModel.predict_next_state() for fast single-step
    dynamics during selection/expansion.  Maintains a PINN-compatible feature
    history tensor (pinn_horizon, 6) that tracks the state+action trajectory.

    The same cost function as mcts_dpw_py/simulator.py is used for selection
    rewards, while the feature history is captured at leaf nodes and passed
    to the BatchedPINNEvaluator for GPU-accelerated rollout evaluation.
    """

    def __init__(
        self,
        state_array,
        known_dynamics_model,
        vehicle_params,
        track_occ_map,
        dt,
        v_ref,
        v_ref_window,
        speed_weight,
        alpha_threshold,
        rollout_horizon,
        discount,
        feature_history,
        pinn_base_dt,
    ):
        self.state = state_array.copy()
        self.dynamics_model = known_dynamics_model
        self.vehicle_params = vehicle_params
        self.track_occ_map = track_occ_map
        self.dt = dt
        self.v_ref = v_ref
        self.v_ref_window = v_ref_window
        self.speed_weight = speed_weight
        self.alpha_threshold = alpha_threshold
        self.rollout_horizon = rollout_horizon
        self.discount = discount
        self._step_count = 0
        self._pinn_base_dt = pinn_base_dt

        # Feature history for PINN context — (pinn_horizon, 6).
        self._feature_history = feature_history.clone()

        # Action bounds.
        p = vehicle_params
        self._a_min = np.array([-p.a_max, p.sv_min])
        self._a_max = np.array([p.a_max, p.sv_max])

    def _constrain_action(self, state, action):
        """Apply steering and acceleration constraints (same as f1tenth_dynamics.py:52-64)."""
        p = self.vehicle_params
        delta, v = state[2], state[3]
        accl, sv = float(action[0]), float(action[1])

        # Steering constraint.
        if (delta <= p.s_min and sv <= 0) or (delta >= p.s_max and sv >= 0):
            sv = 0.0
        sv = max(p.sv_min, min(p.sv_max, sv))

        # Acceleration constraint.
        if v > p.v_switch:
            pos_limit = p.a_max * p.v_switch / (v + 1e-8)
        else:
            pos_limit = p.a_max
        if (v <= p.v_min and accl <= 0) or (v >= p.v_max and accl >= 0):
            accl = 0.0
        accl = max(-p.a_max, min(pos_limit, accl))

        return np.array([accl, sv])

    def _extract_features(self, state, action):
        """Extract PINN features from state + action (same formula as pinn.py).

        Features: [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd]
        """
        v = state[3]
        beta_val = state[6]
        omega = state[5]
        delta = state[2]
        accel = action[0]
        steer_vel = action[1]

        vx = v * math.cos(beta_val)
        vy = v * math.sin(beta_val)
        yaw_rate = omega
        steering_fb = delta
        throttle_cmd = accel * self._pinn_base_dt
        steering_cmd = steer_vel * self._pinn_base_dt

        return torch.tensor(
            [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd],
            dtype=torch.float32,
        )

    def step(self, action_array):
        """Advance one step using known dynamics. Updates feature history.

        :param action_array: np.array [acceleration, steering_vel]
        :return: (next_state, reward, done, info)
        """
        s = self.state

        # Apply constraints to match what known dynamics will actually use.
        constrained_action = self._constrain_action(s, action_array)

        # Update feature history with current state + constrained action.
        features = self._extract_features(s, constrained_action)
        self._feature_history = torch.cat([
            self._feature_history[1:],
            features.unsqueeze(0),
        ], dim=0)

        # Step with known dynamics.
        state_msg = VehicleStateMsg(
            x=float(s[0]), y=float(s[1]), delta=float(s[2]),
            v=float(s[3]), psi=float(s[4]), omega=float(s[5]),
            beta=float(s[6]),
        )
        cmd_msg = ControlCmdMsg(
            acceleration=float(action_array[0]),
            steering_vel=float(action_array[1]),
        )
        next_msg = self.dynamics_model.predict_next_state(state_msg, cmd_msg, self.dt)
        next_state = np.array([
            next_msg.x, next_msg.y, next_msg.delta, next_msg.v,
            next_msg.psi, next_msg.omega, next_msg.beta,
        ])

        self.state = next_state
        t = self._step_count
        self._step_count += 1

        # Cost computation (same as mcts_dpw_py/simulator.py:109-136).
        v = next_state[3]
        beta_val = next_state[6]

        # Track cost: 0.9^t * off_track * 10000
        pos_tensor = torch.tensor(
            [[next_state[0], next_state[1]]],
            dtype=torch.float32,
        )
        off_track = float(self.track_occ_map.compute_cost(pos_tensor)[0])
        track_cost = (0.9 ** t) * off_track * 10000.0

        # Speed cost: (v - v_ref_t)^2
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

        reward = -(track_cost + self.speed_weight * speed_cost + slip_cost)
        done = False

        return next_state, reward, done, {}

    def sample_action(self):
        """Sample a random action uniformly from the continuous action space."""
        return np.random.uniform(self._a_min, self._a_max)

    def copy(self):
        """Create a copy with independent state and feature history."""
        new_sim = copy.copy(self)
        new_sim.state = self.state.copy()
        new_sim._step_count = self._step_count
        new_sim._feature_history = self._feature_history.clone()
        return new_sim
