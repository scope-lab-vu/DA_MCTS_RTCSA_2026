"""
Batched PINN evaluator for MCTS leaf rollout evaluation.

Runs B*K parallel random rollouts through the PINN dynamics model on GPU,
computing discounted cumulative rewards.  K rollouts per leaf are averaged
to reduce variance, at essentially zero additional GPU cost since GPU
latency dominates over batch size scaling.
"""

import logging
import math

import numpy as np
import torch

log = logging.getLogger(__name__)


class BatchedPINNEvaluator:
    """GPU-accelerated batched rollout evaluator using PINN dynamics.

    Given B leaf states and their feature histories, runs B*K parallel random
    rollouts through the PINN (K per leaf), then averages to produce one
    value per leaf.  GPU processes batch sizes 1-64 at constant latency,
    so K>1 provides variance reduction at no extra cost.

    :param pinn_model: PINNModel instance (holds the neural network + scaler)
    :param vehicle_params: VehicleParams
    :param track_occ_map: TrackOccupancyMap (grid moved to device at init)
    :param v_ref: fallback reference velocity
    :param speed_weight: weight on speed cost term
    :param alpha_threshold: slip angle penalty threshold
    :param rollout_horizon: max steps per rollout
    :param discount: discount factor
    :param dt: simulation timestep
    :param device: torch device string ("cuda" or "cpu")
    :param K: number of parallel rollouts per leaf for variance reduction
    """

    def __init__(
        self,
        pinn_model,
        vehicle_params,
        track_occ_map,
        v_ref,
        speed_weight,
        alpha_threshold,
        rollout_horizon,
        discount,
        dt,
        device,
        K=4,
    ):
        self.pinn = pinn_model
        self.vp = vehicle_params
        self.v_ref = v_ref
        self.speed_weight = speed_weight
        self.alpha_threshold = alpha_threshold
        self.rollout_horizon = rollout_horizon
        self.discount = discount
        self.dt = dt
        self.device = torch.device(device)
        self.K = K  # rollouts per leaf

        # PINN sub-stepping (same logic as pinn.py:63-66).
        self._base_dt = 0.02
        self._n_substeps = max(1, round(dt / self._base_dt))

        # Move occupancy map grid to evaluation device.
        self._occ_grid = track_occ_map._map_torch.to(self.device)
        self._occ_x_min = track_occ_map.x_min
        self._occ_y_min = track_occ_map.y_min
        self._occ_cell_size = track_occ_map.cell_size
        self._occ_nx = track_occ_map.nx
        self._occ_ny = track_occ_map.ny

        # Move PINN scaler to device.
        self._scaler_mean = pinn_model._scaler_mean.to(self.device)
        self._scaler_scale = pinn_model._scaler_scale.to(self.device)

        # Move PINN model to device if not already there.
        if next(pinn_model._pinn.parameters()).device != self.device:
            pinn_model._pinn.to(self.device)
            log.info("Moved PINN model to %s for batched evaluation", self.device)

        self._is_rnn = pinn_model._is_rnn

        # Action bounds as tensors on device.
        self._a_min = torch.tensor(
            [-vehicle_params.a_max, vehicle_params.sv_min],
            device=self.device, dtype=torch.float32,
        )
        self._a_max = torch.tensor(
            [vehicle_params.a_max, vehicle_params.sv_max],
            device=self.device, dtype=torch.float32,
        )

        log.info(
            "BatchedPINNEvaluator: device=%s, substeps=%d, horizon=%d, "
            "discount=%.2f, K=%d rollouts/leaf",
            self.device, self._n_substeps, rollout_horizon, discount, K,
        )

    # -- Occupancy map cost (GPU) -------------------------------------------

    def _compute_track_cost(self, positions):
        """Batched off-track lookup. positions: (N, 2) → (N,) float {0, 1}."""
        ix = ((positions[:, 0] - self._occ_x_min) / self._occ_cell_size).long()
        iy = ((positions[:, 1] - self._occ_y_min) / self._occ_cell_size).long()
        oob = (ix < 0) | (ix >= self._occ_nx) | (iy < 0) | (iy >= self._occ_ny)
        ix = torch.clamp(ix, 0, self._occ_nx - 1)
        iy = torch.clamp(iy, 0, self._occ_ny - 1)
        costs = self._occ_grid[ix, iy]
        costs = torch.where(oob, torch.ones_like(costs), costs)
        return costs

    # -- Constraint helpers (same as pinn.py:329-354) -----------------------

    def _accl_constraints(self, vel, accl):
        p = self.vp
        pos_limit = torch.where(
            vel > p.v_switch,
            p.a_max * p.v_switch / (vel + 1e-8),
            torch.full_like(vel, p.a_max),
        )
        result = accl.clone()
        at_min = (vel <= p.v_min) & (accl <= 0)
        at_max = (vel >= p.v_max) & (accl >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(accl), result)
        result = torch.maximum(result, torch.full_like(result, -p.a_max))
        result = torch.minimum(result, pos_limit)
        return result

    def _steering_constraints(self, delta, sv):
        p = self.vp
        result = sv.clone()
        at_min = (delta <= p.s_min) & (sv <= 0)
        at_max = (delta >= p.s_max) & (sv >= 0)
        result = torch.where(at_min | at_max, torch.zeros_like(sv), result)
        result = torch.clamp(result, p.sv_min, p.sv_max)
        return result

    # -- Feature extraction (same formula as pinn.py:155-177) ---------------

    def _extract_features(self, states, actions):
        """(N, 7) state + (N, 2) action → (N, 6) features."""
        v = states[:, 3]
        beta = states[:, 6]
        omega = states[:, 5]
        delta = states[:, 2]
        accel = actions[:, 0]
        steer_vel = actions[:, 1]

        vx = v * torch.cos(beta)
        vy = v * torch.sin(beta)
        yaw_rate = omega
        steering_fb = delta
        throttle_cmd = accel * self._base_dt
        steering_cmd = steer_vel * self._base_dt

        return torch.stack(
            [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd], dim=1,
        )

    # -- PINN forward pass --------------------------------------------------

    def _pinn_predict(self, histories):
        """Run PINN on (N, pinn_horizon, 6) → (omega, beta) each (N,)."""
        features_norm = (histories - self._scaler_mean) / self._scaler_scale
        with torch.no_grad():
            if self._is_rnn:
                N = histories.shape[0]
                h = self.pinn._pinn.init_hidden(N)
                if h.device != self.device:
                    h = h.to(self.device)
                output, _, _ = self.pinn._pinn(histories, features_norm, h)
            else:
                output, _, _ = self.pinn._pinn(histories, features_norm)

        omega = output[:, 2]
        beta = torch.atan2(output[:, 1], output[:, 0] + 1e-8)
        return omega, beta

    # -- Main evaluation entry point ----------------------------------------

    def evaluate_batch(self, states_list, feature_histories_list, v_ref_window=None):
        """Evaluate B leaf nodes with K batched PINN rollouts each.

        Each leaf gets K independent random rollouts.  The K values are
        averaged to produce one value per leaf (variance reduction).

        :param states_list: list of B numpy arrays, each (7,)
        :param feature_histories_list: list of B tensors, each (pinn_horizon, 6)
        :param v_ref_window: numpy array of per-step reference velocities, or None
        :return: numpy array (B,) of averaged cumulative discounted rewards
        """
        B = len(states_list)
        if B == 0:
            return np.array([], dtype=np.float32)

        K = self.K
        N = B * K  # total parallel rollouts

        # Expand each leaf state K times: (B,7) → (N,7)
        states_np = np.array(states_list)  # (B, 7)
        states_expanded = np.repeat(states_np, K, axis=0)  # (N, 7)
        states = torch.tensor(
            states_expanded, device=self.device, dtype=torch.float32,
        )

        # Expand histories: (B, H, 6) → (N, H, 6)
        hist_stack = torch.stack(feature_histories_list)  # (B, H, 6)
        histories = hist_stack.repeat_interleave(K, dim=0).to(self.device)  # (N, H, 6)

        # Reference velocity on device.
        if v_ref_window is not None:
            v_ref_t = torch.tensor(
                v_ref_window, device=self.device, dtype=torch.float32,
            )
        else:
            v_ref_t = None

        total_rewards = torch.zeros(N, device=self.device)

        for t in range(self.rollout_horizon):
            # Sample random actions: uniform in [a_min, a_max].
            rand = torch.rand(N, 2, device=self.device)
            actions = self._a_min + rand * (self._a_max - self._a_min)

            # Apply constraints.
            u_accl = self._accl_constraints(states[:, 3], actions[:, 0])
            u_sv = self._steering_constraints(states[:, 2], actions[:, 1])
            constrained_actions = torch.stack([u_accl, u_sv], dim=1)

            # Run n_substeps of PINN integration.
            cur_x = states[:, 0]
            cur_y = states[:, 1]
            cur_delta = states[:, 2]
            cur_v = states[:, 3]
            cur_psi = states[:, 4]
            cur_omega = states[:, 5]
            cur_beta = states[:, 6]

            for _ in range(self._n_substeps):
                current_state = torch.stack([
                    cur_x, cur_y, cur_delta, cur_v, cur_psi, cur_omega, cur_beta,
                ], dim=1)

                new_features = self._extract_features(
                    current_state, constrained_actions,
                )
                histories = torch.cat([
                    histories[:, 1:, :],
                    new_features.unsqueeze(1),
                ], dim=1)

                omega_pred, beta_pred = self._pinn_predict(histories)

                cur_x = cur_x + self._base_dt * cur_v * torch.cos(beta_pred + cur_psi)
                cur_y = cur_y + self._base_dt * cur_v * torch.sin(beta_pred + cur_psi)
                cur_delta = cur_delta + self._base_dt * u_sv
                cur_v = cur_v + self._base_dt * u_accl
                cur_psi = cur_psi + self._base_dt * omega_pred
                cur_omega = omega_pred
                cur_beta = beta_pred

            # Normalize heading.
            cur_psi = torch.remainder(cur_psi, 2 * math.pi)

            # Update states.
            states = torch.stack([
                cur_x, cur_y, cur_delta, cur_v, cur_psi, cur_omega, cur_beta,
            ], dim=1)

            # Compute reward.
            positions = states[:, :2]
            off_track = self._compute_track_cost(positions)
            track_cost = (0.9 ** t) * off_track * 10000.0

            if v_ref_t is not None and t < len(v_ref_t):
                ref_v = v_ref_t[t]
            else:
                ref_v = self.v_ref
            speed_cost = (states[:, 3] - ref_v) ** 2

            v_x = states[:, 3] * torch.cos(states[:, 6])
            v_y = states[:, 3] * torch.sin(states[:, 6])
            alpha = -torch.atan2(v_y, torch.abs(v_x) + 1e-8)
            slip_cost = alpha ** 2 + 10000.0 * (torch.abs(alpha) > self.alpha_threshold).float()

            reward = -(track_cost + self.speed_weight * speed_cost + slip_cost)
            total_rewards += (self.discount ** t) * reward

        # Average K rollouts per leaf: (N,) → reshape (B, K) → mean → (B,)
        per_leaf = total_rewards.reshape(B, K).mean(dim=1)
        return per_leaf.cpu().numpy()
