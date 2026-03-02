"""
Compare one-step predictions: known dynamics vs GP dynamics vs gym ground truth.
"""

import os, sys, json, glob
import numpy as np
import torch
import pandas as pd
import math

DTYPE = torch.float64
JITTER = 1e-6

# Vehicle params
mu = 1.0489; C_Sf = 4.718; C_Sr = 5.4562
lf = 0.15875; lr = 0.17145; h = 0.074
mass = 3.74; I_z = 0.04712
G = 9.81; L = lf + lr

def known_dynamics_step(state, accl, sv, dt=0.04):
    x, y, delta, v, psi, omega, beta = state
    if (delta <= -0.4189 and sv <= 0) or (delta >= 0.4189 and sv >= 0):
        sv = 0.0
    sv = np.clip(sv, -3.2, 3.2)
    pos_limit = 9.51 * 7.319 / v if v > 7.319 else 9.51
    if (v <= -5.0 and accl <= 0) or (v >= 20.0 and accl >= 0):
        accl = 0.0
    accl = np.clip(accl, -9.51, pos_limit)
    if abs(v) < 0.5:
        f_x = v * np.cos(psi); f_y = v * np.sin(psi); f_delta = sv; f_v = accl
        f_psi = v / L * np.tan(delta)
        f_omega = accl / L * np.tan(delta) + v / (L * np.cos(delta)**2) * sv
        f_beta = 0.0
    else:
        f_x = v * np.cos(beta + psi); f_y = v * np.sin(beta + psi); f_delta = sv; f_v = accl; f_psi = omega
        v_safe = max(abs(v), 0.5)
        f_omega = (-mu * mass / (v_safe * I_z * L) * (lf**2 * C_Sf * (G*lr - accl*h) + lr**2 * C_Sr * (G*lf + accl*h)) * omega
                   + mu * mass / (I_z * L) * (lr * C_Sr * (G*lf + accl*h) - lf * C_Sf * (G*lr - accl*h)) * beta
                   + mu * mass / (I_z * L) * lf * C_Sf * (G*lr - accl*h) * delta)
        f_beta = ((mu / (v_safe**2 * L) * (C_Sr * (G*lf + accl*h) * lr - C_Sf * (G*lr - accl*h) * lf) - 1.0) * omega
                  - mu / (v_safe * L) * (C_Sr * (G*lf + accl*h) + C_Sf * (G*lr - accl*h)) * beta
                  + mu / (v_safe * L) * C_Sf * (G*lr - accl*h) * delta)
    return np.array([x + dt*f_x, y + dt*f_y, delta + dt*f_delta, v + dt*f_v,
                     psi + dt*f_psi, omega + dt*f_omega, beta + dt*f_beta])


def gp_predict_mean(gp_input_raw, gp_params):
    """GP posterior mean prediction -> Pacejka params (no sampling)."""
    ls, sigma, Z, alpha_packed, S_diag, xmean, xscale, output_bias, K_mm_inv, M, D_input, num_outputs = gp_params
    x_norm = (gp_input_raw - xmean) / xscale
    sigma_sq = sigma * sigma
    k_star = np.zeros(M)
    for j in range(M):
        dist = np.sum(((x_norm - Z[j]) / ls) ** 2)
        k_star[j] = sigma_sq * np.exp(-0.5 * dist)
    f_mean = k_star @ alpha_packed
    raw = f_mean + output_bias
    pacejka = np.zeros(num_outputs)
    for i in range(num_outputs):
        if i in {0, 1, 2, 4, 5, 6}:
            pacejka[i] = np.log(1 + np.exp(raw[i])) if raw[i] <= 20 else raw[i]
        else:
            pacejka[i] = raw[i]
    return pacejka


def gp_dynamics_step(state, accl, sv, gp_params, dt=0.04, base_ts=0.02):
    """GP dynamics — one step with substeps, using mean (no sampling)."""
    state = np.array(state, dtype=np.float64)
    n_substeps = max(1, round(dt / base_ts))
    sub_dt = base_ts if n_substeps > 1 else dt
    delta_init = state[2]; v_init = state[3]
    if (delta_init <= -0.4189 and sv <= 0) or (delta_init >= 0.4189 and sv >= 0):
        sv = 0.0
    sv = np.clip(sv, -3.2, 3.2)
    pos_limit = 9.51 * 7.319 / (v_init + 1e-8) if v_init > 7.319 else 9.51
    if (v_init <= -5.0 and accl <= 0) or (v_init >= 20.0 and accl >= 0):
        accl = 0.0
    accl = np.clip(accl, -9.51, pos_limit)

    ls, sigma, Z, alpha_packed, S_diag, xmean, xscale, output_bias, K_mm_inv, M, D_input, num_outputs = gp_params

    all_pacejka = []
    for sub in range(n_substeps):
        x, y, delta, v, psi, omega, beta = state
        VX = v * np.cos(beta); VY = v * np.sin(beta)
        YAW_RATE = omega; STEERING_FB = delta
        THROTTLE_CMD = accl * base_ts; STEERING_CMD = sv * base_ts
        c_alpha = 104.89
        gp_input_raw = np.array([VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD, c_alpha])
        pacejka = gp_predict_mean(gp_input_raw, gp_params)
        all_pacejka.append(pacejka)
        Bf, Cf, Df, Ef, Br, Cr, Dr, Er, Frx = pacejka
        steering = STEERING_FB + STEERING_CMD
        alphaf = steering - np.arctan2(lf * YAW_RATE + VY, abs(VX) + 1e-8)
        alphar = np.arctan2(lr * YAW_RATE - VY, abs(VX) + 1e-8)
        Bf_alphaf = Bf * alphaf; Br_alphar = Br * alphar
        Ffy = Df * np.sin(Cf * np.arctan(Bf_alphaf - Ef * (Bf_alphaf - np.arctan(Bf_alphaf))))
        Fry = Dr * np.sin(Cr * np.arctan(Br_alphar - Er * (Br_alphar - np.arctan(Br_alphar))))
        dVX = (1.0/mass) * (Frx - Ffy * np.sin(steering)) + VY * YAW_RATE
        dVY = (1.0/mass) * (Fry + Ffy * np.cos(steering)) - VX * YAW_RATE
        dYAW_RATE = (1.0/I_z) * (Ffy * lf * np.cos(steering) - Fry * lr)
        next_VX = VX + dVX * sub_dt; next_VY = VY + dVY * sub_dt
        next_YAW_RATE_val = YAW_RATE + dYAW_RATE * sub_dt
        omega_pred = next_YAW_RATE_val
        beta_pred = np.arctan2(next_VY, next_VX + 1e-8)
        next_x = x + sub_dt * v * np.cos(beta + psi)
        next_y = y + sub_dt * v * np.sin(beta + psi)
        next_delta = delta + sub_dt * sv
        next_v = v + sub_dt * accl
        next_psi = psi + sub_dt * omega
        v_from_gp = np.sqrt(next_VX**2 + next_VY**2)
        state = np.array([next_x, next_y, next_delta, next_v, next_psi, omega_pred, beta_pred])

    return state, all_pacejka, v_from_gp, next_v


# Load GP model
model_dir = os.path.join(os.path.dirname(__file__), "..", "deep_pacejka_model_f1tenth",
                         "deep_dynamics", "output", "pi_svgp_no_hip", "run4_low_kl")
ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
scalers = np.load(os.path.join(model_dir, "scalers.npz"))
sd = ckpt["state_dict"]
M = ckpt["M"]; D_input = ckpt["input_dim"]; num_outputs = ckpt["num_outputs"]
ls = np.exp(sd["log_lengthscales"].numpy())
sigma = np.exp(sd["log_outputscale"].numpy())
Z = sd["X_m"].numpy()
m_var = sd["m"].numpy()
K_mm_inv = ckpt["K_mm_inv"].numpy()
alpha_packed = K_mm_inv @ m_var
S_diag = np.exp(2 * sd["m_logstd"].numpy())
xmean = scalers["x_mean"]; xscale = scalers["x_scale"]
output_bias = sd["output_bias"].numpy()
gp_params = (ls, sigma, Z, alpha_packed, S_diag, xmean, xscale, output_bias, K_mm_inv, M, D_input, num_outputs)

# Load trajectory
traj_files = sorted(glob.glob("results/traj_*.csv"))
df = pd.read_csv(traj_files[-1], comment="#")
print(f"Trajectory: {traj_files[-1]} ({len(df)} steps)")
print()

# Compare at every 5th step
test_steps = list(range(21, min(len(df)-1, len(df)), 5))

print("=" * 100)
print("Single-step comparison: Known Dynamics vs GP Dynamics vs Ground Truth (gym)")
print("=" * 100)

omega_errors_known = []
omega_errors_gp = []
beta_errors_known = []
beta_errors_gp = []

for step_i in test_steps:
    row = df.iloc[step_i]
    state = [row.S_x, row.S_y, row.S_delta, row.S_vel, row.S_psi, row.S_omega, row.S_beta]
    accl = row.A_a; sv = row.A_sv
    gt = df.iloc[step_i + 1]

    known = known_dynamics_step(state, accl, sv)
    gp, pacejka_list, v_gp, v_kin = gp_dynamics_step(state, accl, sv, gp_params)

    # Errors vs ground truth
    omega_err_kn = abs(known[5] - gt.S_omega)
    omega_err_gp = abs(gp[5] - gt.S_omega)
    beta_err_kn = abs(known[6] - gt.S_beta)
    beta_err_gp = abs(gp[6] - gt.S_beta)

    omega_errors_known.append(omega_err_kn)
    omega_errors_gp.append(omega_err_gp)
    beta_errors_known.append(beta_err_kn)
    beta_errors_gp.append(beta_err_gp)

    if step_i % 10 == 1 or step_i >= len(df) - 15:
        print(f"\n--- Step {step_i} (v={state[3]:.2f}, beta={state[6]:.4f}, omega={state[5]:.4f}, delta={state[2]:.4f}) ---")
        print(f"  Action: accl={accl:.3f}, sv={sv:.3f}")
        pac = pacejka_list[0]
        print(f"  Pacejka: Bf={pac[0]:.2f} Cf={pac[1]:.2f} Df={pac[2]:.2f} Ef={pac[3]:.2f} Br={pac[4]:.2f} Cr={pac[5]:.2f} Dr={pac[6]:.2f} Er={pac[7]:.2f} Frx={pac[8]:.2f}")
        print(f"  omega: ground_truth={gt.S_omega:9.4f}  known={known[5]:9.4f}  gp={gp[5]:9.4f}  | err_kn={omega_err_kn:.4f}  err_gp={omega_err_gp:.4f}")
        print(f"  beta:  ground_truth={gt.S_beta:9.4f}  known={known[6]:9.4f}  gp={gp[6]:9.4f}  | err_kn={beta_err_kn:.4f}  err_gp={beta_err_gp:.4f}")
        print(f"  v:     ground_truth={gt.S_vel:9.4f}  known={known[3]:9.4f}  gp={gp[3]:9.4f}  v_gp_sqrt={v_gp:.4f}  v_diff={abs(v_gp-v_kin):.4f}")

print("\n" + "=" * 100)
print("SUMMARY: Mean absolute errors vs ground truth")
print("=" * 100)
print(f"  omega MAE: known={np.mean(omega_errors_known):.4f}  gp={np.mean(omega_errors_gp):.4f}")
print(f"  beta  MAE: known={np.mean(beta_errors_known):.4f}  gp={np.mean(beta_errors_gp):.4f}")
print(f"  omega MAX: known={np.max(omega_errors_known):.4f}  gp={np.max(omega_errors_gp):.4f}")
print(f"  beta  MAX: known={np.max(beta_errors_known):.4f}  gp={np.max(beta_errors_gp):.4f}")
