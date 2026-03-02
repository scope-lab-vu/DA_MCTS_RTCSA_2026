"""
Compare one-step prediction RMSE:
  1. Ekin-only (Extended Kinematic baseline)
  2. Ekin + HiP-MDP SVGP residuals (omega & beta corrections)
  3. PI-SVGP (Pacejka tire model from GP, dynamic bicycle model)

All evaluated on trajectory data at dt=0.02s.
"""

import os
import numpy as np
import pandas as pd
import torch

# Vehicle params
mu_friction = 1.0489; C_Sf = 4.718; C_Sr = 5.4562
lf = 0.15875; lr = 0.17145; h = 0.074
mass = 3.74; I_z = 0.04712
G = 9.81; L = lf + lr

# ── HiP-MDP SVGP (Ekin + residual) ──────────────────────────────────────────

def load_svgp_model(path):
    d = np.load(path, allow_pickle=True)
    return {
        "theta": d["model__theta_opt"],
        "Zx": d["model__Zx_opt"],
        "Zh": d["model__Zh_opt"],
        "K_mm_inv": d["model__K_mm_inv"],
        "mu_m": d["model__mu_m_opt"],
        "h_mu_opt": d["model__h_mu_opt"],
        "x_mean": d["xscaler__mean"],
        "x_scale": d["xscaler__scale"],
        "y_mean": d["yscaler__mean"],
        "y_scale": d["yscaler__scale"],
    }

def svgp_predict(model, x_raw, task_index):
    """Batch GP mean prediction. x_raw: (N, 6). Returns: (N,)"""
    theta = model["theta"]
    Zx = model["Zx"]; Zh = model["Zh"]
    alpha = model["K_mm_inv"] @ model["mu_m"]
    D_aug = Zx.shape[1] + Zh.shape[1]
    lengthscales = theta[:D_aug]
    sigma_f_sq = theta[D_aug] ** 2
    h_mu = model["h_mu_opt"][task_index]
    x_norm = (x_raw - model["x_mean"]) / model["x_scale"]
    N = x_norm.shape[0]; M = Zx.shape[0]
    x_aug = np.hstack([x_norm, np.tile(h_mu, (N, 1))])
    Z_aug = np.hstack([Zx, Zh])
    k_star = np.zeros((N, M))
    for j in range(M):
        diff = (x_aug - Z_aug[j]) / lengthscales
        k_star[:, j] = sigma_f_sq * np.exp(-0.5 * np.sum(diff ** 2, axis=1))
    f_mean = (k_star @ alpha).ravel()
    return f_mean * model["y_scale"][0] + model["y_mean"][0]

def ekin_step(state, accl, sv, sub_dt):
    """Extended Kinematic model — one substep."""
    x, y, delta, v, psi, omega, beta = state
    return np.array([
        x + sub_dt * v * np.cos(beta + psi),
        y + sub_dt * v * np.sin(beta + psi),
        delta + sub_dt * sv,
        v + sub_dt * accl,
        psi + sub_dt * omega,
        omega + sub_dt * (sv * v + delta * accl) / L,
        beta + sub_dt * (lf * (sv * v + delta * accl)) / L,
    ])

# ── PI-SVGP (Pacejka GP) ────────────────────────────────────────────────────

def load_pi_svgp_model(model_dir):
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
    xmean = scalers["x_mean"]; xscale = scalers["x_scale"]
    output_bias = sd["output_bias"].numpy()
    return (ls, sigma, Z, alpha_packed, xmean, xscale, output_bias, M, D_input, num_outputs)

def pi_svgp_predict_pacejka(gp_input_raw, pi_params):
    """Predict 9 Pacejka params from 7D input."""
    ls, sigma, Z, alpha_packed, xmean, xscale, output_bias, M, D_input, num_outputs = pi_params
    x_norm = (gp_input_raw - xmean) / xscale
    sigma_sq = sigma * sigma
    k_star = np.zeros(M)
    for j in range(M):
        dist = np.sum(((x_norm - Z[j]) / ls) ** 2)
        k_star[j] = sigma_sq * np.exp(-0.5 * dist)
    raw = k_star @ alpha_packed + output_bias
    pacejka = np.zeros(num_outputs)
    for i in range(num_outputs):
        if i in {0, 1, 2, 4, 5, 6}:  # softplus for Bf,Cf,Df,Br,Cr,Dr
            pacejka[i] = np.log(1 + np.exp(raw[i])) if raw[i] <= 20 else raw[i]
        else:
            pacejka[i] = raw[i]
    return pacejka

def pi_svgp_step(state, accl, sv, pi_params, c_alpha, sub_dt):
    """PI-SVGP Pacejka dynamics — one substep."""
    x, y, delta, v, psi, omega, beta = state
    VX = v * np.cos(beta); VY = v * np.sin(beta)
    YAW_RATE = omega; STEERING_FB = delta
    THROTTLE_CMD = accl * sub_dt; STEERING_CMD = sv * sub_dt

    gp_input = np.array([VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD, c_alpha])
    Bf, Cf, Df, Ef, Br, Cr, Dr, Er, Frx = pi_svgp_predict_pacejka(gp_input, pi_params)

    steering = STEERING_FB + STEERING_CMD
    alphaf = steering - np.arctan2(lf * YAW_RATE + VY, abs(VX) + 1e-8)
    alphar = np.arctan2(lr * YAW_RATE - VY, abs(VX) + 1e-8)
    Bf_af = Bf * alphaf; Br_ar = Br * alphar
    Ffy = Df * np.sin(Cf * np.arctan(Bf_af - Ef * (Bf_af - np.arctan(Bf_af))))
    Fry = Dr * np.sin(Cr * np.arctan(Br_ar - Er * (Br_ar - np.arctan(Br_ar))))
    dVX = (1.0/mass) * (Frx - Ffy * np.sin(steering)) + VY * YAW_RATE
    dVY = (1.0/mass) * (Fry + Ffy * np.cos(steering)) - VX * YAW_RATE
    dYAW_RATE = (1.0/I_z) * (Ffy * lf * np.cos(steering) - Fry * lr)

    next_VX = VX + dVX * sub_dt; next_VY = VY + dVY * sub_dt
    omega_pred = YAW_RATE + dYAW_RATE * sub_dt
    beta_pred = np.arctan2(next_VY, next_VX + 1e-8)

    return np.array([
        x + sub_dt * v * np.cos(beta + psi),
        y + sub_dt * v * np.sin(beta + psi),
        delta + sub_dt * sv,
        v + sub_dt * accl,
        psi + sub_dt * omega,
        omega_pred,
        beta_pred,
    ])

# ── Constrain actions ────────────────────────────────────────────────────────

def constrain_actions(state, accl, sv):
    delta_init = state[2]; v_init = state[3]
    if (delta_init <= -0.4189 and sv <= 0.0) or (delta_init >= 0.4189 and sv >= 0.0):
        sv = 0.0
    sv = np.clip(sv, -3.2, 3.2)
    pos_limit = 9.51 * 7.319 / (v_init + 1e-8) if v_init > 7.319 else 9.51
    if (v_init <= -5.0 and accl <= 0.0) or (v_init >= 20.0 and accl >= 0.0):
        accl = 0.0
    accl = np.clip(accl, -9.51, pos_limit)
    return accl, sv

# ── Main ─────────────────────────────────────────────────────────────────────

# Load HiP-MDP SVGP models
omega_model = load_svgp_model("models/svgp/hip_mdp_svgp_omega_YasMarina_VAR5_jax_Ts0.020_MPPI_data.npz")
beta_model = load_svgp_model("models/svgp/hip_mdp_svgp_beta_YasMarina_VAR6_jax_Ts0.020_MPPI_data.npz")

# Load PI-SVGP model
pi_svgp_dir = os.path.join(os.path.dirname(__file__), "..", "deep_pacejka_model_f1tenth",
                           "deep_dynamics", "output", "pi_svgp_no_hip", "run4_low_kl")
pi_params = load_pi_svgp_model(pi_svgp_dir)

datasets = [
    ("mu=1.0489", 0, 104.89, "results/mcts_pi_svgp_cs_YasMarina_10489_lw1.1/traj_config_mcts_known_dyn_action_prior_cpp_t1_s42_YasMarina_10489_20260301_222248.csv"),
    ("mu=0.8642", 1, 86.42, "results/mcts_pi_svgp_cs_YasMarina_08642_lw1.1/traj_config_mcts_known_dyn_action_prior_cpp_t1_s42_YasMarina_08642_20260301_222349.csv"),
    ("mu=0.4947", 2, 49.47, "results/mcts_pi_svgp_cs_YasMarina_04947_lw1.1/traj_config_mcts_known_dyn_action_prior_cpp_t1_s42_YasMarina_04947_20260301_222451.csv"),
]

dt = 0.02  # trajectory timestep = SVGP base timestep

print("=" * 100)
print("One-step prediction RMSE comparison (dt=0.02s)")
print("  1) Ekin-only:       Extended Kinematic model (no correction)")
print("  2) Ekin+HiP-SVGP:   Ekin + HiP-MDP SVGP residual corrections")
print("  3) PI-SVGP:         Pacejka tire model from GP + dynamic bicycle model")
print("=" * 100)

for label, task_idx, c_alpha, csv_path in datasets:
    df = pd.read_csv(csv_path, comment="#")
    N = len(df) - 1

    errs = {k: [] for k in [
        "ekin_omega", "ekin_beta",
        "hip_omega", "hip_beta",
        "pi_omega", "pi_beta",
    ]}

    for i in range(N):
        row = df.iloc[i]
        state = np.array([row.S_x, row.S_y, row.S_delta, row.S_vel,
                          row.S_psi, row.S_omega, row.S_beta])
        accl, sv = constrain_actions(state, row.A_a, row.A_sv)
        gt = df.iloc[i + 1]

        # 1. Ekin-only
        ekin = ekin_step(state, accl, sv, dt)

        # 2. Ekin + HiP-MDP SVGP
        hip = ekin.copy()
        gp_input = np.array([[state[2], state[3], state[5], state[6], accl, sv]])
        hip[5] += svgp_predict(omega_model, gp_input, task_idx)[0]
        hip[6] += svgp_predict(beta_model, gp_input, task_idx)[0]

        # 3. PI-SVGP (Pacejka)
        pi = pi_svgp_step(state, accl, sv, pi_params, c_alpha, dt)

        errs["ekin_omega"].append(abs(ekin[5] - gt.S_omega))
        errs["ekin_beta"].append(abs(ekin[6] - gt.S_beta))
        errs["hip_omega"].append(abs(hip[5] - gt.S_omega))
        errs["hip_beta"].append(abs(hip[6] - gt.S_beta))
        errs["pi_omega"].append(abs(pi[5] - gt.S_omega))
        errs["pi_beta"].append(abs(pi[6] - gt.S_beta))

    errs = {k: np.array(v) for k, v in errs.items()}

    print(f"\n{'─' * 100}")
    print(f"  {label}  (task_index={task_idx}, c_alpha={c_alpha}, N={N} steps)")
    print(f"{'─' * 100}")
    print(f"                       {'RMSE':>12}  {'MAE':>12}  {'MAX':>12}")
    for var in ["omega", "beta"]:
        print(f"  {var.upper():5s}:")
        for model_name, key in [("Ekin-only   ", f"ekin_{var}"),
                                 ("Ekin+HiP-SVGP", f"hip_{var}"),
                                 ("PI-SVGP     ", f"pi_{var}")]:
            e = errs[key]
            print(f"    {model_name}:  {np.sqrt(np.mean(e**2)):12.6f}  {np.mean(e):12.6f}  {np.max(e):12.6f}")

print(f"\n{'=' * 100}")
