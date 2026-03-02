"""
Compare one-step prediction RMSE:
  1. Ekin-only (Extended Kinematic baseline)
  2. Ekin + HiP-MDP SVGP (mean only) residual corrections
  3. Ekin + HiP-MDP SVGP (sampled from posterior) residual corrections
  4. PI-SVGP (Pacejka tire model from GP, dynamic bicycle model)

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

def svgp_sample(model, x_raw, task_index, rng):
    """Sample from GP posterior. x_raw: (N, 6). Returns: (N,)"""
    theta = model["theta"]
    Zx = model["Zx"]; Zh = model["Zh"]
    K_mm_inv = model["K_mm_inv"]
    alpha = K_mm_inv @ model["mu_m"]
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
    # Mean
    f_mean = (k_star @ alpha).ravel()
    # Variance: sigma_f^2 - k_star^T K_mm_inv k_star (per sample)
    # For each sample i: var_i = sigma_f^2 - k_star[i] @ K_mm_inv @ k_star[i]
    f_var = np.maximum(0.0, sigma_f_sq - np.sum(k_star * (k_star @ K_mm_inv.T), axis=1))
    # Sample
    f_sample = f_mean + rng.standard_normal(N) * np.sqrt(f_var)
    return f_sample * model["y_scale"][0] + model["y_mean"][0]

def svgp_sample_averaged(model, x_raw, task_index, n_samples=100):
    """Draw n_samples from GP posterior, return their mean. No fixed seed. x_raw: (N, 6). Returns: (N,)"""
    theta = model["theta"]
    Zx = model["Zx"]; Zh = model["Zh"]
    K_mm_inv = model["K_mm_inv"]
    alpha = K_mm_inv @ model["mu_m"]
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
    # Mean
    f_mean = (k_star @ alpha).ravel()
    # Variance
    f_var = np.maximum(0.0, sigma_f_sq - np.sum(k_star * (k_star @ K_mm_inv.T), axis=1))
    f_std = np.sqrt(f_var)
    # Draw n_samples and average (no fixed seed — truly random)
    rng = np.random.default_rng()  # seeded from OS entropy
    samples_sum = np.zeros(N)
    for _ in range(n_samples):
        samples_sum += f_mean + rng.standard_normal(N) * f_std
    f_avg = samples_sum / n_samples
    return f_avg * model["y_scale"][0] + model["y_mean"][0]

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
N_SAMPLE_RUNS = 20  # number of Monte Carlo runs for sampled SVGP
N_AVG_SAMPLES = 100  # number of samples to average per prediction
N_AVG_RUNS = 20      # number of Monte Carlo runs for averaged SVGP

print("=" * 120)
print("One-step prediction RMSE comparison (dt=0.02s)")
print("  1) Ekin-only:          Extended Kinematic model (no correction)")
print("  2) Ekin+SVGP (mean):   Ekin + HiP-MDP SVGP residual (mean only)")
print(f"  3) Ekin+SVGP (sample): Ekin + HiP-MDP SVGP residual (sampled from posterior, {N_SAMPLE_RUNS} runs)")
print(f"  4) Ekin+SVGP (avg{N_AVG_SAMPLES}):  Ekin + HiP-MDP SVGP residual (mean of {N_AVG_SAMPLES} samples, {N_AVG_RUNS} runs)")
print("  5) PI-SVGP:            Pacejka tire model from GP + dynamic bicycle model")
print("=" * 120)

for label, task_idx, c_alpha, csv_path in datasets:
    df = pd.read_csv(csv_path, comment="#")
    N = len(df) - 1

    errs = {k: [] for k in [
        "ekin_omega", "ekin_beta",
        "hip_mean_omega", "hip_mean_beta",
        "pi_omega", "pi_beta",
    ]}
    # For sampled: collect over multiple runs
    sampled_errs_omega = []  # shape will be (N_SAMPLE_RUNS, N)
    sampled_errs_beta = []

    for i in range(N):
        row = df.iloc[i]
        state = np.array([row.S_x, row.S_y, row.S_delta, row.S_vel,
                          row.S_psi, row.S_omega, row.S_beta])
        accl, sv = constrain_actions(state, row.A_a, row.A_sv)
        gt = df.iloc[i + 1]

        # 1. Ekin-only
        ekin = ekin_step(state, accl, sv, dt)

        # 2. Ekin + HiP-MDP SVGP (mean only)
        hip = ekin.copy()
        gp_input = np.array([[state[2], state[3], state[5], state[6], accl, sv]])
        hip[5] += svgp_predict(omega_model, gp_input, task_idx)[0]
        hip[6] += svgp_predict(beta_model, gp_input, task_idx)[0]

        # 3. PI-SVGP (Pacejka)
        pi = pi_svgp_step(state, accl, sv, pi_params, c_alpha, dt)

        errs["ekin_omega"].append(abs(ekin[5] - gt.S_omega))
        errs["ekin_beta"].append(abs(ekin[6] - gt.S_beta))
        errs["hip_mean_omega"].append(abs(hip[5] - gt.S_omega))
        errs["hip_mean_beta"].append(abs(hip[6] - gt.S_beta))
        errs["pi_omega"].append(abs(pi[5] - gt.S_omega))
        errs["pi_beta"].append(abs(pi[6] - gt.S_beta))

    # Run sampled predictions N_SAMPLE_RUNS times
    for run in range(N_SAMPLE_RUNS):
        rng = np.random.default_rng(seed=run)
        run_errs_omega = []
        run_errs_beta = []
        for i in range(N):
            row = df.iloc[i]
            state = np.array([row.S_x, row.S_y, row.S_delta, row.S_vel,
                              row.S_psi, row.S_omega, row.S_beta])
            accl, sv = constrain_actions(state, row.A_a, row.A_sv)
            gt = df.iloc[i + 1]

            ekin = ekin_step(state, accl, sv, dt)
            hip_s = ekin.copy()
            gp_input = np.array([[state[2], state[3], state[5], state[6], accl, sv]])
            hip_s[5] += svgp_sample(omega_model, gp_input, task_idx, rng)[0]
            hip_s[6] += svgp_sample(beta_model, gp_input, task_idx, rng)[0]

            run_errs_omega.append(abs(hip_s[5] - gt.S_omega))
            run_errs_beta.append(abs(hip_s[6] - gt.S_beta))
        sampled_errs_omega.append(run_errs_omega)
        sampled_errs_beta.append(run_errs_beta)

    # Run averaged predictions (100 samples averaged per step) over N_AVG_RUNS times
    avg_errs_omega = []  # shape: (N_AVG_RUNS, N)
    avg_errs_beta = []
    for run in range(N_AVG_RUNS):
        run_errs_omega = []
        run_errs_beta = []
        for i in range(N):
            row = df.iloc[i]
            state = np.array([row.S_x, row.S_y, row.S_delta, row.S_vel,
                              row.S_psi, row.S_omega, row.S_beta])
            accl, sv = constrain_actions(state, row.A_a, row.A_sv)
            gt = df.iloc[i + 1]

            ekin = ekin_step(state, accl, sv, dt)
            hip_a = ekin.copy()
            gp_input = np.array([[state[2], state[3], state[5], state[6], accl, sv]])
            hip_a[5] += svgp_sample_averaged(omega_model, gp_input, task_idx, N_AVG_SAMPLES)[0]
            hip_a[6] += svgp_sample_averaged(beta_model, gp_input, task_idx, N_AVG_SAMPLES)[0]

            run_errs_omega.append(abs(hip_a[5] - gt.S_omega))
            run_errs_beta.append(abs(hip_a[6] - gt.S_beta))
        avg_errs_omega.append(run_errs_omega)
        avg_errs_beta.append(run_errs_beta)

    errs = {k: np.array(v) for k, v in errs.items()}
    sampled_errs_omega = np.array(sampled_errs_omega)  # (N_SAMPLE_RUNS, N)
    sampled_errs_beta = np.array(sampled_errs_beta)
    avg_errs_omega = np.array(avg_errs_omega)  # (N_AVG_RUNS, N)
    avg_errs_beta = np.array(avg_errs_beta)

    # Per-run RMSE for sampled
    rmse_per_run_omega = np.sqrt(np.mean(sampled_errs_omega**2, axis=1))
    rmse_per_run_beta = np.sqrt(np.mean(sampled_errs_beta**2, axis=1))
    mae_per_run_omega = np.mean(sampled_errs_omega, axis=1)
    mae_per_run_beta = np.mean(sampled_errs_beta, axis=1)
    max_per_run_omega = np.max(sampled_errs_omega, axis=1)
    max_per_run_beta = np.max(sampled_errs_beta, axis=1)

    # Per-run RMSE for averaged
    avg_rmse_per_run_omega = np.sqrt(np.mean(avg_errs_omega**2, axis=1))
    avg_rmse_per_run_beta = np.sqrt(np.mean(avg_errs_beta**2, axis=1))
    avg_mae_per_run_omega = np.mean(avg_errs_omega, axis=1)
    avg_mae_per_run_beta = np.mean(avg_errs_beta, axis=1)
    avg_max_per_run_omega = np.max(avg_errs_omega, axis=1)
    avg_max_per_run_beta = np.max(avg_errs_beta, axis=1)

    print(f"\n{'─' * 120}")
    print(f"  {label}  (task_index={task_idx}, c_alpha={c_alpha}, N={N} steps)")
    print(f"{'─' * 120}")
    print(f"                            {'RMSE':>12}  {'MAE':>12}  {'MAX':>12}")
    for var in ["omega", "beta"]:
        print(f"  {var.upper():5s}:")
        for model_name, key in [("Ekin-only      ", f"ekin_{var}"),
                                 ("Ekin+SVGP mean ", f"hip_mean_{var}"),
                                 ("PI-SVGP        ", f"pi_{var}")]:
            e = errs[key]
            print(f"    {model_name}:  {np.sqrt(np.mean(e**2)):12.6f}  {np.mean(e):12.6f}  {np.max(e):12.6f}")

        # Sampled stats (single sample)
        if var == "omega":
            rmse_arr, mae_arr, max_arr = rmse_per_run_omega, mae_per_run_omega, max_per_run_omega
        else:
            rmse_arr, mae_arr, max_arr = rmse_per_run_beta, mae_per_run_beta, max_per_run_beta
        print(f"    Ekin+SVGP sample:  {np.mean(rmse_arr):12.6f}  {np.mean(mae_arr):12.6f}  {np.mean(max_arr):12.6f}  "
              f"(±RMSE: {np.std(rmse_arr):.6f}, range: {np.min(rmse_arr):.6f}–{np.max(rmse_arr):.6f})")

        # Averaged stats (100 samples averaged)
        if var == "omega":
            a_rmse, a_mae, a_max = avg_rmse_per_run_omega, avg_mae_per_run_omega, avg_max_per_run_omega
        else:
            a_rmse, a_mae, a_max = avg_rmse_per_run_beta, avg_mae_per_run_beta, avg_max_per_run_beta
        print(f"    Ekin+SVGP avg{N_AVG_SAMPLES}:  {np.mean(a_rmse):12.6f}  {np.mean(a_mae):12.6f}  {np.mean(a_max):12.6f}  "
              f"(±RMSE: {np.std(a_rmse):.6f}, range: {np.min(a_rmse):.6f}–{np.max(a_rmse):.6f})")

print(f"\n{'=' * 120}")
