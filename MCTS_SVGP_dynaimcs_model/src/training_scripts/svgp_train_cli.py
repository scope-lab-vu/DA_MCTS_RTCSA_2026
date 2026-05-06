"""
Train an SVGP (Krasser, ARD RBF kernel, JAX) on one or more split directories.

Single source of truth for the training pipeline used by the existing notebooks
and the HP-search CLI. Reads pre-split data from --task-dir(s), refits a global
StandardScaler across all tasks, runs L-BFGS-B with val-based early stopping,
and saves the final model (.npz + .pkl) plus per-task RMSE.

Examples
--------
# Train psi (VAR4) on stanley single-mu, 6-D no_c_alpha, 12 inducing points
python svgp_train_cli.py \\
    --varidx 4 --feature-mode no_c_alpha \\
    --task-dir ../../data/splits/YasMarina_VAR4_Ts0.100_stanley_mu_1.0489 \\
    --num-inducing 12 --sigma-y 1e-4 \\
    --out-name krasser_svgp_psi_stanley

# Train omega (VAR5) on canonical 3-task data, 7-D c_alpha_smooth, 11 inducing points
python svgp_train_cli.py \\
    --varidx 5 --feature-mode c_alpha_smooth \\
    --task-dir ../../data/splits/YasMarina_VAR5_Ts0.020 \\
                ../../data/splits/YasMarina_VAR5_Ts0.020_mu_0.8642 \\
                ../../data/splits/YasMarina_VAR5_Ts0.020_mu_0.4947 \\
    --num-inducing 11 --sigma-y 1e-4 \\
    --out-name krasser_svgp_omega_canonical
"""

import argparse
import json
import os
import sys
import _pickle as pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, value_and_grad

from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

Y_NAMES = ["Y_x", "Y_y", "Y_delta", "Y_vel", "Y_psi", "Y_omega", "Y_beta"]
C_ALPHA_COL = 6  # column index in the 7-D split feature matrix


# ============================================================
# SVGP Model functions (ARD RBF)
# ============================================================
def kernel_ard(X1, X2, theta):
    ells, sigma = theta[:-1], theta[-1]
    X1s, X2s = X1 / ells, X2 / ells
    sqdist = jnp.sum(X1s ** 2, 1).reshape(-1, 1) + jnp.sum(X2s ** 2, 1) - 2 * jnp.dot(X1s, X2s.T)
    return sigma ** 2 * jnp.exp(-0.5 * sqdist)


def kernel_ard_diag(n, theta):
    return jnp.full(shape=n, fill_value=theta[-1] ** 2)


def jitter(d, value=1e-6):
    return jnp.eye(d) * value


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)


def pack_params(theta, X_m):
    return jnp.concatenate([softplus_inv(theta), X_m.ravel()])


def unpack_params(params, d, num_kernel_params):
    theta = softplus(params[:num_kernel_params])
    X_m = params[num_kernel_params:].reshape(-1, d)
    return theta, X_m


def nlb_fn(X, y, sigma_y, d, num_kernel_params):
    n = X.shape[0]

    def nlb(params):
        theta, X_m = unpack_params(params, d, num_kernel_params)
        m = X_m.shape[0]
        K_mm = kernel_ard(X_m, X_m, theta) + jitter(m)
        K_mn = kernel_ard(X_m, X, theta)
        L = jnp.linalg.cholesky(K_mm)
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y
        AAT = A @ A.T
        B = jnp.eye(m) + AAT
        LB = jnp.linalg.cholesky(B)
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y
        lb = -n / 2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n / 2 * jnp.log(sigma_y ** 2)
        lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5 / sigma_y ** 2 * jnp.sum(kernel_ard_diag(n, theta))
        lb += 0.5 * jnp.trace(AAT)
        return -lb[0, 0]

    nlb_grad = jit(value_and_grad(nlb))

    def wrapper(params):
        v, g = nlb_grad(params)
        return np.array(v), np.array(g)

    return wrapper


@jit
def phi_opt(theta, X_m, X, y, sigma_y):
    precision = 1.0 / sigma_y ** 2
    K_mm = kernel_ard(X_m, X_m, theta) + jitter(X_m.shape[0])
    K_mm_inv = jnp.linalg.inv(K_mm)
    K_nm = kernel_ard(X, X_m, theta)
    K_mn = K_nm.T
    Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
    mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
    A_m = K_mm @ Sigma @ K_mm
    return mu_m, A_m, K_mm_inv


@jit
def q_pred(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    K_ss = kernel_ard(X_test, X_test, theta)
    K_sm = kernel_ard(X_test, X_m, theta)
    K_ms = K_sm.T
    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    return f_q, f_q_cov


# ============================================================
# Data loading
# ============================================================
def load_task_dirs(task_dirs: List[str], feature_mode: str):
    """
    Load X/y for each task dir, inverse-transform with the per-task scalers,
    drop C_alpha column if requested, then refit a global scaler across all tasks.

    Returns dict with concatenated train/val and per-task eval (val+test) lists.
    """
    per_task_raw = []  # list of (X_train, y_train, X_val, y_val, X_test, y_test) in original scale
    all_X, all_y = [], []
    for d in task_dirs:
        d = Path(d)
        splits = np.load(d / "splits.npz")
        with open(d / "xscaler.pkl", "rb") as f:
            xs = pickle.load(f)
        with open(d / "yscaler.pkl", "rb") as f:
            ys = pickle.load(f)

        Xtr = xs.inverse_transform(splits["X_train"])
        ytr = ys.inverse_transform(splits["y_train"])
        Xv = xs.inverse_transform(splits["X_val"])
        yv = ys.inverse_transform(splits["y_val"])
        if splits["X_test"].shape[0] > 0:
            Xte = xs.inverse_transform(splits["X_test"])
            yte = ys.inverse_transform(splits["y_test"])
        else:
            Xte = splits["X_test"]
            yte = splits["y_test"]

        if feature_mode == "no_c_alpha":
            Xtr = np.delete(Xtr, C_ALPHA_COL, axis=1)
            if Xte.shape[0] > 0:
                Xte = np.delete(Xte, C_ALPHA_COL, axis=1)
            Xv = np.delete(Xv, C_ALPHA_COL, axis=1)

        per_task_raw.append((Xtr, ytr, Xv, yv, Xte, yte))
        all_X.append(np.vstack([Xtr, Xv, Xte]))
        all_y.append(np.vstack([ytr, yv, yte]))

    all_X = np.vstack(all_X)
    all_y = np.vstack(all_y)
    xscaler_all = StandardScaler().fit(all_X)
    yscaler_all = StandardScaler().fit(all_y)

    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    eval_X_list, eval_y_list = [], []  # val + test merged for final per-task eval
    for (Xtr, ytr, Xv, yv, Xte, yte) in per_task_raw:
        train_X_list.append(xscaler_all.transform(Xtr))
        train_y_list.append(yscaler_all.transform(ytr))
        val_X_list.append(xscaler_all.transform(Xv))
        val_y_list.append(yscaler_all.transform(yv))
        Xte_s = xscaler_all.transform(Xte) if Xte.shape[0] > 0 else Xte
        yte_s = yscaler_all.transform(yte) if yte.shape[0] > 0 else yte
        if Xte_s.shape[0] > 0:
            eval_X_list.append(np.vstack([val_X_list[-1], Xte_s]))
            eval_y_list.append(np.vstack([val_y_list[-1], yte_s]))
        else:
            eval_X_list.append(val_X_list[-1])
            eval_y_list.append(val_y_list[-1])

    return {
        "X_train_all": jnp.array(np.vstack(train_X_list)),
        "y_train_all": jnp.array(np.vstack(train_y_list)),
        "X_val_all": jnp.array(np.vstack(val_X_list)),
        "y_val_all": jnp.array(np.vstack(val_y_list)),
        "eval_X_per_task": eval_X_list,
        "eval_y_per_task": eval_y_list,
        "xscaler_all": xscaler_all,
        "yscaler_all": yscaler_all,
        "input_dim": train_X_list[0].shape[1],
        "task_names": [Path(d).name for d in task_dirs],
    }


# ============================================================
# Training (used by both this CLI and hp_search_svgp_ard_cli)
# ============================================================
def train_one(X_train, y_train, X_val, y_val, yscaler_all,
              num_inducing, sigma_y, inducing_init, seed,
              maxiter=100000, val_check_every=10, patience=200, verbose=True):
    n, d = X_train.shape[0], X_train.shape[1]
    num_kernel_params = d + 1  # ARD: d length scales + signal std

    # Init inducing
    if inducing_init == "kmeans":
        km = KMeans(n_clusters=num_inducing, random_state=seed, n_init=10)
        km.fit(np.array(X_train))
        X_m0 = jnp.array(km.cluster_centers_)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=num_inducing, replace=False)
        X_m0 = X_train[idx]

    theta0 = jnp.ones(num_kernel_params)
    params0 = pack_params(theta0, X_m0)

    es = {"iter": 0, "best_val_rmse": float("inf"), "best_params": None,
          "best_iter": 0, "checks": 0, "history": []}

    def cb(xk):
        es["iter"] += 1
        if es["iter"] % val_check_every != 0:
            return False
        theta_cur, X_m_cur = unpack_params(xk, d, num_kernel_params)
        mu_m, A_m, K_mm_inv = phi_opt(theta_cur, X_m_cur, X_train, y_train, sigma_y)
        f_val, _ = q_pred(X_val, theta_cur, X_m_cur, mu_m, A_m, K_mm_inv)
        y_orig = yscaler_all.inverse_transform(np.array(y_val)).ravel()
        f_orig = yscaler_all.inverse_transform(np.array(f_val)).ravel()
        rmse = float(np.sqrt(mean_squared_error(y_orig, f_orig)))
        es["history"].append((es["iter"], rmse))
        if rmse < es["best_val_rmse"]:
            es["best_val_rmse"] = rmse
            es["best_params"] = np.array(xk).copy()
            es["best_iter"] = es["iter"]
            es["checks"] = 0
        else:
            es["checks"] += 1
        if verbose and es["iter"] % (val_check_every * 10) == 0:
            print(f"  iter {es['iter']:>5d} | val RMSE: {rmse:.6f} | best: {es['best_val_rmse']:.6f} (iter {es['best_iter']})")
        return es["checks"] >= patience

    res = minimize(
        fun=nlb_fn(X_train, y_train, sigma_y, d, num_kernel_params),
        x0=np.array(params0),
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"maxiter": maxiter, "disp": verbose},
    )

    best_x = es["best_params"] if es["best_params"] is not None else res.x
    theta_opt, X_m_opt = unpack_params(best_x, d, num_kernel_params)
    mu_m_opt, A_m_opt, K_mm_inv = phi_opt(theta_opt, X_m_opt, X_train, y_train, sigma_y)

    return {
        "theta": theta_opt, "X_m": X_m_opt, "mu_m": mu_m_opt, "A_m": A_m_opt,
        "K_mm_inv": K_mm_inv, "use_ard": True,
        "best_iter": es["best_iter"], "best_val_rmse": es["best_val_rmse"],
        "history": es["history"], "res": res, "sigma_y": float(sigma_y),
    }


def per_task_eval(model, eval_X_per_task, eval_y_per_task, yscaler_all, task_names):
    rows = []
    all_pred, all_true = [], []
    for X_te, y_te, name in zip(eval_X_per_task, eval_y_per_task, task_names):
        if X_te.shape[0] == 0:
            continue
        f, _ = q_pred(jnp.array(X_te), model["theta"], model["X_m"],
                      model["mu_m"], model["A_m"], model["K_mm_inv"])
        f = np.array(f).ravel()
        y_te_arr = np.array(y_te).ravel()
        rmse_s = float(np.sqrt(mean_squared_error(y_te_arr, f)))
        y_o = yscaler_all.inverse_transform(y_te_arr.reshape(-1, 1)).ravel()
        f_o = yscaler_all.inverse_transform(f.reshape(-1, 1)).ravel()
        rmse_o = float(np.sqrt(mean_squared_error(y_o, f_o)))
        mae_o = float(mean_absolute_error(y_o, f_o))
        rows.append({"task": name, "rmse_scaled": rmse_s, "rmse_orig": rmse_o, "mae_orig": mae_o})
        all_pred.append(f_o)
        all_true.append(y_o)
    if all_pred:
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        overall_rmse = float(np.sqrt(mean_squared_error(all_true, all_pred)))
        overall_mae = float(mean_absolute_error(all_true, all_pred))
    else:
        overall_rmse = overall_mae = float("nan")
    return rows, overall_rmse, overall_mae


# ============================================================
# Save
# ============================================================
def save_npz(path, model, xscaler, yscaler, meta):
    arrays = {}
    for k in ("theta", "X_m", "mu_m", "A_m", "K_mm_inv"):
        arrays[f"model__{k}"] = np.asarray(model[k])
    model_json = {k: model[k] for k in ("use_ard", "best_iter", "best_val_rmse", "sigma_y")}
    model_json["nlb_final"] = float(model["res"].fun)
    model_json["converged"] = bool(model["res"].success)
    arrays["model_json"] = np.asarray(json.dumps(model_json), dtype=np.str_)
    arrays["xscaler__mean"] = np.asarray(xscaler.mean_, dtype=np.float64)
    arrays["xscaler__scale"] = np.asarray(xscaler.scale_, dtype=np.float64)
    arrays["xscaler__var"] = np.asarray(xscaler.var_, dtype=np.float64)
    arrays["xscaler__n_features_in"] = np.asarray(int(xscaler.n_features_in_), dtype=np.int32)
    arrays["xscaler__with_mean"] = np.asarray(True)
    arrays["xscaler__with_std"] = np.asarray(True)
    arrays["yscaler__mean"] = np.asarray(yscaler.mean_, dtype=np.float64)
    arrays["yscaler__scale"] = np.asarray(yscaler.scale_, dtype=np.float64)
    arrays["yscaler__var"] = np.asarray(yscaler.var_, dtype=np.float64)
    arrays["yscaler__n_features_in"] = np.asarray(int(yscaler.n_features_in_), dtype=np.int32)
    arrays["yscaler__with_mean"] = np.asarray(True)
    arrays["yscaler__with_std"] = np.asarray(True)
    arrays["meta_json"] = np.asarray(json.dumps(meta), dtype=np.str_)
    np.savez_compressed(path, **arrays)


def save_pkl(path, model, xscaler, yscaler):
    save = {k: v for k, v in model.items() if k != "res"}
    save["nlb_final"] = float(model["res"].fun)
    save["converged"] = bool(model["res"].success)
    with open(path, "wb") as f:
        pickle.dump((save, xscaler, yscaler), f)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--varidx", type=int, required=True, choices=range(7),
                    help="Target index (0..6); only y_target depends on this")
    ap.add_argument("--feature-mode", choices=["c_alpha_smooth", "no_c_alpha"], required=True,
                    help="7-D (with C_alpha_smooth) or 6-D (without)")
    ap.add_argument("--task-dir", nargs="+", required=True, help="One or more split dirs")
    ap.add_argument("--num-inducing", type=int, required=True)
    ap.add_argument("--sigma-y", type=float, required=True, help="Observation noise std (fixed)")
    ap.add_argument("--inducing-init", choices=["kmeans", "random_subset"], default="random_subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maxiter", type=int, default=100000)
    ap.add_argument("--val-check-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--out-models-dir", default="../../data/models")
    ap.add_argument("--out-name", required=True, help="Output basename, e.g. krasser_svgp_psi_stanley")
    args = ap.parse_args()

    print(f"varidx={args.varidx} ({Y_NAMES[args.varidx]}), feature_mode={args.feature_mode}")
    print(f"task_dirs: {args.task_dir}")

    data = load_task_dirs(args.task_dir, args.feature_mode)
    print(f"Combined: train={data['X_train_all'].shape}, val={data['X_val_all'].shape}, "
          f"input_dim={data['input_dim']}")

    model = train_one(
        data["X_train_all"], data["y_train_all"],
        data["X_val_all"], data["y_val_all"],
        data["yscaler_all"],
        num_inducing=args.num_inducing, sigma_y=args.sigma_y,
        inducing_init=args.inducing_init, seed=args.seed,
        maxiter=args.maxiter, val_check_every=args.val_check_every, patience=args.patience,
    )
    print(f"\nBest val RMSE: {model['best_val_rmse']:.6f} at iter {model['best_iter']}")

    rows, overall_rmse, overall_mae = per_task_eval(
        model, data["eval_X_per_task"], data["eval_y_per_task"],
        data["yscaler_all"], data["task_names"],
    )
    print(f"\n{'Task':<60} {'RMSE (orig)':<15} {'MAE (orig)':<15}")
    print("-" * 90)
    for r in rows:
        print(f"{r['task']:<60} {r['rmse_orig']:<15.6f} {r['mae_orig']:<15.6f}")
    print("-" * 90)
    print(f"{'OVERALL':<60} {overall_rmse:<15.6f} {overall_mae:<15.6f}")

    out_dir = Path(args.out_models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_M_{args.num_inducing}"
    npz_path = out_dir / f"{args.out_name}{suffix}.npz"
    pkl_path = out_dir / f"{args.out_name}{suffix}.pkl"

    meta = {
        "varidx": args.varidx, "y_name": Y_NAMES[args.varidx],
        "feature_mode": args.feature_mode,
        "task_dirs": [str(Path(d).resolve()) for d in args.task_dir],
        "task_names": data["task_names"],
        "num_inducing": args.num_inducing, "sigma_y": float(args.sigma_y),
        "inducing_init": args.inducing_init, "seed": args.seed,
        "input_dim": data["input_dim"],
        "best_iter": model["best_iter"], "best_val_rmse": model["best_val_rmse"],
        "eval_overall_rmse": overall_rmse, "eval_overall_mae": overall_mae,
        "per_task_eval": rows,
    }
    save_npz(str(npz_path), model, data["xscaler_all"], data["yscaler_all"], meta)
    save_pkl(str(pkl_path), model, data["xscaler_all"], data["yscaler_all"])
    print(f"\nSaved: {npz_path}")
    print(f"Saved: {pkl_path}")


if __name__ == "__main__":
    main()
