"""
Hyperparameter search for the Martin Krasser SVGP model with ARD RBF kernel using Optuna.

Tunes: sigma_y, num_inducing, inducing_init
Uses ARD RBF kernel (per-dimension length scales) instead of isotropic RBF.
Uses validation-based early stopping to prevent overfitting.
Saves the best model as npz.

Usage:
    python hp_search_svgp_ard.py --varidx 5   # omega
    python hp_search_svgp_ard.py --varidx 6   # beta
"""

import argparse
import numpy as np
import _pickle as pickle
import os
import optuna

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, value_and_grad

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from params import get_params
import config_utils

# ============================================================
# SVGP Model Functions (ARD RBF kernel)
# ============================================================

def kernel(X1, X2, theta):
    """ARD squared exponential kernel. theta = [ell_0, ..., ell_{d-1}, signal_std]."""
    ells = theta[:-1]
    sigma = theta[-1]
    X1s = X1 / ells
    X2s = X2 / ells
    sqdist = jnp.sum(X1s ** 2, 1).reshape(-1, 1) + jnp.sum(X2s ** 2, 1) - 2 * jnp.dot(X1s, X2s.T)
    return sigma ** 2 * jnp.exp(-0.5 * sqdist)


def kernel_diag(n, theta):
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

        K_mm = kernel(X_m, X_m, theta) + jitter(m)
        K_mn = kernel(X_m, X, theta)

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
        lb -= 0.5 / sigma_y ** 2 * jnp.sum(kernel_diag(n, theta))
        lb += 0.5 * jnp.trace(AAT)

        return -lb[0, 0]

    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        return np.array(value), np.array(grads)

    return nlb_grad_wrapper


def phi_opt_fn(num_kernel_params):
    """Factory returning a JIT-compiled phi_opt for the ARD kernel."""
    @jit
    def phi_opt(theta, X_m, X, y, sigma_y):
        precision = 1.0 / sigma_y ** 2
        K_mm = kernel(X_m, X_m, theta) + jitter(X_m.shape[0])
        K_mm_inv = jnp.linalg.inv(K_mm)
        K_nm = kernel(X, X_m, theta)
        K_mn = K_nm.T
        Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
        mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
        A_m = K_mm @ Sigma @ K_mm
        return mu_m, A_m, K_mm_inv
    return phi_opt


def q_fn():
    """Factory returning a JIT-compiled predictive function for the ARD kernel."""
    @jit
    def q(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
        K_ss = kernel(X_test, X_test, theta)
        K_sm = kernel(X_test, X_m, theta)
        K_ms = K_sm.T
        f_q = (K_sm @ K_mm_inv).dot(mu_m)
        f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
        return f_q, f_q_cov
    return q


# ============================================================
# Data Loading
# ============================================================

def load_all_data(varidx, seed=42):
    """Load and concatenate data for all tasks, returning scaled arrays."""
    args = get_params()
    map_name = 'YasMarina'
    SAMPLING_TIME = 0.02

    # Get train task configs
    p1 = args['bounds'][0][1]
    if args['n_initial_training_envs'] > 1:
        p2 = config_utils.latin_hypercube_sampling_maxi_min(
            n_tasks=args['n_initial_training_envs'] - 1,
            dim=args['observed_config_space_dim'],
            observed_configuration_space_interval=args['bounds']).ravel()
    else:
        p2 = np.array([])

    train_task_config = np.round(
        np.concatenate((np.array([p1]), p2.ravel()), axis=0),
        args['config_space_decimals']
    )
    train_task_config = np.sort(np.unique(train_task_config))[::-1]

    # Load per-task scalers
    xscalers = {}
    yscalers = {}
    for task in train_task_config:
        if task == 1.0489:
            base = f'./gp/data/{map_name}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}'
        else:
            base = f'./gp/data/{map_name}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}_mu_{task}'
        xscalers[f'{task}'] = pickle.load(open(os.path.join(base, 'xscaler.pkl'), 'rb'))
        yscalers[f'{task}'] = pickle.load(open(os.path.join(base, 'yscaler.pkl'), 'rb'))

    # Load splits, inverse-transform to original scale, then collect
    all_X, all_y = [], []
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for task in train_task_config:
        if task == 1.0489:
            base = f'./gp/data/{map_name}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}'
        else:
            base = f'./gp/data/{map_name}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}_mu_{task}'
        splits = np.load(os.path.join(base, 'splits.npz'))

        xs = xscalers[f'{task}']
        ys = yscalers[f'{task}']

        X_tr = xs.inverse_transform(splits['X_train'])
        y_tr = ys.inverse_transform(splits['y_train'])
        X_val = xs.inverse_transform(splits['X_val'])
        y_val = ys.inverse_transform(splits['y_val'])

        if splits['X_test'].shape[0] > 0:
            X_te = xs.inverse_transform(splits['X_test'])
            y_te = ys.inverse_transform(splits['y_test'])
        else:
            X_te = splits['X_test']
            y_te = splits['y_test']

        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        X_test_list.append(X_te)
        y_test_list.append(y_te)

        all_X.append(np.vstack([X_tr, X_val, X_te]))
        all_y.append(np.vstack([y_tr, y_val, y_te]))

    # Fit global scaler on all data
    all_X = np.vstack(all_X)
    all_y = np.vstack(all_y)
    xscaler_all = StandardScaler().fit(all_X)
    yscaler_all = StandardScaler().fit(all_y)

    # Scale
    X_train_scaled = [xscaler_all.transform(x) for x in X_train_list]
    y_train_scaled = [yscaler_all.transform(y) for y in y_train_list]
    X_val_scaled = [xscaler_all.transform(x) for x in X_val_list]
    y_val_scaled = [yscaler_all.transform(y) for y in y_val_list]
    X_test_scaled = [xscaler_all.transform(x) for x in X_test_list]
    y_test_scaled = [yscaler_all.transform(y) for y in y_test_list]

    # Concatenate
    X_train_all = jnp.array(np.vstack(X_train_scaled))
    y_train_all = jnp.array(np.vstack(y_train_scaled))
    X_val_all = jnp.array(np.vstack(X_val_scaled))
    y_val_all = jnp.array(np.vstack(y_val_scaled))

    # Eval = val + test (for final evaluation after training)
    X_eval_per_task = [np.vstack([X_val_scaled[i], X_test_scaled[i]]) for i in range(len(train_task_config))]
    y_eval_per_task = [np.vstack([y_val_scaled[i], y_test_scaled[i]]) for i in range(len(train_task_config))]

    return (X_train_all, y_train_all, X_val_all, y_val_all,
            train_task_config, X_eval_per_task, y_eval_per_task,
            xscaler_all, yscaler_all)


# ============================================================
# Optuna Objective
# ============================================================

def make_objective(X_train, y_train, X_val, y_val, d, num_kernel_params,
                   yscaler_all, seed, val_check_every=10, patience=200):
    """Create an Optuna objective that minimizes val RMSE with early stopping."""
    _phi_opt = phi_opt_fn(num_kernel_params)
    _q = q_fn()

    def objective(trial):
        sigma_y = trial.suggest_float('sigma_y', 1e-4, 1.0, log=True)
        m = trial.suggest_int('num_inducing', 1, 55)
        inducing_init = trial.suggest_categorical('inducing_init', ['kmeans', 'random_subset'])

        n = X_train.shape[0]

        # Initialize inducing points
        if inducing_init == 'kmeans':
            kmeans = KMeans(n_clusters=m, random_state=seed, n_init=10)
            kmeans.fit(np.array(X_train))
            X_m0 = jnp.array(kmeans.cluster_centers_)
        else:
            rng = np.random.RandomState(seed)
            idx_subset = rng.choice(n, size=m, replace=False)
            X_m0 = X_train[idx_subset]

        # ARD: d length scales + 1 signal std, all init to 1.0
        theta0 = jnp.ones(num_kernel_params)
        params0 = pack_params(theta0, X_m0)

        # Early stopping state
        es = {
            'iter': 0,
            'best_val_rmse': float('inf'),
            'best_params': None,
            'best_iter': 0,
            'checks_without_improvement': 0,
        }

        def callback(xk):
            es['iter'] += 1
            if es['iter'] % val_check_every != 0:
                return False
            theta_cur, X_m_cur = unpack_params(xk, d, num_kernel_params)
            mu_m_cur, A_m_cur, K_mm_inv_cur = _phi_opt(theta_cur, X_m_cur, X_train, y_train, sigma_y)
            f_val, _ = _q(X_val, theta_cur, X_m_cur, mu_m_cur, A_m_cur, K_mm_inv_cur)
            y_val_orig = yscaler_all.inverse_transform(np.array(y_val)).ravel()
            f_val_orig = yscaler_all.inverse_transform(np.array(f_val)).ravel()
            val_rmse = np.sqrt(mean_squared_error(y_val_orig, f_val_orig))
            if val_rmse < es['best_val_rmse']:
                es['best_val_rmse'] = val_rmse
                es['best_params'] = np.array(xk).copy()
                es['best_iter'] = es['iter']
                es['checks_without_improvement'] = 0
            else:
                es['checks_without_improvement'] += 1
            if es['checks_without_improvement'] >= patience:
                return True
            return False

        try:
            res = minimize(
                fun=nlb_fn(X_train, y_train, sigma_y, d, num_kernel_params),
                x0=np.array(params0),
                method='L-BFGS-B',
                jac=True,
                callback=callback,
                options={'maxiter': 30000, 'disp': False}
            )

            # Use best-val params if available
            if es['best_params'] is not None:
                val_rmse = es['best_val_rmse']
            else:
                # No callback fired, compute val RMSE from final params
                theta_f, X_m_f = unpack_params(res.x, d, num_kernel_params)
                mu_m_f, A_m_f, K_mm_inv_f = _phi_opt(theta_f, X_m_f, X_train, y_train, sigma_y)
                f_val, _ = _q(X_val, theta_f, X_m_f, mu_m_f, A_m_f, K_mm_inv_f)
                y_val_orig = yscaler_all.inverse_transform(np.array(y_val)).ravel()
                f_val_orig = yscaler_all.inverse_transform(np.array(f_val)).ravel()
                val_rmse = np.sqrt(mean_squared_error(y_val_orig, f_val_orig))

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                return float('inf')

            trial.set_user_attr('best_iter', es['best_iter'])
            trial.set_user_attr('nlb_final', float(res.fun))
            return val_rmse

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return float('inf')

    return objective


# ============================================================
# Main
# ============================================================

def save_model_and_scalers_npz(path, model_dict, xscaler, yscaler, meta=None):
    """Save model arrays, scalers, and metadata to a single .npz file."""
    import json as _json
    meta = {} if meta is None else dict(meta)

    def to_np(v):
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        try:
            return np.asarray(v)
        except Exception:
            return v

    arrays = {}
    model_json = {}
    for k, v in model_dict.items():
        vv = to_np(v)
        if isinstance(vv, np.ndarray):
            arrays[f"model__{k}"] = vv
        elif isinstance(vv, (int, float, bool)) or vv is None or isinstance(vv, str):
            model_json[k] = vv
        elif isinstance(vv, (list, tuple)):
            try:
                arrays[f"model__{k}"] = np.asarray(vv)
            except Exception:
                model_json[k] = vv
        else:
            try:
                _json.dumps(vv)
                model_json[k] = vv
            except Exception:
                pass
    arrays["model_json"] = np.asarray(_json.dumps(model_json), dtype=np.str_)

    def pack_scaler(prefix, s):
        arrays[f"{prefix}__mean"] = np.asarray(s.mean_, dtype=np.float64)
        arrays[f"{prefix}__scale"] = np.asarray(s.scale_, dtype=np.float64)
        if hasattr(s, "var_") and s.var_ is not None:
            arrays[f"{prefix}__var"] = np.asarray(s.var_, dtype=np.float64)
        arrays[f"{prefix}__n_features_in"] = np.asarray(
            int(getattr(s, "n_features_in_", len(s.mean_))), dtype=np.int32)
        arrays[f"{prefix}__with_mean"] = np.asarray(bool(getattr(s, "with_mean", True)))
        arrays[f"{prefix}__with_std"] = np.asarray(bool(getattr(s, "with_std", True)))

    pack_scaler("xscaler", xscaler)
    pack_scaler("yscaler", yscaler)
    arrays["meta_json"] = np.asarray(_json.dumps(meta), dtype=np.str_)
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description='SVGP ARD Hyperparameter Search')
    parser.add_argument('--varidx', type=int, required=True, choices=[5, 6],
                        help='Variable index: 5 (omega) or 6 (beta)')
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Number of Optuna trials (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_check_every', type=int, default=10,
                        help='Check val RMSE every N L-BFGS-B iterations')
    parser.add_argument('--patience', type=int, default=200,
                        help='Stop if no val improvement for this many checks')
    cli_args = parser.parse_args()

    varidx = cli_args.varidx
    n_trials = cli_args.n_trials
    seed = cli_args.seed
    val_check_every = cli_args.val_check_every
    patience = cli_args.patience
    var_name = 'omega' if varidx == 5 else 'beta'

    print(f"SVGP ARD HP Search | varidx={varidx} ({var_name}) | n_trials={n_trials}")
    print(f"Early stopping: val_check_every={val_check_every}, patience={patience}")
    print("=" * 60)

    # Load data
    (X_train, y_train, X_val, y_val,
     train_task_config, X_eval_per_task, y_eval_per_task,
     xscaler_all, yscaler_all) = load_all_data(varidx, seed)

    d = X_train.shape[1]
    num_kernel_params = d + 1  # ARD: d length scales + 1 signal std
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, d={d}")
    print(f"Kernel: ARD RBF ({num_kernel_params} kernel params)")
    print(f"Tasks: {train_task_config}")

    # Create study
    study_name = f'svgp_krasser_ard_var{varidx}'
    storage = f'sqlite:///hp_search_svgp_ard_var{varidx}.db'
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
    )

    objective = make_objective(
        X_train, y_train, X_val, y_val, d, num_kernel_params,
        yscaler_all, seed, val_check_every, patience
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Val RMSE: {best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    if 'best_iter' in best.user_attrs:
        print(f"  best_iter: {best.user_attrs['best_iter']}")

    # Retrain best model with early stopping and evaluate
    sigma_y = best.params['sigma_y']
    m = best.params['num_inducing']
    inducing_init = best.params['inducing_init']

    if inducing_init == 'kmeans':
        kmeans = KMeans(n_clusters=m, random_state=seed, n_init=10)
        kmeans.fit(np.array(X_train))
        X_m0 = jnp.array(kmeans.cluster_centers_)
    else:
        rng = np.random.RandomState(seed)
        X_m0 = X_train[rng.choice(X_train.shape[0], size=m, replace=False)]

    theta0 = jnp.ones(num_kernel_params)
    params0 = pack_params(theta0, X_m0)

    # Early stopping for retrain
    _phi_opt = phi_opt_fn(num_kernel_params)
    _q = q_fn()

    es_state = {
        'iter': 0,
        'best_val_rmse': float('inf'),
        'best_params': None,
        'best_iter': 0,
        'checks_without_improvement': 0,
    }

    def retrain_callback(xk):
        es_state['iter'] += 1
        it = es_state['iter']
        if it % val_check_every != 0:
            return False
        theta_cur, X_m_cur = unpack_params(xk, d, num_kernel_params)
        mu_m_cur, A_m_cur, K_mm_inv_cur = _phi_opt(theta_cur, X_m_cur, X_train, y_train, sigma_y)
        f_val, _ = _q(X_val, theta_cur, X_m_cur, mu_m_cur, A_m_cur, K_mm_inv_cur)
        y_val_orig = yscaler_all.inverse_transform(np.array(y_val)).ravel()
        f_val_orig = yscaler_all.inverse_transform(np.array(f_val)).ravel()
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, f_val_orig))
        if val_rmse < es_state['best_val_rmse']:
            es_state['best_val_rmse'] = val_rmse
            es_state['best_params'] = np.array(xk).copy()
            es_state['best_iter'] = it
            es_state['checks_without_improvement'] = 0
        else:
            es_state['checks_without_improvement'] += 1
        if it % (val_check_every * 10) == 0:
            print(f"  iter {it:>5d} | val RMSE: {val_rmse:.6f} | best: {es_state['best_val_rmse']:.6f} (iter {es_state['best_iter']})")
        if es_state['checks_without_improvement'] >= patience:
            print(f"\n  Early stopping at iter {it}. No improvement for {patience} checks.")
            print(f"  Best val RMSE: {es_state['best_val_rmse']:.6f} at iter {es_state['best_iter']}")
            return True
        return False

    res = minimize(
        fun=nlb_fn(X_train, y_train, sigma_y, d, num_kernel_params),
        x0=np.array(params0),
        method='L-BFGS-B',
        jac=True,
        callback=retrain_callback,
        options={'maxiter': 100000, 'disp': True}
    )

    # Use best-val params if available
    if es_state['best_params'] is not None:
        best_x = es_state['best_params']
        print(f"\nUsing best-val params from iter {es_state['best_iter']} (val RMSE={es_state['best_val_rmse']:.6f})")
    else:
        best_x = res.x
        print(f"\nNo val improvement tracked; using final L-BFGS-B params.")

    theta_opt, X_m_opt = unpack_params(best_x, d, num_kernel_params)
    mu_m_opt, A_m_opt, K_mm_inv = _phi_opt(theta_opt, X_m_opt, X_train, y_train, sigma_y)

    print(f"\nOptimized ARD length scales: {np.array(theta_opt[:-1]).round(4)}")
    print(f"Optimized signal_std: {float(theta_opt[-1]):.4f}")

    # Evaluate per task (on val+test combined)
    print(f"\n{'Task':<10} {'Eval RMSE (orig)':<18} {'Eval MAE (orig)':<15}")
    print("-" * 43)

    all_pred, all_true = [], []
    for i, task in enumerate(train_task_config):
        X_te = jnp.array(X_eval_per_task[i])
        y_te = np.array(y_eval_per_task[i]).ravel()

        f_te, _ = _q(X_te, theta_opt, X_m_opt, mu_m_opt, A_m_opt, K_mm_inv)
        f_te = np.array(f_te).ravel()

        y_orig = yscaler_all.inverse_transform(y_te.reshape(-1, 1)).ravel()
        f_orig = yscaler_all.inverse_transform(f_te.reshape(-1, 1)).ravel()

        rmse = np.sqrt(mean_squared_error(y_orig, f_orig))
        mae = np.mean(np.abs(y_orig - f_orig))
        all_pred.append(f_orig)
        all_true.append(y_orig)
        print(f"mu={task:<6} {rmse:<18.6f} {mae:<15.6f}")

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    rmse_all = np.sqrt(mean_squared_error(all_true, all_pred))
    mae_all = np.mean(np.abs(all_true - all_pred))
    print("-" * 43)
    print(f"{'OVERALL':<10} {rmse_all:<18.6f} {mae_all:<15.6f}")

    # Save best params
    import json
    out_path = f'best_svgp_ard_hparams_var{varidx}.json'
    best_hparams = {
        'varidx': varidx,
        'var_name': var_name,
        'sigma_y': sigma_y,
        'num_inducing': m,
        'inducing_init': inducing_init,
        'kernel_type': 'ard_rbf',
        'num_kernel_params': num_kernel_params,
        'init_length_scale': 1.0,
        'init_signal_std': 1.0,
        'optimizer': 'L-BFGS-B',
        'val_check_every': val_check_every,
        'patience': patience,
        'input_dim': d,
        'jitter': 1e-6,
        'seed': seed,
        'best_val_rmse': float(es_state['best_val_rmse']),
        'best_iter': es_state['best_iter'],
        'optimized_length_scales': np.array(theta_opt[:-1]).tolist(),
        'optimized_signal_std': float(theta_opt[-1]),
        'eval_rmse_overall': float(rmse_all),
    }
    with open(out_path, 'w') as f:
        json.dump(best_hparams, f, indent=2)
    print(f"\nBest hyperparameters saved to: {out_path}")

    # Save model + scalers as npz
    model_save = {
        'theta': theta_opt,
        'X_m': X_m_opt,
        'mu_m': mu_m_opt,
        'A_m': A_m_opt,
        'K_mm_inv': K_mm_inv,
        'use_ard': True,
        'sigma_y': float(sigma_y),
        'nlb_final': float(res.fun),
        'converged': bool(res.success),
        'best_iter': es_state['best_iter'],
        'best_val_rmse': float(es_state['best_val_rmse']),
    }

    map_name = 'YasMarina'
    SAMPLING_TIME = 0.02
    name = 'krasser_svgp_omega' if varidx == 5 else 'krasser_svgp_beta'
    os.makedirs('./gp/data/models', exist_ok=True)
    npz_path = f'./gp/data/models/{name}_{map_name}_VAR{varidx}_jax_Ts{SAMPLING_TIME:.3f}_MPPI_data_hp_ard.npz'

    save_model_and_scalers_npz(
        npz_path,
        model_dict=model_save,
        xscaler=xscaler_all,
        yscaler=yscaler_all,
        meta={
            'VARIDX': varidx,
            'map_name': map_name,
            'SAMPLING_TIME': SAMPLING_TIME,
            'num_inducing': m,
            'sigma_y': float(sigma_y),
            'train_tasks': train_task_config.tolist(),
            'val_check_every': val_check_every,
            'patience': patience,
            'best_iter': es_state['best_iter'],
            'best_val_rmse': float(es_state['best_val_rmse']),
            'eval_rmse_overall': float(rmse_all),
        }
    )
    print(f"Model saved to: {npz_path}")


if __name__ == '__main__':
    main()
