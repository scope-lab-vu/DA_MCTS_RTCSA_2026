"""
Hyperparameter search for the Martin Krasser SVGP model using Optuna.

Tunes: sigma_y, num_inducing, inducing_init
Minimizes: negative lower bound (NLB) on the combined training set.

Usage:
    python hp_search_svgp.py --varidx 5   # omega
    python hp_search_svgp.py --varidx 6   # beta
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
# SVGP Model Functions
# ============================================================

def kernel(X1, X2, theta):
    sqdist = jnp.sum(X1 ** 2, 1).reshape(-1, 1) + jnp.sum(X2 ** 2, 1) - 2 * jnp.dot(X1, X2.T)
    return theta[1] ** 2 * jnp.exp(-0.5 / theta[0] ** 2 * sqdist)


def kernel_diag(n, theta):
    return jnp.full(shape=n, fill_value=theta[1] ** 2)


def jitter(d, value=1e-6):
    return jnp.eye(d) * value


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)


def pack_params(theta, X_m):
    return jnp.concatenate([softplus_inv(theta), X_m.ravel()])


def unpack_params(params, d):
    theta = softplus(params[:2])
    X_m = params[2:].reshape(-1, d)
    return theta, X_m


def nlb_fn(X, y, sigma_y, d):
    n = X.shape[0]

    def nlb(params):
        theta, X_m = unpack_params(params, d)
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


@jit
def q(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    K_ss = kernel(X_test, X_test, theta)
    K_sm = kernel(X_test, X_m, theta)
    K_ms = K_sm.T
    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    return f_q, f_q_cov


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
        X_test_list.append(np.vstack([X_val, X_te]))
        y_test_list.append(np.vstack([y_val, y_te]))

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
    X_test_scaled = [xscaler_all.transform(x) for x in X_test_list]
    y_test_scaled = [yscaler_all.transform(y) for y in y_test_list]

    # Concatenate
    X_train_all = jnp.array(np.vstack(X_train_scaled))
    y_train_all = jnp.array(np.vstack(y_train_scaled))
    X_test_all = jnp.array(np.vstack(X_test_scaled))
    y_test_all = jnp.array(np.vstack(y_test_scaled))

    return (X_train_all, y_train_all, X_test_all, y_test_all,
            train_task_config, X_test_scaled, y_test_scaled,
            xscaler_all, yscaler_all)


# ============================================================
# Optuna Objective
# ============================================================

def make_objective(X_train, y_train, d, seed):
    """Create an Optuna objective that minimizes the NLB on training data."""

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

        theta0 = jnp.array([1.0, 1.0])
        params0 = pack_params(theta0, X_m0)

        try:
            res = minimize(
                fun=nlb_fn(X_train, y_train, sigma_y, d),
                x0=np.array(params0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'disp': False}
            )
            nlb_val = float(res.fun)

            # Check for NaN
            if np.isnan(nlb_val) or np.isinf(nlb_val):
                return float('inf')

            return nlb_val

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return float('inf')

    return objective


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SVGP Hyperparameter Search')
    parser.add_argument('--varidx', type=int, required=True, choices=[5, 6],
                        help='Variable index: 5 (omega) or 6 (beta)')
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Number of Optuna trials (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    cli_args = parser.parse_args()

    varidx = cli_args.varidx
    n_trials = cli_args.n_trials
    seed = cli_args.seed
    var_name = 'omega' if varidx == 5 else 'beta'

    print(f"SVGP HP Search | varidx={varidx} ({var_name}) | n_trials={n_trials}")
    print("=" * 60)

    # Load data
    (X_train, y_train, X_test, y_test,
     train_task_config, X_test_per_task, y_test_per_task,
     xscaler_all, yscaler_all) = load_all_data(varidx, seed)

    d = X_train.shape[1]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, d={d}")
    print(f"Tasks: {train_task_config}")

    # Create study
    study_name = f'svgp_krasser_var{varidx}'
    storage = f'sqlite:///hp_search_svgp_var{varidx}.db'
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
    )

    objective = make_objective(X_train, y_train, d, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  NLB: {best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Retrain best model and evaluate
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

    theta0 = jnp.array([1.0, 1.0])
    params0 = pack_params(theta0, X_m0)

    res = minimize(
        fun=nlb_fn(X_train, y_train, sigma_y, d),
        x0=np.array(params0),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'disp': True}
    )

    theta_opt, X_m_opt = unpack_params(res.x, d)
    mu_m_opt, A_m_opt, K_mm_inv = phi_opt(theta_opt, X_m_opt, X_train, y_train, sigma_y)

    print(f"\nOptimized kernel: length_scale={float(theta_opt[0]):.4f}, signal_std={float(theta_opt[1]):.4f}")

    # Evaluate per task
    print(f"\n{'Task':<10} {'Test RMSE (orig)':<18} {'Test MAE (orig)':<15}")
    print("-" * 43)

    all_pred, all_true = [], []
    for i, task in enumerate(train_task_config):
        X_te = jnp.array(X_test_per_task[i])
        y_te = np.array(y_test_per_task[i]).ravel()

        f_te, _ = q(X_te, theta_opt, X_m_opt, mu_m_opt, A_m_opt, K_mm_inv)
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
    out_path = f'best_svgp_hparams_var{varidx}.json'
    best_hparams = {
        'varidx': varidx,
        'var_name': var_name,
        'sigma_y': sigma_y,
        'num_inducing': m,
        'inducing_init': inducing_init,
        'kernel_type': 'isotropic_rbf',
        'init_length_scale': 1.0,
        'init_signal_std': 1.0,
        'optimizer': 'L-BFGS-B',
        'maxiter': 2000,
        'input_dim': d,
        'jitter': 1e-6,
        'seed': seed,
        'best_nlb': best.value,
        'optimized_length_scale': float(theta_opt[0]),
        'optimized_signal_std': float(theta_opt[1]),
        'test_rmse_overall': rmse_all,
    }
    with open(out_path, 'w') as f:
        json.dump(best_hparams, f, indent=2)
    print(f"\nBest hyperparameters saved to: {out_path}")


if __name__ == '__main__':
    main()
