"""
Optuna hyperparameter search for the Krasser SVGP (ARD RBF) on any dataset.

Tunes: sigma_y (log-uniform), num_inducing (int), inducing_init (categorical).
After the search finishes, retrains the final model with the best params and
saves it via svgp_train_cli's save functions.

Reuses data loading and training functions from svgp_train_cli.py.

Output:
  - Optuna study DB:  hp_search_<out_name>.db (sqlite)
  - Best hparams:     best_<out_name>.json
  - Final model:      <out-models-dir>/<out_name>_M_<best_M>.{npz,pkl}

Examples
--------
# VAR4 (psi) on stanley, 6-D no_c_alpha
python hp_search_svgp_ard_cli.py \
    --varidx 4 --feature-mode no_c_alpha \
    --task-dir ../../data/splits/YasMarina_VAR4_Ts0.100_stanley_mu_1.0489 \
    --n-trials 200 \
    --out-name krasser_svgp_psi_stanley

python hp_search_svgp_ard_cli.py \
    --varidx 4 --feature-mode c_alpha_smooth \
    --task-dir ../../data/splits/YasMarina_VAR4_Ts0.100_stanley_mu_1.0489 \
    --n-trials 200 \
    --out-name krasser_svgp_psi_stanley
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

# Reuse all the SVGP infrastructure from svgp_train_cli
from svgp_train_cli import (
    Y_NAMES, load_task_dirs, train_one, per_task_eval, save_npz, save_pkl,
)


def make_objective(data, seed, m_min, m_max, sigma_low, sigma_high,
                   maxiter, val_check_every, patience):
    """Optuna objective: minimize val RMSE for the suggested (sigma_y, M, init)."""
    def objective(trial):
        sigma_y = trial.suggest_float("sigma_y", sigma_low, sigma_high, log=True)
        m = trial.suggest_int("num_inducing", m_min, m_max)
        init = trial.suggest_categorical("inducing_init", ["kmeans", "random_subset"])

        try:
            model = train_one(
                data["X_train_all"], data["y_train_all"],
                data["X_val_all"], data["y_val_all"],
                data["yscaler_all"],
                num_inducing=m, sigma_y=sigma_y, inducing_init=init, seed=seed,
                maxiter=maxiter, val_check_every=val_check_every, patience=patience,
                verbose=False,
            )
            rmse = model["best_val_rmse"]
            if rmse is None or np.isnan(rmse) or np.isinf(rmse):
                return float("inf")
            return rmse
        except Exception as e:
            print(f"  trial failed: {type(e).__name__}: {e}")
            return float("inf")

    return objective


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--varidx", type=int, required=True, choices=range(7))
    ap.add_argument("--feature-mode", choices=["c_alpha_smooth", "no_c_alpha"], required=True)
    ap.add_argument("--task-dir", nargs="+", required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--m-min", type=int, default=1)
    ap.add_argument("--m-max", type=int, default=15)
    ap.add_argument("--sigma-low", type=float, default=1e-4)
    ap.add_argument("--sigma-high", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-maxiter", type=int, default=30000,
                    help="L-BFGS-B maxiter inside each Optuna trial")
    ap.add_argument("--final-maxiter", type=int, default=100000,
                    help="L-BFGS-B maxiter for the final retrain at best params")
    ap.add_argument("--val-check-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--out-models-dir", default="../../data/models")
    ap.add_argument("--out-name", required=True, help="Output basename, e.g. krasser_svgp_psi_stanley")
    ap.add_argument("--study-storage-dir", default=".",
                    help="Where to save the Optuna sqlite DB and best-hparams JSON")
    args = ap.parse_args()

    print(f"varidx={args.varidx} ({Y_NAMES[args.varidx]}), feature_mode={args.feature_mode}")
    print(f"task_dirs: {args.task_dir}")

    data = load_task_dirs(args.task_dir, args.feature_mode)
    print(f"Combined: train={data['X_train_all'].shape}, val={data['X_val_all'].shape}, "
          f"input_dim={data['input_dim']}")

    storage_dir = Path(args.study_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"hp_search_{args.out_name}.db"
    study_name = args.out_name
    storage_url = f"sqlite:///{db_path}"

    print(f"\n=== Optuna study: {study_name} (storage={db_path}) ===")
    study = optuna.create_study(
        study_name=study_name, storage=storage_url, load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(data, args.seed,
                       args.m_min, args.m_max, args.sigma_low, args.sigma_high,
                       args.search_maxiter, args.val_check_every, args.patience),
        n_trials=args.n_trials, show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: val RMSE = {best.value:.6f}")
    print(f"  sigma_y      = {best.params['sigma_y']:.6e}")
    print(f"  num_inducing = {best.params['num_inducing']}")
    print(f"  inducing_init= {best.params['inducing_init']}")

    # Persist a JSON of best hparams (CLI-readable)
    best_json = {
        "varidx": args.varidx, "y_name": Y_NAMES[args.varidx],
        "feature_mode": args.feature_mode,
        "task_dirs": [str(Path(d).resolve()) for d in args.task_dir],
        "n_trials": args.n_trials,
        "best_val_rmse": float(best.value),
        "best_trial": best.number,
        "params": {
            "sigma_y": float(best.params["sigma_y"]),
            "num_inducing": int(best.params["num_inducing"]),
            "inducing_init": best.params["inducing_init"],
        },
        "search_space": {
            "sigma_y": [args.sigma_low, args.sigma_high],
            "num_inducing": [args.m_min, args.m_max],
            "inducing_init": ["kmeans", "random_subset"],
        },
    }
    json_path = storage_dir / f"best_{args.out_name}.json"
    with open(json_path, "w") as f:
        json.dump(best_json, f, indent=2)
    print(f"Best hparams JSON: {json_path}")

    # Retrain final model with best params at higher maxiter
    print(f"\n=== Retraining final model (maxiter={args.final_maxiter}) ===")
    model = train_one(
        data["X_train_all"], data["y_train_all"],
        data["X_val_all"], data["y_val_all"],
        data["yscaler_all"],
        num_inducing=int(best.params["num_inducing"]),
        sigma_y=float(best.params["sigma_y"]),
        inducing_init=best.params["inducing_init"],
        seed=args.seed,
        maxiter=args.final_maxiter,
        val_check_every=args.val_check_every,
        patience=args.patience,
        verbose=True,
    )
    print(f"\nFinal val RMSE: {model['best_val_rmse']:.6f} at iter {model['best_iter']}")

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
    M = int(best.params["num_inducing"])
    npz_path = out_dir / f"{args.out_name}_M_{M}.npz"
    pkl_path = out_dir / f"{args.out_name}_M_{M}.pkl"

    meta = {
        "varidx": args.varidx, "y_name": Y_NAMES[args.varidx],
        "feature_mode": args.feature_mode,
        "task_dirs": [str(Path(d).resolve()) for d in args.task_dir],
        "task_names": data["task_names"],
        "num_inducing": M, "sigma_y": float(best.params["sigma_y"]),
        "inducing_init": best.params["inducing_init"], "seed": args.seed,
        "input_dim": data["input_dim"],
        "best_iter": model["best_iter"], "best_val_rmse": model["best_val_rmse"],
        "eval_overall_rmse": overall_rmse, "eval_overall_mae": overall_mae,
        "per_task_eval": rows,
        "hp_search": {"n_trials": args.n_trials, "best_trial": best.number, "study_db": str(db_path)},
    }
    save_npz(str(npz_path), model, data["xscaler_all"], data["yscaler_all"], meta)
    save_pkl(str(pkl_path), model, data["xscaler_all"], data["yscaler_all"])
    print(f"\nSaved: {npz_path}")
    print(f"Saved: {pkl_path}")


if __name__ == "__main__":
    main()
