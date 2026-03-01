#!/usr/bin/env python3
"""
Hyperparameter search for SGPR (Krasserm VFE) action prior using Optuna.

Tunes: num_inducing, epochs, lr, initial length scale, output scale, noise.
Objective: minimise sum of negative ELBO (accel + steer) on a validation set.
After search, retrains best HPs on train+val and evaluates on held-out test.

Data split: train (64%) / val (16%) / test (20%)

Usage:
    python MCTS_SVGP_action_sampler/hp_search_krasserm.py \\
        --data-dir MCTS_SVGP_action_sampler/training_data \\
        --feature-set no_avg_kappa --n-trials 100

    # Run all three feature sets:
    for fs in no_avg_kappa no_lateral general; do
        python MCTS_SVGP_action_sampler/hp_search_krasserm.py \\
            --data-dir MCTS_SVGP_action_sampler/training_data \\
            --feature-set $fs --n-trials 100
    done
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_krasserm import (
    DTYPE,
    FEATURE_SETS,
    OUTPUT_COLS,
    OUTPUT_NAMES,
    SparseGPR,
    evaluate,
    get_input_features,
    load_data,
    save_model,
    select_inducing_points,
    train_sgpr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("hp_search_krasserm")


def compute_val_neg_elbo(models, X_val, Y_val) -> float:
    """Sum of neg ELBO on validation set across all outputs (lower is better)."""
    total = 0.0
    for i, m in enumerate(models):
        with torch.no_grad():
            neg_lb = m.neg_elbo(X_val, Y_val[:, i])
        total += neg_lb.item()
    return total


def objective(
    trial: optuna.Trial,
    data_splits: dict,
    args: argparse.Namespace,
) -> float:
    """Single Optuna trial: train 2 independent SGPR GPs, return val neg ELBO."""

    # ── Sample hyperparameters ────────────────────────────────────────────
    num_inducing = trial.suggest_int("num_inducing", 1, 150, step=10)
    epochs = trial.suggest_int("epochs", 1, 100, step=10)
    lr = trial.suggest_float("lr", 0.01, 1.0, log=True)
    init_lengthscale = trial.suggest_float("init_lengthscale", 0.1, 10.0, log=True)
    init_outputscale = trial.suggest_float("init_outputscale", 0.1, 10.0, log=True)
    init_noise = trial.suggest_float("init_noise", 1e-4, 1.0, log=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train = data_splits["X_train"]
    Y_train = data_splits["Y_train"]
    X_val = data_splits["X_val"]
    Y_val = data_splits["Y_val"]
    input_dim = data_splits["input_dim"]

    # ── Inducing points ───────────────────────────────────────────────────
    n_ind = min(num_inducing, len(X_train))
    X_m_init = select_inducing_points(X_train.numpy(), n_ind)

    # ── Train 2 independent GPs ──────────────────────────────────────────
    models = []
    for i in range(2):
        model = SparseGPR(
            X_m_init=X_m_init.clone(),
            input_dim=input_dim,
            init_lengthscale=init_lengthscale,
            init_outputscale=init_outputscale,
            init_noise=init_noise,
        )
        try:
            train_sgpr(model, X_train, Y_train[:, i],
                       epochs=epochs, lr=lr, verbose=False)
        except Exception as e:
            log.warning("Trial %d failed: %s", trial.number, e)
            return float("inf")

        model.cache_posterior(X_train, Y_train[:, i])
        models.append(model)

    # ── Evaluate on validation set ────────────────────────────────────────
    val_neg_elbo = compute_val_neg_elbo(models, X_val, Y_val)

    trial.set_user_attr("val_neg_elbo", val_neg_elbo)
    log.info(
        "Trial %d: val_neg_ELBO=%.4f  (M=%d, ep=%d, lr=%.4f, "
        "ls=%.3f, os=%.3f, noise=%.5f)",
        trial.number, val_neg_elbo,
        num_inducing, epochs, lr,
        init_lengthscale, init_outputscale, init_noise,
    )

    return val_neg_elbo


def parse_args():
    ap = argparse.ArgumentParser(
        description="Hyperparameter search for SGPR (Krasserm VFE)")
    ap.add_argument("--data-dir", default="results/",
                    help="Dir with trajectory CSVs")
    ap.add_argument("--output-dir", default="models/sgpr_krasserm/",
                    help="Dir to save best model")
    ap.add_argument("--n-trials", type=int, default=50,
                    help="Number of Optuna trials")
    ap.add_argument("--test-split", type=float, default=0.2,
                    help="Fraction for held-out test set")
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="Fraction of remaining data for validation")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--study-name", default="sgpr_krasserm",
                    help="Optuna study name")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL (default: in-memory)")
    ap.add_argument("--feature-set", default="no_avg_kappa",
                    choices=list(FEATURE_SETS.keys()),
                    help="Input feature set (default: no_avg_kappa)")
    return ap.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_cols, input_names = get_input_features(args.feature_set)
    log.info("Feature set '%s' (%dD): %s", args.feature_set, len(input_cols), input_cols)

    # Append feature set name to output dir and study name for separation
    if not args.output_dir.rstrip("/").endswith(args.feature_set):
        args.output_dir = os.path.join(args.output_dir.rstrip("/") + f"_{args.feature_set}")
    if not args.study_name.endswith(args.feature_set):
        args.study_name = f"{args.study_name}_{args.feature_set}"

    # ── Load data ─────────────────────────────────────────────────────────
    data = load_data(args.data_dir, input_cols)

    X_all = data[input_cols].values.astype(np.float64)
    Y_all = data[OUTPUT_COLS].values.astype(np.float64)
    log.info("Loaded %d rows total", len(X_all))

    # ── Split: train / val / test ─────────────────────────────────────────
    X_trainval, X_test_np, Y_trainval, Y_test_np = train_test_split(
        X_all, Y_all, test_size=args.test_split, random_state=args.seed)
    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X_trainval, Y_trainval, test_size=args.val_split, random_state=args.seed)
    log.info("Split: train=%d  val=%d  test=%d",
             len(X_train_np), len(X_val_np), len(X_test_np))

    # ── Normalise (fit on train only) ─────────────────────────────────────
    x_scaler = StandardScaler().fit(X_train_np)
    y_scaler = StandardScaler().fit(Y_train_np)

    X_train = torch.tensor(x_scaler.transform(X_train_np), dtype=DTYPE)
    X_val = torch.tensor(x_scaler.transform(X_val_np), dtype=DTYPE)
    Y_train = torch.tensor(y_scaler.transform(Y_train_np), dtype=DTYPE)
    Y_val = torch.tensor(y_scaler.transform(Y_val_np), dtype=DTYPE)

    data_splits = {
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "input_dim": len(input_cols),
    }

    # ── Run Optuna study ──────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    t0 = time.monotonic()
    study.optimize(
        lambda trial: objective(trial, data_splits, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    search_elapsed = time.monotonic() - t0

    # ── Print search results ──────────────────────────────────────────────
    best = study.best_trial
    bp = best.params
    log.info("=" * 60)
    log.info("HP search completed: %d trials in %.1fs", args.n_trials, search_elapsed)
    log.info("=" * 60)
    log.info("Best trial: #%d  val_neg_ELBO=%.4f", best.number, best.value)
    log.info("  Params:")
    for k, v in bp.items():
        log.info("    %-20s %s", k, v)

    # Top 5
    log.info("")
    log.info("Top 5 trials:")
    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else 1e9)
    for t in sorted_trials[:5]:
        log.info(
            "  #%d  neg_ELBO=%.4f  | M=%s ep=%s lr=%s ls=%s os=%s noise=%s",
            t.number, t.value,
            t.params.get("num_inducing"),
            t.params.get("epochs"),
            t.params.get("lr"),
            t.params.get("init_lengthscale"),
            t.params.get("init_outputscale"),
            t.params.get("init_noise"),
        )

    # ── Retrain best HPs on train+val, evaluate on test ───────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Retraining best HPs on train+val, evaluating on test ...")
    log.info("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Refit scalers on full train+val
    X_trainval_full = np.concatenate([X_train_np, X_val_np], axis=0)
    Y_trainval_full = np.concatenate([Y_train_np, Y_val_np], axis=0)

    x_scaler_final = StandardScaler().fit(X_trainval_full)
    y_scaler_final = StandardScaler().fit(Y_trainval_full)

    X_tv = torch.tensor(x_scaler_final.transform(X_trainval_full), dtype=DTYPE)
    X_te = torch.tensor(x_scaler_final.transform(X_test_np), dtype=DTYPE)
    Y_tv = torch.tensor(y_scaler_final.transform(Y_trainval_full), dtype=DTYPE)
    Y_te = torch.tensor(y_scaler_final.transform(Y_test_np), dtype=DTYPE)

    # Inducing points from train+val
    n_ind = min(bp["num_inducing"], len(X_tv))
    X_m_init = select_inducing_points(X_tv.numpy(), n_ind)

    # Train 2 GPs with best HPs
    models = []
    loss_histories = []
    total_time = 0.0

    for i in range(2):
        log.info("Training GP for %s ...", OUTPUT_NAMES[i])
        model = SparseGPR(
            X_m_init=X_m_init.clone(),
            input_dim=len(input_cols),
            init_lengthscale=bp["init_lengthscale"],
            init_outputscale=bp["init_outputscale"],
            init_noise=bp["init_noise"],
        )
        t1 = time.monotonic()
        hist = train_sgpr(model, X_tv, Y_tv[:, i],
                          epochs=bp["epochs"], lr=bp["lr"])
        elapsed = time.monotonic() - t1
        total_time += elapsed
        log.info("  Trained in %.1fs", elapsed)

        model.cache_posterior(X_tv, Y_tv[:, i])
        models.append(model)
        loss_histories.append(hist)

    # ── Evaluate on held-out test set ─────────────────────────────────────
    log.info("Evaluation on test set:")
    metrics, _, _, _ = evaluate(models, X_te, Y_te, y_scaler_final)

    log.info("=== Test Metrics (original scale) ===")
    log.info("  %-25s  %-12s  %-12s", "", "Acceleration", "Steering Vel")
    log.info("  %-25s  %-12.4f  %-12.4f", "MAE",
             metrics["mae"][0], metrics["mae"][1])
    log.info("  %-25s  %-12.4f  %-12.4f", "RMSE",
             metrics["rmse"][0], metrics["rmse"][1])
    log.info("  %-25s  %-11.1f%%  %-11.1f%%", "Coverage (+/- 2std)",
             metrics["coverage"][0], metrics["coverage"][1])

    # ── Save best model ───────────────────────────────────────────────────
    config = {
        "model_type": "sgpr_krasserm",
        "input_cols": input_cols,
        "input_names": input_names,
        "output_cols": OUTPUT_COLS,
        "output_names": OUTPUT_NAMES,
        "feature_set": args.feature_set,
        "num_input_dims": len(input_cols),
        "num_outputs": 2,
        "best_params": bp,
        "n_trials": args.n_trials,
        "best_trial": best.number,
        "best_val_neg_elbo": best.value,
        "n_train": len(X_train_np),
        "n_val": len(X_val_np),
        "n_test": len(X_test_np),
        "mae_accel": float(metrics["mae"][0]),
        "mae_steer": float(metrics["mae"][1]),
        "rmse_accel": float(metrics["rmse"][0]),
        "rmse_steer": float(metrics["rmse"][1]),
        "coverage_2std_accel": float(metrics["coverage"][0]),
        "coverage_2std_steer": float(metrics["coverage"][1]),
        "search_time_s": round(search_elapsed, 2),
        "retrain_time_s": round(total_time, 2),
    }
    save_model(args.output_dir, models, x_scaler_final, y_scaler_final, config)

    log.info("Done.")


if __name__ == "__main__":
    main()
