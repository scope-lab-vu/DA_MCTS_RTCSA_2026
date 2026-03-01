#!/usr/bin/env python3
"""
Hyperparameter search for the action-prior SVGP using Optuna.

Tunes: num_inducing, epochs, batch_size, learning_rate,
       initial kernel lengthscale, outputscale, and likelihood noise.
Objective: minimise negative ELBO on a validation set.
After search, retrains best HPs and evaluates on held-out test set.

Data split: train (64%) / val (16%) / test (20%)
  - Optuna trials train on train, minimise neg-ELBO on val
  - Best trial is retrained on train+val, evaluated on test

Usage:
    python MCTS_SVGP_action_sampler/hp_search.py --data-dir results/ --n-trials 50
    python MCTS_SVGP_action_sampler/hp_search.py --data-dir results/ --n-trials 100 --device cuda
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
from torch.utils.data import DataLoader, TensorDataset

import gpytorch

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    FEATURE_SETS,
    OUTPUT_COLS,
    OUTPUT_NAMES,
    ActionPriorSVGP,
    evaluate_svgp,
    get_input_features,
    load_data,
    save_model,
    select_inducing_points,
    train_svgp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("hp_search")


def compute_val_elbo(
    model: ActionPriorSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    n_train: int,
    device: str = "cpu",
) -> float:
    """Compute negative ELBO on validation set (lower is better)."""
    model.eval()
    likelihood.eval()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = -mll(output, y_batch)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def objective(trial: optuna.Trial, data_splits: dict, args: argparse.Namespace) -> float:
    """Single Optuna trial: train SVGP, return negative ELBO on validation set."""

    # ── Sample hyperparameters ────────────────────────────────────────────
    num_inducing = trial.suggest_int("num_inducing", 32, 512, step=32)
    epochs = trial.suggest_int("epochs", 20, 150, step=10)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)

    # Kernel / likelihood initial values
    init_lengthscale = trial.suggest_float("init_lengthscale", 0.1, 10.0, log=True)
    init_outputscale = trial.suggest_float("init_outputscale", 0.1, 10.0, log=True)
    init_noise = trial.suggest_float("init_noise", 1e-4, 1.0, log=True)

    seed = args.seed
    device = args.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x = data_splits["train_x"]
    train_y = data_splits["train_y"]
    val_x = data_splits["val_x"]
    val_y = data_splits["val_y"]
    X_train_norm = data_splits["X_train_norm"]

    # ── Inducing points ───────────────────────────────────────────────────
    n_inducing = min(num_inducing, len(X_train_norm))
    inducing_points = select_inducing_points(X_train_norm, n_inducing)

    # ── Model ─────────────────────────────────────────────────────────────
    num_input_dims = data_splits["num_input_dims"]
    model = ActionPriorSVGP(inducing_points, num_tasks=2,
                            num_input_dims=num_input_dims)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    # Set initial kernel / likelihood hyperparameters
    model.covar_module.base_kernel.lengthscale = init_lengthscale
    model.covar_module.outputscale = init_outputscale
    likelihood.noise = init_noise

    # ── Train ─────────────────────────────────────────────────────────────
    train_svgp(
        model, likelihood, train_x, train_y,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    # Move back to CPU for validation
    model = model.cpu()
    likelihood = likelihood.cpu()

    # ── Compute negative ELBO on validation set ───────────────────────────
    neg_elbo = compute_val_elbo(
        model, likelihood, val_x, val_y,
        n_train=len(train_x), device="cpu",
    )

    trial.set_user_attr("neg_elbo", neg_elbo)

    log.info(
        "Trial %d: neg_ELBO=%.4f  (M=%d, ep=%d, bs=%d, lr=%.5f, "
        "ls=%.3f, os=%.3f, noise=%.5f)",
        trial.number, neg_elbo,
        num_inducing, epochs, batch_size, lr,
        init_lengthscale, init_outputscale, init_noise,
    )

    return neg_elbo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for action-prior SVGP.",
    )
    parser.add_argument("--data-dir", default="results/",
                        help="Directory containing trajectory CSVs")
    parser.add_argument("--output-dir", default="models/action_prior_svgp/",
                        help="Directory to save best model")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data for held-out test set")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of remaining data for validation")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--study-name", default="svgp_action_prior",
                        help="Optuna study name")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (default: in-memory)")
    parser.add_argument("--feature-set", default="no_avg_kappa",
                        choices=list(FEATURE_SETS.keys()),
                        help="Input feature set (default: no_avg_kappa)")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Feature selection ─────────────────────────────────────────────────
    input_cols, input_names = get_input_features(args.feature_set)
    log.info("Feature set '%s' (%dD): %s", args.feature_set, len(input_cols), input_cols)

    # Append feature set name to output dir and study name for separation
    if not args.output_dir.rstrip("/").endswith(args.feature_set):
        args.output_dir = os.path.join(args.output_dir.rstrip("/") + f"_{args.feature_set}")
    if not args.study_name.endswith(args.feature_set):
        args.study_name = f"{args.study_name}_{args.feature_set}"

    # ── Pre-load all data ─────────────────────────────────────────────────
    data = load_data(args.data_dir, input_cols=input_cols)

    X_all = data[input_cols].values.astype(np.float32)
    Y_all = data[OUTPUT_COLS].values.astype(np.float32)
    log.info("Loaded %d rows total", len(X_all))

    # ── Split: train / val / test ─────────────────────────────────────────
    # First split off test set
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_all, Y_all, test_size=args.test_split, random_state=args.seed,
    )
    # Then split trainval into train and val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=args.val_split, random_state=args.seed,
    )
    log.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # ── Normalise (fit on train only) ─────────────────────────────────────
    x_scaler = StandardScaler()
    X_train_norm = x_scaler.fit_transform(X_train)
    X_val_norm = x_scaler.transform(X_val)
    y_scaler = StandardScaler()
    Y_train_norm = y_scaler.fit_transform(Y_train)
    Y_val_norm = y_scaler.transform(Y_val)

    train_x = torch.tensor(X_train_norm, dtype=torch.float32)
    train_y = torch.tensor(Y_train_norm, dtype=torch.float32)
    val_x = torch.tensor(X_val_norm, dtype=torch.float32)
    val_y = torch.tensor(Y_val_norm, dtype=torch.float32)
    data_splits = {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "X_train_norm": X_train_norm,
        "num_input_dims": len(input_cols),
    }

    # ── Run Optuna study ──────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, data_splits, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    search_elapsed = time.time() - t0

    # ── Print search results ──────────────────────────────────────────────
    best = study.best_trial
    bp = best.params
    log.info("=" * 60)
    log.info("HP search completed: %d trials in %.1fs", args.n_trials, search_elapsed)
    log.info("=" * 60)
    log.info("Best trial: #%d  neg_ELBO=%.4f", best.number, best.value)
    log.info("  Params:")
    for k, v in bp.items():
        log.info("    %-20s %s", k, v)

    # ── Print top 5 trials ────────────────────────────────────────────────
    log.info("")
    log.info("Top 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 1e9)
    for t in sorted_trials[:5]:
        log.info(
            "  #%d  neg_ELBO=%.4f  | M=%s ep=%s bs=%s lr=%s ls=%s os=%s noise=%s",
            t.number, t.value,
            t.params.get("num_inducing"),
            t.params.get("epochs"),
            t.params.get("batch_size"),
            t.params.get("lr"),
            t.params.get("init_lengthscale"),
            t.params.get("init_outputscale"),
            t.params.get("init_noise"),
        )

    # ── Retrain best HPs on train+val, evaluate on test ───────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Retraining best HPs on train+val, evaluating on test...")
    log.info("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Refit scalers on full train+val
    x_scaler_final = StandardScaler()
    X_trainval_final = np.concatenate([X_train, X_val], axis=0)
    Y_trainval_final = np.concatenate([Y_train, Y_val], axis=0)
    X_trainval_norm_final = x_scaler_final.fit_transform(X_trainval_final)
    X_test_norm_final = x_scaler_final.transform(X_test)

    y_scaler_final = StandardScaler()
    Y_trainval_norm_final = y_scaler_final.fit_transform(Y_trainval_final)
    Y_test_norm_final = y_scaler_final.transform(Y_test)

    trainval_x_final = torch.tensor(X_trainval_norm_final, dtype=torch.float32)
    trainval_y_final = torch.tensor(Y_trainval_norm_final, dtype=torch.float32)
    test_x_final = torch.tensor(X_test_norm_final, dtype=torch.float32)
    test_y_final = torch.tensor(Y_test_norm_final, dtype=torch.float32)

    # Inducing points from train+val
    n_inducing = min(bp["num_inducing"], len(X_trainval_norm_final))
    inducing_points = select_inducing_points(X_trainval_norm_final, n_inducing)

    model = ActionPriorSVGP(inducing_points, num_tasks=2,
                            num_input_dims=len(input_cols))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    model.covar_module.base_kernel.lengthscale = bp["init_lengthscale"]
    model.covar_module.outputscale = bp["init_outputscale"]
    likelihood.noise = bp["init_noise"]

    t0 = time.time()
    loss_history = train_svgp(
        model, likelihood, trainval_x_final, trainval_y_final,
        num_epochs=bp["epochs"],
        batch_size=bp["batch_size"],
        lr=bp["lr"],
        device=args.device,
    )
    train_elapsed = time.time() - t0
    log.info("Retraining completed in %.1fs", train_elapsed)

    model = model.cpu()
    likelihood = likelihood.cpu()

    # ── Evaluate on held-out test set ─────────────────────────────────────
    metrics, _, _, _ = evaluate_svgp(
        model, likelihood, test_x_final, test_y_final, y_scaler_final, device="cpu",
    )
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    coverage = metrics["coverage_2std"]

    log.info("=== Test Metrics (original scale) ===")
    log.info("  %-25s  %-12s  %-12s", "", "Acceleration", "Steering Vel")
    log.info("  %-25s  %-12.4f  %-12.4f", "MAE", mae[0], mae[1])
    log.info("  %-25s  %-12.4f  %-12.4f", "RMSE", rmse[0], rmse[1])
    log.info("  %-25s  %-11.2f%%  %-11.2f%%", "Coverage (mean +/- 2std)", coverage[0], coverage[1])

    # ── Save best model ───────────────────────────────────────────────────
    config = {
        "input_cols": input_cols,
        "input_names": input_names,
        "output_cols": OUTPUT_COLS,
        "output_names": OUTPUT_NAMES,
        "feature_set": args.feature_set,
        "best_params": bp,
        "n_trials": args.n_trials,
        "best_trial": best.number,
        "best_val_neg_elbo": best.value,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "final_loss": loss_history[-1],
        "mae_accel": float(mae[0]),
        "mae_steer": float(mae[1]),
        "rmse_accel": float(rmse[0]),
        "rmse_steer": float(rmse[1]),
        "coverage_2std_accel": float(coverage[0]),
        "coverage_2std_steer": float(coverage[1]),
        "search_time_s": search_elapsed,
        "retrain_time_s": train_elapsed,
    }

    save_model(args.output_dir, model, likelihood, inducing_points,
               x_scaler_final, y_scaler_final, config)

    log.info("Done.")


if __name__ == "__main__":
    main()
