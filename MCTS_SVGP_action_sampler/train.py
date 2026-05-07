#!/usr/bin/env python3
"""
Train a multi-output SVGP action prior for MCTS.

Learns the mapping:
    (v, delta, omega, beta, max_kappa, avg_kappa, C_alpha) -> (acceleration, steering_vel)

from MPPI demonstration trajectories stored as CSVs in a results directory.

Usage:
    python MCTS_SVGP_action_sampler/train.py --data-dir results/ --epochs 50 --plot
"""

import argparse
import glob
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("train_svgp")

# ── Feature definitions ──────────────────────────────────────────────────────

ALL_INPUT_COLS = ["S_vel", "S_delta", "S_omega", "S_beta", "max_kappa", "avg_kappa", "C_alpha"]
ALL_INPUT_NAMES = ["velocity", "steering_angle", "yaw_rate", "slip_angle",
                   "max_curvature", "avg_curvature", "cornering_stiffness"]
INPUT_COLS_NO_AVG_KAPPA = ["S_vel", "S_delta", "S_omega", "S_beta", "max_kappa", "C_alpha"]
INPUT_NAMES_NO_AVG_KAPPA = ["velocity", "steering_angle", "yaw_rate", "slip_angle",
                            "max_curvature", "cornering_stiffness"]
INPUT_COLS_GENERAL = ["max_kappa", "C_alpha"]
INPUT_NAMES_GENERAL = ["max_curvature", "cornering_stiffness"]
INPUT_COLS_CURVATURE_ONLY = ["max_kappa", "avg_kappa"]
INPUT_NAMES_CURVATURE_ONLY = ["max_curvature", "avg_curvature"]
INPUT_COLS_VEL_KAPPA = ["S_vel", "max_kappa"]
INPUT_NAMES_VEL_KAPPA = ["velocity", "max_curvature"]
# 3D minimal feature set used by `minimal_feature_1st_train/`.
INPUT_COLS_NO_LATERAL = ["S_vel", "max_kappa", "C_alpha"]
INPUT_NAMES_NO_LATERAL = ["velocity", "max_curvature", "cornering_stiffness"]
OUTPUT_COLS = ["A_a", "A_sv"]
OUTPUT_NAMES = ["acceleration", "steering_velocity"]

# Default (will be overridden by --feature-set flag)
INPUT_COLS = ALL_INPUT_COLS
INPUT_NAMES = ALL_INPUT_NAMES

FEATURE_SETS = {
    "no_avg_kappa":   (INPUT_COLS_NO_AVG_KAPPA,   INPUT_NAMES_NO_AVG_KAPPA),
    "no_lateral":     (INPUT_COLS_NO_LATERAL,      INPUT_NAMES_NO_LATERAL),
    "general":        (INPUT_COLS_GENERAL,         INPUT_NAMES_GENERAL),
    "curvature_only": (INPUT_COLS_CURVATURE_ONLY,  INPUT_NAMES_CURVATURE_ONLY),
    "vel_kappa":      (INPUT_COLS_VEL_KAPPA,       INPUT_NAMES_VEL_KAPPA),
    "all":            (ALL_INPUT_COLS,              ALL_INPUT_NAMES),
}


def get_input_features(feature_set: str = "no_avg_kappa"):
    """Return (INPUT_COLS, INPUT_NAMES) for the given feature set name."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Choose from: {list(FEATURE_SETS.keys())}"
        )
    cols, names = FEATURE_SETS[feature_set]
    return list(cols), list(names)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(results_dir: str, input_cols: list = None) -> pd.DataFrame:
    """Load all CSV trajectory files from results_dir, filter, return DataFrame."""
    if input_cols is None:
        input_cols = INPUT_COLS

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, comment="#")
        log.info("Loaded %s: %d rows", os.path.basename(f), len(df))
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    log.info("Total rows before filtering: %d", len(data))

    # Drop rows with NaN in required columns
    required = input_cols + OUTPUT_COLS
    data = data.dropna(subset=required)
    log.info("After dropping NaN: %d rows", len(data))

    return data


# ── SVGP Model ───────────────────────────────────────────────────────────────

class ActionPriorSVGP(ApproximateGP):
    """Multi-output SVGP for learning action prior from MPPI demonstrations.

    Uses ARD RBF kernel (separate length scales per input dimension, per output).
    Two outputs: acceleration and steering velocity.
    """

    def __init__(self, inducing_points: torch.Tensor, num_tasks: int = 2,
                 num_input_dims: int = None):
        if num_input_dims is None:
            num_input_dims = inducing_points.size(-1)
        batch_shape = torch.Size([num_tasks])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=num_input_dims, batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── Inducing point selection ─────────────────────────────────────────────────

def select_inducing_points(X_train: np.ndarray, M: int) -> torch.Tensor:
    """Select M inducing points via k-means clustering on training data."""
    log.info("Selecting %d inducing points via k-means...", M)
    kmeans = MiniBatchKMeans(n_clusters=M, random_state=42, batch_size=1024)
    kmeans.fit(X_train)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train_svgp(
    model: ActionPriorSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 0.01,
    device: str = "cpu",
) -> list:
    """Train multi-output SVGP with VariationalELBO and Adam."""
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=train_y.size(0),
    )

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info("Epoch %3d/%d  loss: %.4f", epoch + 1, num_epochs, avg_loss)

    return loss_history


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_svgp(
    model: ActionPriorSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    y_scaler: StandardScaler,
    device: str = "cpu",
):
    """Evaluate on test set. Returns metrics and predictions in original scale.

    Returns:
        dict with keys: mae, rmse, coverage_2std (each shape (2,))
        pred_mean_original: (N, 2)
        pred_std_original: (N, 2)
        true_original: (N, 2)
    """
    model.eval()
    likelihood.eval()

    all_means = []
    all_vars = []
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            preds = likelihood(model(x_batch))
            all_means.append(preds.mean.cpu())
            all_vars.append(preds.variance.cpu())

    pred_mean = torch.cat(all_means, dim=0).numpy()  # (N, 2)
    pred_var = torch.cat(all_vars, dim=0).numpy()     # (N, 2)
    true = test_y.numpy()  # (N, 2)

    # Inverse transform to original scale
    # mean: y_original = y_scaled * scale + mean
    # std:  std_original = std_scaled * scale
    pred_mean_original = y_scaler.inverse_transform(pred_mean)
    true_original = y_scaler.inverse_transform(true)
    pred_std_original = np.sqrt(pred_var) * y_scaler.scale_  # (N, 2)

    # MAE
    mae = np.mean(np.abs(pred_mean_original - true_original), axis=0)

    # RMSE
    rmse = np.sqrt(np.mean((pred_mean_original - true_original) ** 2, axis=0))

    # Coverage: % of true values within mean ± 2*std
    lower = pred_mean_original - 2.0 * pred_std_original
    upper = pred_mean_original + 2.0 * pred_std_original
    within = (true_original >= lower) & (true_original <= upper)
    coverage_2std = np.mean(within, axis=0) * 100.0  # percentage

    metrics = {"mae": mae, "rmse": rmse, "coverage_2std": coverage_2std}
    return metrics, pred_mean_original, pred_std_original, true_original


# ── Saving ───────────────────────────────────────────────────────────────────

def save_model(
    output_dir: str,
    model: ActionPriorSVGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    inducing_points: torch.Tensor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    config: dict,
):
    """Save trained model, likelihood, inducing points, scalers, and config."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(output_dir, "likelihood.pth"))
    torch.save(inducing_points, os.path.join(output_dir, "inducing_points.pt"))

    np.savez(
        os.path.join(output_dir, "scalers.npz"),
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
    )

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved model to %s", output_dir)


def load_model(model_dir: str, device: str = "cpu"):
    """Load trained action prior for inference."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    inducing_points = torch.load(
        os.path.join(model_dir, "inducing_points.pt"), map_location=device,
    )

    num_input_dims = len(config.get("input_cols", ALL_INPUT_COLS))
    model = ActionPriorSVGP(inducing_points, num_tasks=2,
                            num_input_dims=num_input_dims)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pth"), map_location=device),
    )

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    likelihood.load_state_dict(
        torch.load(os.path.join(model_dir, "likelihood.pth"), map_location=device),
    )

    scalers_data = np.load(os.path.join(model_dir, "scalers.npz"))
    x_scaler = StandardScaler()
    x_scaler.mean_ = scalers_data["x_mean"]
    x_scaler.scale_ = scalers_data["x_scale"]
    x_scaler.var_ = x_scaler.scale_ ** 2
    x_scaler.n_features_in_ = len(x_scaler.mean_)

    y_scaler = StandardScaler()
    y_scaler.mean_ = scalers_data["y_mean"]
    y_scaler.scale_ = scalers_data["y_scale"]
    y_scaler.var_ = y_scaler.scale_ ** 2
    y_scaler.n_features_in_ = len(y_scaler.mean_)

    model = model.to(device)
    likelihood = likelihood.to(device)
    model.eval()
    likelihood.eval()

    return model, likelihood, x_scaler, y_scaler, config


# ── Plotting (optional) ─────────────────────────────────────────────────────

def plot_results(
    pred: np.ndarray,
    true: np.ndarray,
    loss_history: list,
    model: ActionPriorSVGP,
    output_dir: str,
    input_names: list = None,
):
    """Generate diagnostic plots and save to output_dir."""
    if input_names is None:
        input_names = ALL_INPUT_NAMES
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predicted vs true: acceleration
    axes[0].scatter(true[:, 0], pred[:, 0], alpha=0.3, s=5)
    lims = [min(true[:, 0].min(), pred[:, 0].min()),
            max(true[:, 0].max(), pred[:, 0].max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("True acceleration")
    axes[0].set_ylabel("Predicted acceleration")
    axes[0].set_title("Acceleration: Predicted vs True")

    # Predicted vs true: steering velocity
    axes[1].scatter(true[:, 1], pred[:, 1], alpha=0.3, s=5)
    lims = [min(true[:, 1].min(), pred[:, 1].min()),
            max(true[:, 1].max(), pred[:, 1].max())]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("True steering velocity")
    axes[1].set_ylabel("Predicted steering velocity")
    axes[1].set_title("Steering Vel: Predicted vs True")

    # Training loss
    axes[2].plot(loss_history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Negative ELBO")
    axes[2].set_title("Training Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_diagnostics.png"), dpi=150)
    log.info("Saved plot to %s/training_diagnostics.png", output_dir)
    plt.show()

    # ARD length scales
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    ls = lengthscales.squeeze()  # (2, D)
    x_pos = np.arange(len(input_names))

    for i, (ax, name) in enumerate(zip(axes2, OUTPUT_NAMES)):
        ax.bar(x_pos, ls[i])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(input_names, rotation=45, ha="right")
        ax.set_ylabel("Length scale")
        ax.set_title(f"ARD Length Scales: {name}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ard_lengthscales.png"), dpi=150)
    log.info("Saved ARD plot to %s/ard_lengthscales.png", output_dir)
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train action-prior SVGP from MPPI demonstration data.",
    )
    parser.add_argument("--data-dir", default="results/",
                        help="Directory containing trajectory CSVs")
    parser.add_argument("--output-dir", default="models/action_prior_svgp/",
                        help="Directory to save trained model")
    parser.add_argument("--num-inducing", type=int, default=128,
                        help="Number of inducing points")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data for test set")
    parser.add_argument("--init-lengthscale", type=float, default=None,
                        help="Initial RBF kernel lengthscale")
    parser.add_argument("--init-outputscale", type=float, default=None,
                        help="Initial ScaleKernel outputscale")
    parser.add_argument("--init-noise", type=float, default=None,
                        help="Initial likelihood noise")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--feature-set", default="no_avg_kappa",
                        choices=list(FEATURE_SETS.keys()),
                        help="Input feature set (default: no_avg_kappa)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Feature selection ─────────────────────────────────────────────────
    input_cols, input_names = get_input_features(args.feature_set)
    log.info("Feature set '%s' (%dD): %s", args.feature_set, len(input_cols), input_cols)

    # ── 1. Load data ──────────────────────────────────────────────────────
    data = load_data(args.data_dir, input_cols=input_cols)

    X = data[input_cols].values.astype(np.float32)
    Y = data[OUTPUT_COLS].values.astype(np.float32)
    log.info("Features shape: %s  Targets shape: %s", X.shape, Y.shape)

    # ── 2. Train/test split ───────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_split, random_state=args.seed,
    )
    log.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # ── 3. Normalize ──────────────────────────────────────────────────────
    x_scaler = StandardScaler()
    X_train_norm = x_scaler.fit_transform(X_train)
    X_test_norm = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    Y_train_norm = y_scaler.fit_transform(Y_train)
    Y_test_norm = y_scaler.transform(Y_test)

    # Convert to tensors
    train_x = torch.tensor(X_train_norm, dtype=torch.float32)
    train_y = torch.tensor(Y_train_norm, dtype=torch.float32)
    test_x = torch.tensor(X_test_norm, dtype=torch.float32)
    test_y = torch.tensor(Y_test_norm, dtype=torch.float32)

    # ── 4. Select inducing points ─────────────────────────────────────────
    inducing_np = select_inducing_points(X_train_norm, args.num_inducing)
    # Expand for batch: (M, D) — VariationalStrategy handles batch internally
    inducing_points = inducing_np.clone()

    # ── 5. Create model and likelihood ────────────────────────────────────
    model = ActionPriorSVGP(inducing_points, num_tasks=2,
                            num_input_dims=len(input_cols))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    if args.init_lengthscale is not None:
        model.covar_module.base_kernel.lengthscale = args.init_lengthscale
    if args.init_outputscale is not None:
        model.covar_module.outputscale = args.init_outputscale
    if args.init_noise is not None:
        likelihood.noise = args.init_noise

    log.info("Model: %d inducing points, %d input dims, 2 output tasks",
             args.num_inducing, len(input_cols))

    # ── 6. Train ──────────────────────────────────────────────────────────
    t0 = time.time()
    loss_history = train_svgp(
        model, likelihood, train_x, train_y,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    elapsed = time.time() - t0
    log.info("Training completed in %.1fs", elapsed)

    # Move back to CPU for evaluation/saving
    model = model.cpu()
    likelihood = likelihood.cpu()

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    metrics, pred, pred_std, true = evaluate_svgp(
        model, likelihood, test_x, test_y, y_scaler, device="cpu",
    )
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    coverage = metrics["coverage_2std"]

    log.info("=== Test Metrics (original scale) ===")
    log.info("  %-25s  %-12s  %-12s", "", "Acceleration", "Steering Vel")
    log.info("  %-25s  %-12.4f  %-12.4f", "MAE", mae[0], mae[1])
    log.info("  %-25s  %-12.4f  %-12.4f", "RMSE", rmse[0], rmse[1])
    log.info("  %-25s  %-11.2f%%  %-11.2f%%", "Coverage (mean +/- 2std)", coverage[0], coverage[1])

    # Print ARD length scales
    lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze()
    log.info("=== ARD Length Scales ===")
    for task_idx, task_name in enumerate(OUTPUT_NAMES):
        log.info("  %s:", task_name)
        for feat_idx, feat_name in enumerate(input_names):
            log.info("    %-25s  %.4f", feat_name, lengthscales[task_idx, feat_idx])

    # ── 8. Save ───────────────────────────────────────────────────────────
    config = {
        "input_cols": input_cols,
        "output_cols": OUTPUT_COLS,
        "input_names": input_names,
        "output_names": OUTPUT_NAMES,
        "feature_set": args.feature_set,
        "num_inducing": args.num_inducing,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "test_split": args.test_split,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "final_loss": loss_history[-1],
        "mae_accel": float(mae[0]),
        "mae_steer": float(mae[1]),
        "rmse_accel": float(rmse[0]),
        "rmse_steer": float(rmse[1]),
        "coverage_2std_accel": float(coverage[0]),
        "coverage_2std_steer": float(coverage[1]),
        "training_time_s": elapsed,
    }

    save_model(args.output_dir, model, likelihood, inducing_points,
               x_scaler, y_scaler, config)

    # ── 9. Plot (optional) ────────────────────────────────────────────────
    if args.plot:
        plot_results(pred, true, loss_history, model, args.output_dir,
                     input_names=input_names)

    log.info("Done.")


if __name__ == "__main__":
    main()
