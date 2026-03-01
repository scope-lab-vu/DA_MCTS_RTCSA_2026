#!/usr/bin/env python3
"""
Sparse GP Regression (SGPR) with VFE — Krasserm implementation.

Trains independent single-output sparse GPs for acceleration and steering
velocity using the Variational Free Energy lower bound with:
  - Analytical variational parameters (no variational optimisation needed)
  - L-BFGS optimisation of kernel hyperparameters + inducing locations
  - ARD RBF kernel (per-dimension length scales)

Two independent GPs are trained (one per output), then cached for prediction.

Reference: https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/

Usage:
    python MCTS_SVGP_action_sampler/train_krasserm.py \\
        --data-dir MCTS_SVGP_action_sampler/training_data --epochs 100 --plot
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
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("train_krasserm")

# ── Feature / target columns (shared with train.py) ─────────────────────────

ALL_INPUT_COLS = [
    "S_vel", "S_delta", "S_omega", "S_beta",
    "max_kappa", "avg_kappa", "C_alpha",
]
ALL_INPUT_NAMES = [
    "velocity", "steering_angle", "yaw_rate", "slip_angle",
    "max_curvature", "avg_curvature", "cornering_stiffness",
]
INPUT_COLS_NO_AVG_KAPPA = [
    "S_vel", "S_delta", "S_omega", "S_beta", "max_kappa", "C_alpha",
]
INPUT_NAMES_NO_AVG_KAPPA = [
    "velocity", "steering_angle", "yaw_rate", "slip_angle",
    "max_curvature", "cornering_stiffness",
]
INPUT_COLS_NO_LATERAL = [
    "S_vel", "max_kappa", "C_alpha",
]
INPUT_NAMES_NO_LATERAL = [
    "velocity", "max_curvature", "cornering_stiffness",
]
INPUT_COLS_GENERAL = [
    "max_kappa", "C_alpha",
]
INPUT_NAMES_GENERAL = [
    "max_curvature", "cornering_stiffness",
]
OUTPUT_COLS = ["A_a", "A_sv"]
OUTPUT_NAMES = ["acceleration", "steering_velocity"]

DTYPE = torch.float64
JITTER = 1e-6

FEATURE_SETS = {
    "no_avg_kappa": (INPUT_COLS_NO_AVG_KAPPA, INPUT_NAMES_NO_AVG_KAPPA),
    "no_lateral":   (INPUT_COLS_NO_LATERAL,    INPUT_NAMES_NO_LATERAL),
    "general":      (INPUT_COLS_GENERAL,       INPUT_NAMES_GENERAL),
    "all":          (ALL_INPUT_COLS,            ALL_INPUT_NAMES),
}


def get_input_features(feature_set: str = "no_avg_kappa"):
    """Return (input_cols, input_names) for the given feature set name."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Choose from: {list(FEATURE_SETS.keys())}"
        )
    cols, names = FEATURE_SETS[feature_set]
    return list(cols), list(names)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(results_dir: str, input_cols: list) -> pd.DataFrame:
    """Load all CSV trajectory files from results_dir."""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    frames = []
    required = input_cols + OUTPUT_COLS
    for f in csv_files:
        df = pd.read_csv(f, comment="#")
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.warning("Skipping %s (missing: %s)", os.path.basename(f), missing)
            continue
        df = df.dropna(subset=required)
        frames.append(df)
        log.info("Loaded %s: %d rows", os.path.basename(f), len(df))

    if not frames:
        raise ValueError("No usable CSV files found")

    data = pd.concat(frames, ignore_index=True)
    log.info("Total: %d samples from %d files", len(data), len(frames))
    return data


# ── ARD RBF Kernel ───────────────────────────────────────────────────────────

def _ard_rbf(X1, X2, log_ls, log_os):
    """K(X1,X2) = sigma_f^2 * exp(-0.5 * sum_d (x1_d-x2_d)^2 / l_d^2)."""
    ls = torch.exp(log_ls)          # (d,)
    os2 = torch.exp(2.0 * log_os)   # scalar
    X1s = X1 / ls
    X2s = X2 / ls
    sq = torch.cdist(X1s, X2s) ** 2
    return os2 * torch.exp(-0.5 * sq)


def _rbf_diag_sum(n, log_os):
    """Sum of diagonal of K(X,X) for RBF = n * sigma_f^2."""
    return n * torch.exp(2.0 * log_os)


# ── Sparse GPR model (single output) ────────────────────────────────────────

class SparseGPR(nn.Module):
    """Single-output Sparse GP Regression using VFE (Krasserm).

    Optimises kernel hyperparameters (ARD length scales, output scale, noise)
    and inducing point locations via L-BFGS.  Variational parameters (mu_m,
    A_m) are computed analytically after training.
    """

    def __init__(
        self,
        X_m_init: torch.Tensor,
        input_dim: int,
        init_lengthscale: float = 1.0,
        init_outputscale: float = 1.0,
        init_noise: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.log_lengthscales = nn.Parameter(
            torch.full((input_dim,), np.log(init_lengthscale), dtype=DTYPE))
        self.log_outputscale = nn.Parameter(
            torch.tensor(np.log(init_outputscale), dtype=DTYPE))
        self.log_noise = nn.Parameter(
            torch.tensor(np.log(init_noise), dtype=DTYPE))
        self.X_m = nn.Parameter(X_m_init.to(DTYPE))

        # Cached posterior (filled by cache_posterior)
        self._mu_m: torch.Tensor = None
        self._A_m: torch.Tensor = None
        self._K_mm_inv: torch.Tensor = None

    # ── VFE lower bound (eq. 13 in Krasserm) ────────────────────────────

    def neg_elbo(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative VFE lower bound (minimise this)."""
        n = X.shape[0]
        m = self.X_m.shape[0]
        sigma_y = torch.exp(self.log_noise)
        dev = X.device

        K_mm = _ard_rbf(self.X_m, self.X_m, self.log_lengthscales, self.log_outputscale)
        K_mm = K_mm + JITTER * torch.eye(m, dtype=DTYPE, device=dev)
        K_mn = _ard_rbf(self.X_m, X, self.log_lengthscales, self.log_outputscale)

        L = torch.linalg.cholesky(K_mm)                          # (m, m)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False) / sigma_y  # (m, n)
        AAT = A @ A.T                                             # (m, m)
        B = torch.eye(m, dtype=DTYPE, device=dev) + AAT
        LB = torch.linalg.cholesky(B)                             # (m, m)
        Ay = (A @ y).unsqueeze(1)                                  # (m, 1)
        c = torch.linalg.solve_triangular(LB, Ay, upper=False).squeeze(1) / sigma_y

        lb = -0.5 * n * np.log(2.0 * np.pi)
        lb -= torch.sum(torch.log(torch.diag(LB)))
        lb -= 0.5 * n * torch.log(sigma_y ** 2)
        lb -= 0.5 / sigma_y ** 2 * torch.dot(y, y)
        lb += 0.5 * torch.dot(c, c)
        lb -= 0.5 / sigma_y ** 2 * _rbf_diag_sum(n, self.log_outputscale)
        lb += 0.5 * torch.trace(AAT)

        return -lb

    # ── Analytical variational parameters ────────────────────────────────

    @torch.no_grad()
    def cache_posterior(self, X: torch.Tensor, y: torch.Tensor):
        """Compute and cache optimal mu_m, A_m, K_mm_inv for prediction."""
        m = self.X_m.shape[0]
        dev = X.device
        sigma_y = torch.exp(self.log_noise)
        prec = 1.0 / sigma_y ** 2

        K_mm = _ard_rbf(self.X_m, self.X_m, self.log_lengthscales, self.log_outputscale)
        K_mm = K_mm + JITTER * torch.eye(m, dtype=DTYPE, device=dev)
        K_mn = _ard_rbf(self.X_m, X, self.log_lengthscales, self.log_outputscale)

        Sigma = torch.linalg.inv(K_mm + prec * K_mn @ K_mn.T)
        self._mu_m = prec * (K_mm @ Sigma @ K_mn) @ y
        self._A_m = K_mm @ Sigma @ K_mm
        self._K_mm_inv = torch.linalg.inv(K_mm)

    # ── Prediction ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X_test: torch.Tensor):
        """Predictive mean, function variance, and observation variance."""
        assert self._mu_m is not None, "Call cache_posterior() first."

        K_sm = _ard_rbf(X_test, self.X_m, self.log_lengthscales, self.log_outputscale)
        alpha = self._K_mm_inv @ self._mu_m          # (m,)
        f_mean = K_sm @ alpha                         # (n_test,)

        V = self._K_mm_inv @ K_sm.T                   # (m, n_test)
        k_diag = torch.exp(2.0 * self.log_outputscale) * torch.ones(
            X_test.shape[0], dtype=DTYPE, device=X_test.device)
        qf_diag = torch.sum(K_sm.T * V, dim=0)
        corr = torch.sum((self._A_m @ V) * V, dim=0)

        f_var = torch.clamp(k_diag - qf_diag + corr, min=1e-10)
        y_var = f_var + torch.exp(self.log_noise) ** 2

        return f_mean, f_var, y_var


# ── Inducing point initialisation ────────────────────────────────────────────

def select_inducing_points(X: np.ndarray, m: int) -> torch.Tensor:
    """Select m inducing points via k-means clustering."""
    log.info("Selecting %d inducing points via k-means ...", m)
    km = MiniBatchKMeans(n_clusters=m, batch_size=min(1024, len(X)),
                         n_init=3, random_state=42)
    km.fit(X)
    return torch.tensor(km.cluster_centers_, dtype=DTYPE)


# ── Training ─────────────────────────────────────────────────────────────────

def train_sgpr(
    model: SparseGPR,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1,
    verbose: bool = True,
) -> list:
    """Train a single-output SparseGPR with L-BFGS."""
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr,
        max_iter=20, history_size=20,
        line_search_fn="strong_wolfe",
    )
    history = []

    for ep in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = model.neg_elbo(X, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        lv = loss.item()
        history.append(lv)

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            ls = torch.exp(model.log_lengthscales).detach().cpu().numpy()
            log.info(
                "  epoch %3d/%d  neg_LB=%.2f  noise=%.4f  outscale=%.4f  ls=%s",
                ep, epochs, lv,
                torch.exp(model.log_noise).item(),
                torch.exp(model.log_outputscale).item(),
                np.array2string(ls, precision=3, separator=", "),
            )

    return history


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(models, X_test, Y_test, y_scaler):
    """Evaluate independent GPs on the test set (metrics in original scale)."""
    all_mae, all_rmse, all_cov = [], [], []
    all_mean_orig, all_std_orig, all_true_orig = [], [], []

    for i, m in enumerate(models):
        f_mean, _, y_var = m.predict(X_test)
        mean_orig = f_mean.cpu().numpy() * y_scaler.scale_[i] + y_scaler.mean_[i]
        std_orig = np.sqrt(y_var.cpu().numpy()) * y_scaler.scale_[i]
        y_true = Y_test[:, i].cpu().numpy() * y_scaler.scale_[i] + y_scaler.mean_[i]

        err = np.abs(mean_orig - y_true)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        in_band = np.abs(y_true - mean_orig) <= 2.0 * std_orig
        cov = float(np.mean(in_band) * 100)

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_cov.append(cov)
        all_mean_orig.append(mean_orig)
        all_std_orig.append(std_orig)
        all_true_orig.append(y_true)

        log.info(
            "  %s: MAE=%.4f  RMSE=%.4f  coverage(+/-2std)=%.1f%%",
            OUTPUT_NAMES[i], mae, rmse, cov,
        )

    return (
        {"mae": all_mae, "rmse": all_rmse, "coverage": all_cov},
        np.column_stack(all_mean_orig),
        np.column_stack(all_std_orig),
        np.column_stack(all_true_orig),
    )


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_model(output_dir, models, x_scaler, y_scaler, config):
    """Save trained SGPR models, scalers, and config."""
    os.makedirs(output_dir, exist_ok=True)

    for i, m in enumerate(models):
        torch.save({
            "state_dict": m.state_dict(),
            "mu_m": m._mu_m,
            "A_m": m._A_m,
            "K_mm_inv": m._K_mm_inv,
            "input_dim": m.input_dim,
        }, os.path.join(output_dir, f"gp_{i}.pt"))

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
    """Load trained SGPR models for inference.

    Returns (models, x_scaler, y_scaler, config).
    """
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    scalers = np.load(os.path.join(model_dir, "scalers.npz"))
    x_scaler = StandardScaler()
    x_scaler.mean_ = scalers["x_mean"]
    x_scaler.scale_ = scalers["x_scale"]
    x_scaler.var_ = scalers["x_scale"] ** 2
    x_scaler.n_features_in_ = len(scalers["x_mean"])

    y_scaler = StandardScaler()
    y_scaler.mean_ = scalers["y_mean"]
    y_scaler.scale_ = scalers["y_scale"]
    y_scaler.var_ = scalers["y_scale"] ** 2
    y_scaler.n_features_in_ = len(scalers["y_mean"])

    num_outputs = config.get("num_outputs", 2)
    models = []
    for i in range(num_outputs):
        ckpt = torch.load(
            os.path.join(model_dir, f"gp_{i}.pt"), map_location=device)
        input_dim = ckpt["input_dim"]
        m_pts = ckpt["state_dict"]["X_m"].shape[0]
        model = SparseGPR(
            X_m_init=torch.zeros(m_pts, input_dim, dtype=DTYPE),
            input_dim=input_dim,
        )
        model.load_state_dict(ckpt["state_dict"])
        model._mu_m = ckpt["mu_m"].to(device)
        model._A_m = ckpt["A_m"].to(device)
        model._K_mm_inv = ckpt["K_mm_inv"].to(device)
        model.eval()
        models.append(model)

    return models, x_scaler, y_scaler, config


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(pred, pred_std, true, loss_histories, models, output_dir,
                 input_names=None):
    """Generate diagnostic plots."""
    if input_names is None:
        input_names = ALL_INPUT_NAMES
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_out = pred.shape[1]

    # ── Pred vs True + Training loss ─────────────────────────────────────
    fig, axes = plt.subplots(1, n_out + 1, figsize=(6 * (n_out + 1), 5))

    for i in range(n_out):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], alpha=0.3, s=5)
        lims = [min(true[:, i].min(), pred[:, i].min()),
                max(true[:, i].max(), pred[:, i].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"True {OUTPUT_NAMES[i]}")
        ax.set_ylabel(f"Predicted {OUTPUT_NAMES[i]}")
        ax.set_title(f"{OUTPUT_NAMES[i]}: Predicted vs True")

    ax = axes[-1]
    for i, h in enumerate(loss_histories):
        ax.plot(h, label=OUTPUT_NAMES[i])
    ax.set_xlabel("L-BFGS epoch")
    ax.set_ylabel("Neg. lower bound")
    ax.set_title("Training loss")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_diagnostics.png"), dpi=150)
    plt.close(fig)
    log.info("Saved plot to %s/training_diagnostics.png", output_dir)

    # ── ARD length scales ────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, n_out, figsize=(7 * n_out, 5))
    if n_out == 1:
        axes2 = [axes2]
    for i, m in enumerate(models):
        ls = torch.exp(m.log_lengthscales).detach().cpu().numpy()
        x_pos = np.arange(len(input_names))
        axes2[i].bar(x_pos, ls)
        axes2[i].set_xticks(x_pos)
        axes2[i].set_xticklabels(input_names, rotation=45, ha="right")
        axes2[i].set_ylabel("Length scale")
        axes2[i].set_title(f"ARD Length Scales: {OUTPUT_NAMES[i]}")

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "ard_lengthscales.png"), dpi=150)
    plt.close(fig2)
    log.info("Saved ARD plot to %s/ard_lengthscales.png", output_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="SGPR (Krasserm VFE) training for action prior")
    ap.add_argument("--data-dir", default="results/",
                    help="Dir with trajectory CSVs")
    ap.add_argument("--output-dir",
                    default="models/sgpr_krasserm/",
                    help="Dir to save trained model")
    ap.add_argument("--num-inducing", type=int, default=128,
                    help="Number of inducing points")
    ap.add_argument("--epochs", type=int, default=100,
                    help="L-BFGS epochs")
    ap.add_argument("--lr", type=float, default=0.1,
                    help="L-BFGS learning rate")
    ap.add_argument("--test-split", type=float, default=0.2,
                    help="Fraction of data for test set")
    ap.add_argument("--init-lengthscale", type=float, default=1.0,
                    help="Initial ARD RBF length scale")
    ap.add_argument("--init-outputscale", type=float, default=1.0,
                    help="Initial kernel output scale")
    ap.add_argument("--init-noise", type=float, default=0.1,
                    help="Initial observation noise")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--plot", action="store_true",
                    help="Generate diagnostic plots")
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

    # ── 1. Load data ─────────────────────────────────────────────────────
    data = load_data(args.data_dir, input_cols)

    X_np = data[input_cols].values.astype(np.float64)
    Y_np = data[OUTPUT_COLS].values.astype(np.float64)
    log.info("Features: %s  Targets: %s", X_np.shape, Y_np.shape)

    # ── 2. Train/test split ──────────────────────────────────────────────
    X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
        X_np, Y_np, test_size=args.test_split, random_state=args.seed)
    log.info("Train: %d  Test: %d", len(X_train_np), len(X_test_np))

    # ── 3. Normalise ─────────────────────────────────────────────────────
    x_scaler = StandardScaler().fit(X_train_np)
    y_scaler = StandardScaler().fit(Y_train_np)

    X_train = torch.tensor(x_scaler.transform(X_train_np), dtype=DTYPE)
    X_test = torch.tensor(x_scaler.transform(X_test_np), dtype=DTYPE)
    Y_train = torch.tensor(y_scaler.transform(Y_train_np), dtype=DTYPE)
    Y_test = torch.tensor(y_scaler.transform(Y_test_np), dtype=DTYPE)

    # ── 4. Inducing points ───────────────────────────────────────────────
    X_m_init = select_inducing_points(X_train.numpy(), args.num_inducing)

    # ── 5. Train independent GPs ─────────────────────────────────────────
    models = []
    loss_histories = []
    total_time = 0.0

    for i in range(2):
        log.info("=" * 50)
        log.info("Training GP for %s ...", OUTPUT_NAMES[i])
        log.info("=" * 50)

        model = SparseGPR(
            X_m_init=X_m_init.clone(),
            input_dim=X_train.shape[1],
            init_lengthscale=args.init_lengthscale,
            init_outputscale=args.init_outputscale,
            init_noise=args.init_noise,
        )

        t0 = time.monotonic()
        hist = train_sgpr(model, X_train, Y_train[:, i],
                          epochs=args.epochs, lr=args.lr)
        elapsed = time.monotonic() - t0
        total_time += elapsed
        log.info("  Trained in %.1fs", elapsed)

        model.cache_posterior(X_train, Y_train[:, i])
        models.append(model)
        loss_histories.append(hist)

    # ── 6. Evaluate ──────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Evaluation on test set")
    log.info("=" * 50)
    metrics, pred, pred_std, true = evaluate(models, X_test, Y_test, y_scaler)

    log.info("=== Test Metrics (original scale) ===")
    log.info("  %-25s  %-12s  %-12s", "", "Acceleration", "Steering Vel")
    log.info("  %-25s  %-12.4f  %-12.4f", "MAE",
             metrics["mae"][0], metrics["mae"][1])
    log.info("  %-25s  %-12.4f  %-12.4f", "RMSE",
             metrics["rmse"][0], metrics["rmse"][1])
    log.info("  %-25s  %-11.1f%%  %-11.1f%%", "Coverage (+/- 2std)",
             metrics["coverage"][0], metrics["coverage"][1])

    # Print ARD length scales
    log.info("=== ARD Length Scales ===")
    for i, m in enumerate(models):
        ls = torch.exp(m.log_lengthscales).detach().cpu().numpy()
        log.info("  %s:", OUTPUT_NAMES[i])
        for j, name in enumerate(input_names):
            log.info("    %-25s  %.4f", name, ls[j])

    # ── 7. Save ──────────────────────────────────────────────────────────
    config = {
        "model_type": "sgpr_krasserm",
        "input_cols": input_cols,
        "output_cols": OUTPUT_COLS,
        "input_names": input_names,
        "output_names": OUTPUT_NAMES,
        "feature_set": args.feature_set,
        "num_input_dims": len(input_cols),
        "num_outputs": 2,
        "num_inducing": args.num_inducing,
        "epochs": args.epochs,
        "lr": args.lr,
        "init_lengthscale": args.init_lengthscale,
        "init_outputscale": args.init_outputscale,
        "init_noise": args.init_noise,
        "seed": args.seed,
        "test_split": args.test_split,
        "n_train": len(X_train_np),
        "n_test": len(X_test_np),
        "train_time_s": round(total_time, 2),
        "mae_accel": float(metrics["mae"][0]),
        "mae_steer": float(metrics["mae"][1]),
        "rmse_accel": float(metrics["rmse"][0]),
        "rmse_steer": float(metrics["rmse"][1]),
        "coverage_2std_accel": float(metrics["coverage"][0]),
        "coverage_2std_steer": float(metrics["coverage"][1]),
    }
    save_model(args.output_dir, models, x_scaler, y_scaler, config)

    # ── 8. Plot (optional) ───────────────────────────────────────────────
    if args.plot:
        plot_results(pred, pred_std, true, loss_histories, models,
                     args.output_dir, input_names=input_names)

    log.info("Done.")


if __name__ == "__main__":
    main()
