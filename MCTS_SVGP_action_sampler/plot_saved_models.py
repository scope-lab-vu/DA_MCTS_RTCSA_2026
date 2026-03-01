#!/usr/bin/env python3
"""Generate ARD lengthscale and prediction diagnostic plots for saved SGPR models."""

import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_krasserm import (
    DTYPE, OUTPUT_COLS, OUTPUT_NAMES,
    get_input_features, load_data, load_model,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_DIRS = {
    "no_avg_kappa": "models/sgpr_krasserm_no_avg_kappa",
    "no_lateral":   "models/sgpr_krasserm_no_lateral",
    "general":      "models/sgpr_krasserm_general",
}
DATA_DIR = "MCTS_SVGP_action_sampler/training_data"
SEED = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2


def generate_plots(feature_set, model_dir):
    print(f"\n{'='*60}")
    print(f"  Generating plots for: {feature_set} ({model_dir})")
    print(f"{'='*60}")

    models, x_scaler, y_scaler, config = load_model(model_dir)
    input_cols = config["input_cols"]
    input_names = config["input_names"]
    n_out = len(models)

    # Load and split data identically to hp_search
    data = load_data(DATA_DIR, input_cols)
    X_all = data[input_cols].values.astype(np.float64)
    Y_all = data[OUTPUT_COLS].values.astype(np.float64)

    X_trainval, X_test_np, Y_trainval, Y_test_np = train_test_split(
        X_all, Y_all, test_size=TEST_SPLIT, random_state=SEED)
    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X_trainval, Y_trainval, test_size=VAL_SPLIT, random_state=SEED)

    # Refit scalers on train+val (matches hp_search retrain phase)
    X_tv = np.concatenate([X_train_np, X_val_np], axis=0)
    Y_tv = np.concatenate([Y_train_np, Y_val_np], axis=0)
    x_sc = StandardScaler().fit(X_tv)
    y_sc = StandardScaler().fit(Y_tv)

    X_test = torch.tensor(x_sc.transform(X_test_np), dtype=DTYPE)

    # Predictions
    all_pred, all_std, all_true = [], [], []
    for i, m in enumerate(models):
        f_mean, _, y_var = m.predict(X_test)
        mean_orig = f_mean.cpu().numpy() * y_sc.scale_[i] + y_sc.mean_[i]
        std_orig = np.sqrt(y_var.cpu().numpy()) * y_sc.scale_[i]
        y_true = Y_test_np[:, i]
        all_pred.append(mean_orig)
        all_std.append(std_orig)
        all_true.append(y_true)

        err = np.abs(mean_orig - y_true)
        print(f"  {OUTPUT_NAMES[i]}: MAE={np.mean(err):.4f}  "
              f"RMSE={np.sqrt(np.mean(err**2)):.4f}  "
              f"Coverage={np.mean(np.abs(y_true - mean_orig) <= 2*std_orig)*100:.1f}%")

    pred = np.column_stack(all_pred)
    pred_std = np.column_stack(all_std)
    true = np.column_stack(all_true)

    # ── Pred vs True ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_out, figsize=(6 * n_out, 5))
    if n_out == 1:
        axes = [axes]
    for i in range(n_out):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], alpha=0.3, s=5)
        lims = [min(true[:, i].min(), pred[:, i].min()),
                max(true[:, i].max(), pred[:, i].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"True {OUTPUT_NAMES[i]}")
        ax.set_ylabel(f"Predicted {OUTPUT_NAMES[i]}")
        ax.set_title(f"{OUTPUT_NAMES[i]}: Predicted vs True")
    fig.suptitle(f"Training Diagnostics — {feature_set}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(model_dir, "training_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── ARD Length Scales ─────────────────────────────────────────────────
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
    fig2.suptitle(f"ARD Length Scales — {feature_set}", fontsize=14, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(model_dir, "ard_lengthscales.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")


if __name__ == "__main__":
    for fs, mdir in MODEL_DIRS.items():
        if os.path.isdir(mdir):
            generate_plots(fs, mdir)
        else:
            print(f"Skipping {fs}: {mdir} not found")
    print("\nDone.")
