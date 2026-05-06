"""
Verify two SVGP training runs produce equivalent models.

Compares for each (VAR5/omega, VAR6/beta) the model in
    data/models/                (the existing/production version)
vs the freshly-trained sibling in
    data/models_verify/

Test inputs come from data/splits_verify/<TRACK>_VAR{5,6}_Ts0.020/splits.npz
(`X_test`). We re-implement the Krasser-SVGP posterior in pure numpy from
the .npz fields so the script doesn't depend on the training stack.

For each model pair we report:
    - max abs error / mean abs error / max relative error of mean predictions
    - mean variance ratio (new / old)
    - hyperparameter diffs (lengthscales, σ_f, scaler mean/scale, inducing points)

Run:
    python MCTS_SVGP_dynaimcs_model/src/verify/verify_models.py
    python MCTS_SVGP_dynaimcs_model/src/verify/verify_models.py --tol 1e-6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]   # MCTS_SVGP_dynaimcs_model/
MODELS_OLD = PROJECT / "data" / "models"
MODELS_NEW = PROJECT / "data" / "models_verify"
SPLITS = PROJECT / "data" / "splits_verify"

# What to compare. (var_idx, name, M, split_dir).
PAIRS = [
    (5, "omega", 11, "YasMarina_VAR5_Ts0.020"),
    (6, "beta",  12, "YasMarina_VAR6_Ts0.020"),
]
KERNEL_TAG = "ard_kernel_c_alpha_smooth"   # both old and new have this variant
TS_TAG = "Ts0.020"
TRACK = "YasMarina"


# ── Krasser-SVGP I/O + predict (pure numpy, ARD kernel) ──────────────────
def load_model(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "theta":     np.asarray(d["model__theta"], dtype=np.float64),
        "X_m":       np.asarray(d["model__X_m"], dtype=np.float64),
        "mu_m":      np.asarray(d["model__mu_m"], dtype=np.float64),
        "A_m":       np.asarray(d["model__A_m"], dtype=np.float64),
        "K_mm_inv":  np.asarray(d["model__K_mm_inv"], dtype=np.float64),
        "x_mean":    np.asarray(d["xscaler__mean"], dtype=np.float64),
        "x_scale":   np.asarray(d["xscaler__scale"], dtype=np.float64),
        "y_mean":    np.asarray(d["yscaler__mean"], dtype=np.float64),
        "y_scale":   np.asarray(d["yscaler__scale"], dtype=np.float64),
    }


def predict(model: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean + variance for the Krasser SVGP at points X (raw, not pre-scaled).

    Mirrors the inference path used by `_load_krasser_model` + the C++
    runtime: ARD kernel, standard sparse-GP variational predictive.
    """
    X = np.atleast_2d(X).astype(np.float64)
    Xs = (X - model["x_mean"]) / model["x_scale"]                          # (N, D)
    X_m = model["X_m"]                                                     # (M, D)
    D = X_m.shape[1]
    theta = model["theta"]
    lengthscales = theta[:D]                                               # (D,)
    sigma_f2 = float(theta[D]) ** 2 if theta.shape[0] == D + 1 else float(theta[D])
    # ARD-RBF kernel (training pickles store sigma_f, not sigma_f² — square it).
    # See controller.py: theta[D] is treated as σ_f directly. Keep parity:
    sigma_f = float(theta[D])

    # k(X*, X_m): (N, M)
    diff = (Xs[:, None, :] - X_m[None, :, :]) / lengthscales[None, None, :]
    sq = np.sum(diff * diff, axis=-1)
    K_sm = (sigma_f ** 2) * np.exp(-0.5 * sq)

    K_mm_inv = model["K_mm_inv"]
    A_m = model["A_m"]
    alpha = K_mm_inv @ model["mu_m"]                                       # (M, 1)
    B_m = K_mm_inv @ A_m @ K_mm_inv - K_mm_inv                              # (M, M)

    mean_norm = (K_sm @ alpha).ravel()                                      # (N,)
    var_norm = (sigma_f ** 2) + np.einsum("nm,mk,nk->n", K_sm, B_m, K_sm)
    var_norm = np.maximum(var_norm, 0.0)

    y_mean = float(model["y_mean"].ravel()[0])
    y_scale = float(model["y_scale"].ravel()[0])
    mean = mean_norm * y_scale + y_mean
    var = var_norm * (y_scale ** 2)
    return mean, var


# ── Reporting helpers ────────────────────────────────────────────────────
def fmt_array_diff(label: str, a: np.ndarray, b: np.ndarray, indent: int = 4) -> str:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    diff = np.abs(a - b)
    rel = diff / (np.maximum(np.abs(a), np.abs(b)) + 1e-30)
    pad = " " * indent
    return (
        f"{pad}{label:<14} shape={a.shape}  "
        f"max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}  "
        f"max rel={rel.max():.3e}"
    )


def compare_pair(var_idx: int, name: str, M: int, split_dir: str,
                 tol: float, verbose: bool) -> bool:
    base = f"krasser_svgp_{name}_{TRACK}_VAR{var_idx}_jax_{TS_TAG}_MPPI_data_{KERNEL_TAG}_M_{M}.npz"
    old_path = MODELS_OLD / base
    new_path = MODELS_NEW / base

    print(f"\n{'='*78}")
    print(f"VAR{var_idx} ({name})  M={M}")
    print(f"  old : {old_path.relative_to(PROJECT)}")
    print(f"  new : {new_path.relative_to(PROJECT)}")
    print('='*78)

    if not old_path.exists():
        print(f"  ❌ missing: {old_path}")
        return False
    if not new_path.exists():
        print(f"  ❌ missing: {new_path}")
        return False

    old = load_model(old_path)
    new = load_model(new_path)

    # Hyperparameter / scaler diffs (sanity checks before prediction).
    print("  Parameter diffs:")
    for key in ("theta", "X_m", "mu_m", "A_m", "K_mm_inv",
                "x_mean", "x_scale", "y_mean", "y_scale"):
        if old[key].shape != new[key].shape:
            print(f"    {key:<12}  ⚠ SHAPE MISMATCH  old={old[key].shape}  new={new[key].shape}")
            continue
        print("  " + fmt_array_diff(key, old[key], new[key]))

    # Predict on the test split.
    splits_path = SPLITS / split_dir / "splits.npz"
    if not splits_path.exists():
        print(f"  ❌ no splits at {splits_path}")
        return False
    s = np.load(splits_path, allow_pickle=True)
    X_test = np.asarray(s["X_test"], dtype=np.float64)
    y_test = np.asarray(s["y_test"], dtype=np.float64).ravel()

    mean_old, var_old = predict(old, X_test)
    mean_new, var_new = predict(new, X_test)

    diff = np.abs(mean_old - mean_new)
    rel = diff / (np.maximum(np.abs(mean_old), np.abs(mean_new)) + 1e-30)
    var_ratio = var_new / (var_old + 1e-30)

    print(f"\n  Predictions on X_test (N={len(X_test)}):")
    print(f"    mean    max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}  "
          f"max rel={rel.max():.3e}")
    print(f"    var     ratio new/old: min={var_ratio.min():.4f}  "
          f"max={var_ratio.max():.4f}  mean={var_ratio.mean():.4f}")
    # Sanity: how well does each model fit the test labels?
    rmse_old = float(np.sqrt(np.mean((mean_old - y_test) ** 2)))
    rmse_new = float(np.sqrt(np.mean((mean_new - y_test) ** 2)))
    print(f"    RMSE vs y_test:  old={rmse_old:.6f}   new={rmse_new:.6f}   "
          f"Δ={rmse_new - rmse_old:+.3e}")

    if verbose:
        # First 5 side-by-side predictions.
        print(f"\n  First 5 predictions (mean):")
        print(f"    {'i':>3}  {'old':>14}  {'new':>14}  {'|Δ|':>10}  {'y_test':>14}")
        for i in range(min(5, len(X_test))):
            print(f"    {i:>3}  {mean_old[i]:>14.8f}  {mean_new[i]:>14.8f}  "
                  f"{abs(mean_old[i]-mean_new[i]):>10.2e}  {y_test[i]:>14.8f}")

    ok = bool(diff.max() <= tol)
    print(f"\n  → {'PASS' if ok else 'FAIL'}  (tol={tol:.0e}, max|Δ|={diff.max():.3e})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Pass tolerance on max|Δmean| across the test split.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print side-by-side predictions for the first 5 points.")
    args = p.parse_args()

    print(f"PROJECT root: {PROJECT}")
    print(f"comparing:    {MODELS_OLD.relative_to(PROJECT)}  vs  {MODELS_NEW.relative_to(PROJECT)}")

    results = []
    for var_idx, name, M, split_dir in PAIRS:
        results.append(compare_pair(var_idx, name, M, split_dir,
                                    args.tol, args.verbose))

    print(f"\n{'='*78}")
    print(f"OVERALL: {sum(results)}/{len(results)} model pairs PASSED at tol={args.tol:.0e}")
    print('='*78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
