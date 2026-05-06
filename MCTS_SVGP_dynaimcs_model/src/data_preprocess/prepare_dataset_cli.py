"""
Generate train/val/test SVGP splits from a raw trajectory CSV.

Two split modes:
  - canonical:  hardcoded train=1737, val=217, test=rest. Used by the original
                traj_MPPI_*_mu_*.csv reference datasets to reproduce the
                checkpoints in data/models_verify/.
  - fractional: 75% train / 10% val / 15% test (default for any non-canonical CSV).

Auto-detection rule: CSVs whose basename starts with "traj_MPPI_" are treated as
canonical (Ts=0.02, hardcoded sizes). Anything else falls back to fractional.
Override with --split-mode {canonical,fractional}.

Example:
    python prepare_dataset_cli.py \\
        --csv ../../data/raw/traj_MPPI_YasMarina_mu_1.0489.csv \\
        --map YasMarina --mu 1.0489 --ts 0.02 --varidx 5

    python prepare_dataset_cli.py \\
        --csv ../../data/raw/stanley_dt0.1_dataset/dataset_YasMarina_mu10489.csv \\
        --map YasMarina --mu 1.0489 --ts 0.1 --varidx 5 --tag stanley
"""

import argparse
import datetime as dt
import json
import os
import _pickle as pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import f110
from ext_kinematic_casadi import Ekinematic
from prepare_dataset import prepare_dataset_from_raw_pure_pursuit_df_input

Y_NAMES = ["Y_x", "Y_y", "Y_delta", "Y_vel", "Y_psi", "Y_omega", "Y_beta"]
X_FEATURES = ["S_delta", "S_vel", "S_omega", "S_beta", "A_a", "A_sv", "C_alpha_smooth"]
CANONICAL_TRAIN = 1737
CANONICAL_VAL = 217


def auto_split_mode(csv_path: str) -> str:
    return "canonical" if Path(csv_path).name.startswith("traj_MPPI_") else "fractional"


def split_indices(n_total, mode, train_frac, val_frac):
    if mode == "canonical":
        if n_total < CANONICAL_TRAIN + CANONICAL_VAL:
            raise ValueError(
                f"Canonical split needs >={CANONICAL_TRAIN + CANONICAL_VAL} rows, got {n_total}"
            )
        return CANONICAL_TRAIN, CANONICAL_VAL, n_total - CANONICAL_TRAIN - CANONICAL_VAL
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"train_frac+val_frac too large; no test rows left ({n_total} rows)")
    return n_train, n_val, n_test


def load_and_residualize(csv_path: str, sampling_time: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    if "C_alpha_smooth" in df.columns:
        df = df.dropna(subset=["C_alpha_smooth"]).reset_index(drop=True)
    return prepare_dataset_from_raw_pure_pursuit_df_input(df, sampling_time)


def build_xy(df: pd.DataFrame, varidx: int):
    X = df[X_FEATURES].values
    y = df[Y_NAMES[varidx]].values.reshape(-1, 1)
    return X, y


def fit_scalers(X, y):
    xs, ys = StandardScaler(), StandardScaler()
    xs.fit(X)
    ys.fit(y)
    return xs, ys


def output_dir(out_root: str, map_name: str, varidx: int, ts: float, mu: float, tag: str):
    parts = [f"{map_name}_VAR{varidx}_Ts{ts:.3f}"]
    if tag:
        parts.append(tag)
    # Match the original convention: omit mu suffix only for the canonical anchor
    is_canonical_anchor = (map_name == "YasMarina" and abs(mu - 1.0489) < 1e-9 and not tag)
    if not is_canonical_anchor:
        parts.append(f"mu_{mu}")
    return os.path.join(out_root, "_".join(parts))


def save_outputs(base_dir, X_train, y_train, X_val, y_val, X_test, y_test, xscaler, yscaler, meta):
    os.makedirs(base_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(base_dir, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        x_scaler_mean=xscaler.mean_, x_scaler_std=xscaler.scale_,
        y_scaler_mean=yscaler.mean_, y_scaler_std=yscaler.scale_,
    )
    with open(os.path.join(base_dir, "xscaler.pkl"), "wb") as f:
        pickle.dump(xscaler, f)
    with open(os.path.join(base_dir, "yscaler.pkl"), "wb") as f:
        pickle.dump(yscaler, f)
    with open(os.path.join(base_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to raw trajectory CSV")
    ap.add_argument("--map", required=True, help="Map name (e.g. YasMarina)")
    ap.add_argument("--mu", type=float, required=True, help="Friction coefficient")
    ap.add_argument("--ts", type=float, required=True, help="Sampling time used by the kinematic integrator")
    ap.add_argument("--varidx", type=int, required=True, choices=range(7),
                    help="Target index: 0=x 1=y 2=delta 3=vel 4=psi 5=omega 6=beta")
    ap.add_argument("--out-root", default="../../data/splits", help="Output root for splits")
    ap.add_argument("--tag", default="", help="Optional tag injected into the output dir name")
    ap.add_argument("--split-mode", choices=["auto", "canonical", "fractional"], default="auto")
    ap.add_argument("--train-frac", type=float, default=0.75, help="Fractional mode: train fraction")
    ap.add_argument("--val-frac", type=float, default=0.10, help="Fractional mode: val fraction")
    args = ap.parse_args()

    mode = auto_split_mode(args.csv) if args.split_mode == "auto" else args.split_mode
    print(f"[mode={mode}] csv={args.csv}")

    df = load_and_residualize(args.csv, args.ts)
    X, y = build_xy(df, args.varidx)
    print(f"After residualization: {X.shape[0]} rows, {X.shape[1]} features")

    xscaler, yscaler = fit_scalers(X, y)
    Xs = xscaler.transform(X)
    ys = yscaler.transform(y)

    n_train, n_val, n_test = split_indices(Xs.shape[0], mode, args.train_frac, args.val_frac)
    X_train, y_train = Xs[:n_train], ys[:n_train]
    X_val, y_val = Xs[n_train:n_train + n_val], ys[n_train:n_train + n_val]
    X_test, y_test = Xs[n_train + n_val:], ys[n_train + n_val:]
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    base_dir = output_dir(args.out_root, args.map, args.varidx, args.ts, args.mu, args.tag)
    meta = {
        "map_name": args.map,
        "VARIDX": args.varidx,
        "Ts": args.ts,
        "mu": args.mu,
        "tag": args.tag,
        "split_mode": mode,
        "split_fracs": {"train": args.train_frac, "val": args.val_frac} if mode == "fractional" else None,
        "sizes": {"X": int(X.shape[0]), "train": n_train, "val": n_val, "test": n_test},
        "source_csv": str(Path(args.csv).resolve()),
        "saved_at": dt.datetime.now().isoformat(),
    }
    save_outputs(base_dir, X_train, y_train, X_val, y_val, X_test, y_test, xscaler, yscaler, meta)
    print(f"Wrote: {base_dir}")


if __name__ == "__main__":
    main()
