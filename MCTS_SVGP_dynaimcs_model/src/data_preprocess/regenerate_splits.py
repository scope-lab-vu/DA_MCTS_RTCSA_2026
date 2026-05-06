"""
Regenerate train/val/test splits for all (VARIDX, mu) combinations.

Uses C_alpha_smooth as the 7th input feature (via updated prepare_dataset.py).

Usage:
    cd svgp_martin_krasser_c_alpha_smooth/gp/
    python regenerate_splits.py
"""

import os
import json
import datetime as dt
import _pickle as pickle

import numpy as np
from prepare_dataset import load_data

MAP_NAME = 'YasMarina'
SAMPLING_TIME = 0.02
SEED = 42

# All (VARIDX, mu) combinations to generate
VARIDX_LIST = [5, 6]
MU_LIST = [1.0489, 0.8642, 0.4947]


def make_splits(varidx, mu):
    """Load data, split, and save splits/scalers/meta."""
    print(f"\n{'='*60}")
    print(f"VARIDX={varidx}, mu={mu}")
    print(f"{'='*60}")

    X, y, xscaler, yscaler = load_data(
        MAP_NAME, varidx, SAMPLING_TIME, xscaler=None, yscaler=None, mu=mu
    )

    # Same split sizes as the original notebook
    X_train, y_train = X[:1737], y[:1737]
    X_val, y_val = X[1737:1737+217], y[1737:1737+217]
    X_test, y_test = X[1737+217:], y[1737+217:]

    print(f"  Total: {X.shape[0]}, Train: {X_train.shape[0]}, "
          f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Directory
    if mu == 1.0489 and MAP_NAME == 'YasMarina':
        base_dir = os.path.join("data", f"{MAP_NAME}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}")
    else:
        base_dir = os.path.join("data", f"{MAP_NAME}_VAR{varidx}_Ts{SAMPLING_TIME:.3f}_mu_{mu}")
    os.makedirs(base_dir, exist_ok=True)

    # Save splits
    np.savez_compressed(
        os.path.join(base_dir, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        x_scaler_mean=xscaler.mean_,
        x_scaler_std=xscaler.scale_,
        y_scaler_mean=yscaler.mean_,
        y_scaler_std=yscaler.scale_,
    )

    # Save scalers
    with open(os.path.join(base_dir, "xscaler.pkl"), "wb") as f:
        pickle.dump(xscaler, f)
    with open(os.path.join(base_dir, "yscaler.pkl"), "wb") as f:
        pickle.dump(yscaler, f)

    # Save metadata
    with open(os.path.join(base_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "map_name": MAP_NAME,
                "VARIDX": varidx,
                "mu": mu,
                "Ts": SAMPLING_TIME,
                "c_alpha_feature": "C_alpha_smooth",
                "split": {"train": 1737, "val": 217, "test": "rest"},
                "sizes": {
                    "X": int(X.shape[0]),
                    "train": int(X_train.shape[0]),
                    "val": int(X_val.shape[0]),
                    "test": int(X_test.shape[0]),
                },
                "saved_at": dt.datetime.now().isoformat(),
            },
            f, indent=2,
        )

    print(f"  Saved to: {base_dir}/")

    # Quick sanity check: 7th feature (C_alpha_smooth) stats in original scale
    X_orig = xscaler.inverse_transform(X)
    c_alpha_col = X_orig[:, 6]
    print(f"  C_alpha_smooth (original scale): "
          f"mean={c_alpha_col.mean():.2f}, std={c_alpha_col.std():.2f}, "
          f"min={c_alpha_col.min():.2f}, max={c_alpha_col.max():.2f}")


def main():
    print("Regenerating splits with C_alpha_smooth feature")
    print(f"Map: {MAP_NAME}, Ts: {SAMPLING_TIME}")

    for varidx in VARIDX_LIST:
        for mu in MU_LIST:
            make_splits(varidx, mu)

    print(f"\nDone. Generated {len(VARIDX_LIST) * len(MU_LIST)} split directories.")


if __name__ == '__main__':
    main()
