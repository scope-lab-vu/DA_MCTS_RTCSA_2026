"""
Reduce the MCTS action probability dataset via bin-and-deduplicate.

Strategy:
  - n_iter, max_depth : discrete, keep as-is
  - t_remain_s        : quantize to N_T bins
  - v                 : quantize to N_V bins
  - kappa             : quantize to N_K bins

Within each (n_iter, max_depth, t_bin, v_bin, kappa_bin) group the targets
are averaged, giving one representative row per unique input region.

Output: MCTS_action_prob_MLP/data/dataset_reduced.csv
"""

import glob
import numpy as np
import pandas as pd

DATA_DIR  = "MCTS_action_prob_MLP/data"
OUT_FILE  = "MCTS_action_prob_MLP/data/dataset_reduced.csv"

# Bin counts for continuous features
N_T = 15    # t_remain_s  (0 – 0.03 s  → ~2 ms resolution)
N_V = 10    # v            (1.5 – 12 m/s → ~1.0 m/s resolution)
N_K = 10    # kappa        (0 – 0.79 rad/m → ~0.08 rad/m resolution)

FEATURES = ["n_iter", "t_remain_s", "v", "kappa", "max_depth"]
TARGETS  = ["p_agg", "p_slow", "p_feas"]

# ── 1. Load ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(f"{DATA_DIR}/mcts_weight_log_*.csv"))
print(f"Loading {len(files)} files …")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  Raw rows: {len(df):,}")

# ── 2. Bin continuous features ─────────────────────────────────────────────
df["t_bin"] = pd.cut(df["t_remain_s"], bins=N_T, labels=False)
df["v_bin"] = pd.cut(df["v"],          bins=N_V, labels=False)
df["k_bin"] = pd.cut(df["kappa"],      bins=N_K, labels=False)

# ── 3. Group & average targets ─────────────────────────────────────────────
group_keys = ["n_iter", "max_depth", "t_bin", "v_bin", "k_bin"]
agg = (
    df.groupby(group_keys, observed=True)
      .agg(
          t_remain_s=("t_remain_s", "mean"),
          v=("v", "mean"),
          kappa=("kappa", "mean"),
          p_agg=("p_agg", "mean"),
          p_slow=("p_slow", "mean"),
          p_feas=("p_feas", "mean"),
      )
      .reset_index()
)

# ── 4. Keep only the original columns in original order ────────────────────
out = agg[FEATURES + TARGETS]
print(f"  Reduced rows: {len(out):,}  ({len(out)/len(df)*100:.1f}% of original)")

# ── 5. Save ────────────────────────────────────────────────────────────────
out.to_csv(OUT_FILE, index=False)
print(f"  Saved → {OUT_FILE}")
