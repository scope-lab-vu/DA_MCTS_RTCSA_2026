"""
Evaluation script for the trained action-probability MLP.

Loads architecture from best_hp.json and weights from model_best.pt.

1. GPU evaluation  — MAE / MSE on ALL raw data files
2. CPU benchmark   — single-sample inference time over 10,000 real data points
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_FILE  = Path("MCTS_action_prob_MLP/model_best.pt")
SCALER_FILE = Path("MCTS_action_prob_MLP/scaler.json")
HP_FILE     = Path("MCTS_action_prob_MLP/best_hp.json")
DATA_DIR    = Path("MCTS_action_prob_MLP/data")

FEATURES    = ["n_iter", "t_remain_s", "v", "kappa", "max_depth"]
TARGETS     = ["p_agg", "p_slow", "p_feas"]
CHUNK       = 50_000
N_CPU_BENCH = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ── Model definition ────────────────────────────────────────────────────────
class ProbMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super().__init__()
        dims   = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

# ── Load best HPs (for reference only) ─────────────────────────────────────
with open(HP_FILE) as f:
    hp = json.load(f)
print(f"Best HP search result : val_mse={hp['val_mse']:.6f}")
print(f"  lr={hp['lr']:.2e}, batch_size={hp['batch_size']}, "
      f"n_layers={hp['n_layers']}, hidden_dim={hp['hidden_dim']}, dropout={hp.get('dropout',0)}")

# ── Load scaler ─────────────────────────────────────────────────────────────
with open(SCALER_FILE) as f:
    scaler = json.load(f)
mu  = np.array(scaler["mean"], dtype=np.float32)
std = np.array(scaler["std"],  dtype=np.float32)

# ── Infer architecture from state dict, then load weights ───────────────────
sd = torch.load(MODEL_FILE, weights_only=True)
# Linear layers appear as net.0.weight, net.2.weight, ... — extract output dims
linear_weights = [v for k, v in sd.items() if k.endswith(".weight") and "net." in k]
hidden_dims    = [w.shape[0] for w in linear_weights[:-1]]  # all but output layer
dropout        = hp.get("dropout", 0.0)

model = ProbMLP(len(FEATURES), hidden_dims, len(TARGETS), dropout)
model.load_state_dict(sd)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nArchitecture (from checkpoint) : {len(FEATURES)} → {hidden_dims} → {len(TARGETS)}")
print(f"Parameters : {n_params:,}")

# ── 1. GPU evaluation on ALL raw data ───────────────────────────────────────
print("\n─── GPU evaluation (all raw data) ───────────────────────────────────")
model_gpu = model.to(DEVICE)
model_gpu.eval()

all_files  = sorted(DATA_DIR.glob("mcts_weight_log_*.csv"))
abs_errs   = {t: 0.0 for t in TARGETS}
sq_errs    = {t: 0.0 for t in TARGETS}
total_rows = 0

with torch.no_grad():
    for fpath in all_files:
        raw  = pd.read_csv(fpath)
        Xr   = (raw[FEATURES].values.astype(np.float32) - mu) / std
        Yr   = raw[TARGETS].values.astype(np.float32)
        for start in range(0, len(Xr), CHUNK):
            xb   = torch.from_numpy(Xr[start:start + CHUNK]).to(DEVICE)
            yb   = Yr[start:start + CHUNK]
            yhat = model_gpu(xb).cpu().numpy()
            err  = yhat - yb
            for i, t in enumerate(TARGETS):
                abs_errs[t] += np.abs(err[:, i]).sum()
                sq_errs[t]  += (err[:, i] ** 2).sum()
        total_rows += len(raw)

print(f"Total rows : {total_rows:,}")
print(f"\n{'Output':<10}  {'MAE':>10}  {'RMSE':>10}")
print("-" * 34)
for t in TARGETS:
    mae  = abs_errs[t] / total_rows
    rmse = np.sqrt(sq_errs[t] / total_rows)
    print(f"{t:<10}  {mae:>10.5f}  {rmse:>10.5f}")

# ── 2. CPU inference benchmark — batch_size=1, 10 000 real samples ──────────
print(f"\n─── CPU inference benchmark (batch=1, n={N_CPU_BENCH:,}) ──────────────────")
model_cpu = ProbMLP(len(FEATURES), hidden_dims, len(TARGETS), dropout=0.0)
model_cpu.load_state_dict(sd)
model_cpu.eval()

# Load 10 000 samples from reduced dataset
df_bench  = pd.read_csv(DATA_DIR / "dataset_reduced.csv")
bench_raw = df_bench[FEATURES].values[:N_CPU_BENCH].astype(np.float32)
bench_X   = torch.from_numpy((bench_raw - mu) / std)   # CPU tensor

# Warm-up
with torch.no_grad():
    for i in range(50):
        _ = model_cpu(bench_X[i].unsqueeze(0))

# Timed run — one sample at a time
times = np.empty(N_CPU_BENCH)
with torch.no_grad():
    for i in range(N_CPU_BENCH):
        t0       = time.perf_counter()
        _        = model_cpu(bench_X[i].unsqueeze(0))
        times[i] = (time.perf_counter() - t0) * 1e6   # µs

print(f"{'mean'  :>8} : {times.mean():.2f} µs")
print(f"{'median':>8} : {np.median(times):.2f} µs")
print(f"{'p95'   :>8} : {np.percentile(times, 95):.2f} µs")
print(f"{'p99'   :>8} : {np.percentile(times, 99):.2f} µs")
print(f"{'max'   :>8} : {times.max():.2f} µs")
print(f"\n  → {1e6 / times.mean():,.0f} inferences/s  (mean)")
