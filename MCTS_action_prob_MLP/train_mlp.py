"""
Train the action-probability MLP using the best HPs from best_hp.json.

Outputs sum to 1  →  softmax final layer.
Saves model_best.pt (state dict) + scaler.json.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Config ─────────────────────────────────────────────────────────────────
DATA_FILE  = Path("MCTS_action_prob_MLP/data/dataset_reduced.csv")
HP_FILE    = Path("MCTS_action_prob_MLP/best_hp.json")
OUT_DIR    = Path("MCTS_action_prob_MLP")

EPOCHS     = 100
VAL_FRAC   = 0.20
PATIENCE   = 10
SEED       = 42

FEATURES   = ["n_iter", "t_remain_s", "v", "kappa", "max_depth"]
TARGETS    = ["p_agg", "p_slow", "p_feas"]

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load best HPs ──────────────────────────────────────────────────────────
with open(HP_FILE) as f:
    hp = json.load(f)

HIDDEN  = [hp["hidden_dim"]] * hp["n_layers"]
LR      = hp["lr"]
BATCH   = hp["batch_size"]
DROPOUT = hp.get("dropout", 0.0)

print(f"Loaded HPs from {HP_FILE}")
print(f"  Architecture : {len(FEATURES)} → {HIDDEN} → {len(TARGETS)}")
print(f"  lr={LR:.2e}  batch={BATCH}  dropout={DROPOUT}")
print(f"  HP search val MSE : {hp['val_mse']:.6f}")

# ── 1. Load & normalise ────────────────────────────────────────────────────
print("\nLoading data …")
df = pd.read_csv(DATA_FILE)
print(f"  {len(df):,} rows")

X_raw = df[FEATURES].values.astype(np.float32)
Y     = df[TARGETS].values.astype(np.float32)

mu  = X_raw.mean(axis=0)
std = X_raw.std(axis=0);  std[std == 0] = 1.0
X   = (X_raw - mu) / std

scaler = {"mean": mu.tolist(), "std": std.tolist(), "features": FEATURES}
with open(OUT_DIR / "scaler.json", "w") as f:
    json.dump(scaler, f, indent=2)
print(f"  Scaler saved → {OUT_DIR}/scaler.json")

# ── 2. Split into plain tensors (no DataLoader — avoids Python 3.8 bus error)
Xt  = torch.from_numpy(X)
Yt  = torch.from_numpy(Y)
idx = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(SEED))

n_val   = int(len(Xt) * VAL_FRAC)
n_train = len(Xt) - n_val
train_X = Xt[idx[:n_train]].to(DEVICE)
train_Y = Yt[idx[:n_train]].to(DEVICE)
val_X   = Xt[idx[n_train:]].to(DEVICE)
val_Y   = Yt[idx[n_train:]].to(DEVICE)
print(f"  Train: {n_train:,}   Val: {n_val:,}")

# ── 3. Model ───────────────────────────────────────────────────────────────
class ProbMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list, out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims   = [in_dim] + hidden + [out_dim]
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

model = ProbMLP(len(FEATURES), HIDDEN, len(TARGETS), DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_params:,}")

# ── 4. Training ────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_fn   = nn.MSELoss()

best_val   = math.inf
patience_c = 0
t0         = time.time()

print(f"\nTraining for up to {EPOCHS} epochs (early-stop patience={PATIENCE}) …")
print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>8}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm       = torch.randperm(n_train, device=DEVICE)
    train_loss = 0.0
    for start in range(0, n_train, BATCH):
        idx_b = perm[start:start + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(train_X[idx_b]), train_Y[idx_b])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(idx_b)
    train_loss /= n_train

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(val_X), val_Y).item()

    scheduler.step(val_loss)
    lr_now = optimizer.param_groups[0]["lr"]

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {lr_now:>8.2e}")

    if val_loss < best_val - 1e-7:
        best_val   = val_loss
        patience_c = 0
        torch.save(model.state_dict(), OUT_DIR / "model_best.pt")
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}  (best val MSE={best_val:.6f})")
            break

print(f"\nTraining time: {time.time()-t0:.1f}s")
print(f"Best val MSE : {best_val:.6f}")
print(f"Model saved  → {OUT_DIR}/model_best.pt")

# ── 5. CPU inference speed — batch_size=1, 1000 real data points ───────────
model.load_state_dict(torch.load(OUT_DIR / "model_best.pt", weights_only=True))
model_cpu = model.cpu()
model_cpu.eval()

bench_X = torch.from_numpy(X[:1000])

with torch.no_grad():
    for i in range(50):
        _ = model_cpu(bench_X[i].unsqueeze(0))

times = np.empty(1000)
with torch.no_grad():
    for i in range(1000):
        t1       = time.perf_counter()
        _        = model_cpu(bench_X[i].unsqueeze(0))
        times[i] = (time.perf_counter() - t1) * 1e6

print(f"\nCPU inference (batch=1, n=1000):")
print(f"  mean  : {times.mean():.1f} µs")
print(f"  median: {np.median(times):.1f} µs")
print(f"  p99   : {np.percentile(times, 99):.1f} µs")
print(f"  max   : {times.max():.1f} µs")
