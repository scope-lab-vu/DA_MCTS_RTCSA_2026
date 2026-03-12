"""
Train a small MLP to predict (p_agg, p_slow, p_feas) from
(n_iter, t_remain_s, v, kappa, max_depth).

Outputs sum to 1  →  softmax final layer.
Exported as TorchScript (model_scripted.pt) for fast CPU inference with no
Python overhead, plus scaler.json for input normalisation.

Architecture: 5 → 64 → 64 → 3   (~8 k parameters)
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Config ─────────────────────────────────────────────────────────────────
DATA_FILE  = Path("MCTS_action_prob_MLP/data/dataset_reduced.csv")
OUT_DIR    = Path("MCTS_action_prob_MLP")

HIDDEN     = [64, 64]        # hidden layer widths — keep small for CPU speed
LR         = 1e-3
BATCH      = 2048
EPOCHS     = 1
VAL_FRAC   = 0.20
PATIENCE   = 10              # early-stopping patience (epochs)
SEED       = 42

FEATURES   = ["n_iter", "t_remain_s", "v", "kappa", "max_depth"]
TARGETS    = ["p_agg", "p_slow", "p_feas"]

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 1. Load & normalise ────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_FILE)
print(f"  {len(df):,} rows")

X_raw = df[FEATURES].values.astype(np.float32)
Y     = df[TARGETS].values.astype(np.float32)

# StandardScaler (manual, so we can serialise to JSON)
mu  = X_raw.mean(axis=0)
std = X_raw.std(axis=0)
std[std == 0] = 1.0          # avoid /0 for constant columns
X   = (X_raw - mu) / std

# Save scaler
scaler = {"mean": mu.tolist(), "std": std.tolist(), "features": FEATURES}
with open(OUT_DIR / "scaler.json", "w") as f:
    json.dump(scaler, f, indent=2)
print(f"  Scaler saved → {OUT_DIR}/scaler.json")

# ── 2. Dataset / split ─────────────────────────────────────────────────────
Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)
ds = TensorDataset(Xt, Yt)

n_val   = int(len(ds) * VAL_FRAC)
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
print(f"  Train: {n_train:,}   Val: {n_val:,}")

# ── 3. Model ───────────────────────────────────────────────────────────────
class ProbMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list, out_dim: int):
        super().__init__()
        dims   = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

model = ProbMLP(len(FEATURES), HIDDEN, len(TARGETS)).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {FEATURES} → {HIDDEN} → {TARGETS}")
print(f"  Parameters: {n_params:,}")

# ── 4. Training ────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=False)
loss_fn = nn.MSELoss()

best_val   = math.inf
patience_c = 0
t0         = time.time()

print(f"\nTraining for up to {EPOCHS} epochs (early-stop patience={PATIENCE}) …")
print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>8}")

for epoch in range(1, EPOCHS + 1):
    # -- train
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= n_train

    # -- validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(model(xb), yb).item() * len(xb)
    val_loss /= n_val

    scheduler.step(val_loss)
    lr_now = optimizer.param_groups[0]["lr"]

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {lr_now:>8.2e}")

    # -- early stopping
    if val_loss < best_val - 1e-7:
        best_val = val_loss
        patience_c = 0
        torch.save(model.state_dict(), OUT_DIR / "model_best.pt")
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}  (best val MSE={best_val:.6f})")
            break

print(f"\nTraining time: {time.time()-t0:.1f}s")

# ── 5. Reload best weights & export TorchScript ───────────────────────────
model.load_state_dict(torch.load(OUT_DIR / "model_best.pt", weights_only=True))
model.eval()

scripted = torch.jit.script(model)
scripted.save(str(OUT_DIR / "model_scripted.pt"))
print(f"TorchScript model saved → {OUT_DIR}/model_scripted.pt")

# ── 6. CPU inference speed — batch_size=1, 1000 real data points ──────────
cpu_model = scripted.cpu()
cpu_model.eval()

# Take 1000 samples from the dataset (already normalised)
bench_X = torch.from_numpy(X[:1000])   # shape (1000, 5), on CPU

# Warm-up
with torch.no_grad():
    for i in range(50):
        _ = cpu_model(bench_X[i].unsqueeze(0))

# Timed run — one sample at a time
times = []
with torch.no_grad():
    for i in range(1000):
        t0 = time.perf_counter()
        _ = cpu_model(bench_X[i].unsqueeze(0))
        times.append((time.perf_counter() - t0) * 1e6)

times = np.array(times)
print(f"\nCPU inference (batch=1, n=1000 real samples):")
print(f"  mean : {times.mean():.1f} µs")
print(f"  median: {np.median(times):.1f} µs")
print(f"  p99  : {np.percentile(times, 99):.1f} µs")
print(f"  max  : {times.max():.1f} µs")

# ── 7. Final metrics on ALL raw data (chunked to avoid OOM) ───────────────
print("\nEvaluating on all raw data …")
all_files = sorted(Path("MCTS_action_prob_MLP/data").glob("mcts_weight_log_*.csv"))
model.to(DEVICE)
model.eval()

abs_errs   = {t: 0.0 for t in TARGETS}
total_rows = 0
CHUNK      = 50_000

with torch.no_grad():
    for fpath in all_files:
        raw = pd.read_csv(fpath)
        Xr  = ((raw[FEATURES].values.astype(np.float32) - mu) / std)
        Yr  = raw[TARGETS].values.astype(np.float32)
        for start in range(0, len(Xr), CHUNK):
            xb  = torch.from_numpy(Xr[start:start+CHUNK]).to(DEVICE)
            yb  = Yr[start:start+CHUNK]
            yhat = model(xb).cpu().numpy()
            for i, t in enumerate(TARGETS):
                abs_errs[t] += np.abs(yhat[:, i] - yb[:, i]).sum()
        total_rows += len(raw)

print(f"  Total rows evaluated: {total_rows:,}")
print("\nMAE per output (all raw data):")
for t in TARGETS:
    print(f"  {t}: {abs_errs[t]/total_rows:.4f}")
