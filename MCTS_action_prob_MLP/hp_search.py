"""
Hyperparameter search for the action-probability MLP using Optuna.
Minimises validation MSE.

Search space:
  - n_layers   : 1 – 4 hidden layers
  - hidden_dim : 32 / 64 / 128 / 256  (same width per layer)
  - lr         : log-uniform [1e-4, 1e-2]
  - batch_size : 512 / 1024 / 2048 / 4096
  - dropout    : 0.0 / 0.1 / 0.2

Usage:
  python MCTS_action_prob_MLP/hp_search.py [--trials 50] [--epochs 40]
"""

import argparse
import math
from pathlib import Path

from optuna.storages import JournalStorage, JournalFileStorage

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

# ── Config ─────────────────────────────────────────────────────────────────
DATA_FILE = Path("MCTS_action_prob_MLP/data/dataset_reduced.csv")
OUT_DIR   = Path("MCTS_action_prob_MLP")

VAL_FRAC  = 0.20
SEED      = 42
FEATURES  = ["n_iter", "t_remain_s", "v", "kappa", "max_depth"]
TARGETS   = ["p_agg", "p_slow", "p_feas"]

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Parse args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100,
                    help="Max epochs per trial (early-stop patience=5)")
args = parser.parse_args()

# ── Load & normalise once (shared across all trials) ──────────────────────
print(f"Loading data from {DATA_FILE} …")
df    = pd.read_csv(DATA_FILE)
X_raw = df[FEATURES].values.astype(np.float32)
Y     = df[TARGETS].values.astype(np.float32)

mu  = X_raw.mean(axis=0)
std = X_raw.std(axis=0);  std[std == 0] = 1.0
X   = (X_raw - mu) / std

Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)

# Split into plain tensors — no DataLoader, avoids Python 3.8 collate bus error
idx     = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(SEED))
n_val   = int(len(Xt) * VAL_FRAC)
n_train = len(Xt) - n_val
train_X = Xt[idx[:n_train]].to(DEVICE)
train_Y = Yt[idx[:n_train]].to(DEVICE)
val_X   = Xt[idx[n_train:]].to(DEVICE)
val_Y   = Yt[idx[n_train:]].to(DEVICE)
print(f"  Train: {n_train:,}   Val: {n_val:,}   Device: {DEVICE}\n")

# ── Model ──────────────────────────────────────────────────────────────────
class ProbMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout):
        super().__init__()
        dims   = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # not after last layer
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

# ── Objective ──────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    # -- sample HPs
    n_layers   = trial.suggest_int("n_layers",    1, 2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
    lr         = trial.suggest_float("lr",         1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    dropout    = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])

    hidden_dims = [hidden_dim] * n_layers
    n_params    = sum(
        (([len(FEATURES)] + hidden_dims)[i]) * d + d
        for i, d in enumerate([hidden_dim] * n_layers + [len(TARGETS)])
    )

    model = ProbMLP(len(FEATURES), hidden_dims, len(TARGETS), dropout).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val   = math.inf
    patience_c = 0
    PATIENCE   = 5

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, batch_size):
            idx_b = perm[start:start + batch_size]
            opt.zero_grad()
            loss_fn(model(train_X[idx_b]), train_Y[idx_b]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X), val_Y).item()

        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                break

    return best_val

# ── Run study ──────────────────────────────────────────────────────────────
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)
storage = JournalStorage(JournalFileStorage(str(OUT_DIR / "hp_search.log")))
study   = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
    study_name="prob_mlp_hp_search",
    storage=storage,
    load_if_exists=True,
)

print(f"Running {args.trials} trials …\n")
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

# ── Results ────────────────────────────────────────────────────────────────
best = study.best_trial
print(f"\n{'='*50}")
print(f"Best val MSE : {best.value:.6f}")
print(f"Best params  :")
for k, v in best.params.items():
    print(f"  {k:>12} = {v}")

# Save best params
import json
with open(OUT_DIR / "best_hp.json", "w") as f:
    json.dump({"val_mse": best.value, **best.params}, f, indent=2)
print(f"\nBest HPs saved → {OUT_DIR}/best_hp.json")
print(f"Full study DB  → {OUT_DIR}/hp_search.db  (open with optuna-dashboard)")
