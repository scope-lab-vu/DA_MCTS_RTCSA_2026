#!/usr/bin/env python3
"""Export the PINN model to TorchScript (.pt) for C++ LibTorch inference.

Usage:
    python scripts/export_pinn_to_torchscript.py

Produces:
    models/pinn/weights/pinn_traced.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make project root importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models.pinn.models import string_to_model

# ── Paths ─────────────────────────────────────────────────────────────────────
cfg_path = project_root / "models/pinn/deep_pacejka_f1tenth_multi_fric_best_param.yaml"
weights_path = project_root / "models/pinn/weights/epoch_388.pth"
scaler_path = project_root / "models/pinn/weights/scaler_safe.npz"
output_path = project_root / "models/pinn/weights/pinn_traced.pt"

# ── Load model ────────────────────────────────────────────────────────────────
with open(cfg_path, "r") as f:
    param_dict = yaml.safe_load(f)

model_name = param_dict["MODEL"]["NAME"]
horizon = param_dict["MODEL"]["HORIZON"]  # 10
num_features = len(param_dict["STATE"]) + len(param_dict["ACTIONS"])  # 4+2=6

print(f"Model:    {model_name}")
print(f"Horizon:  {horizon}")
print(f"Features: {num_features}")

model = string_to_model[model_name](param_dict, eval=True)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()
model.cpu()

# ── Trace ─────────────────────────────────────────────────────────────────────
batch = 1
x = torch.randn(batch, horizon, num_features)
x_norm = torch.randn(batch, horizon, num_features)
h0 = model.init_hidden(batch)

print(f"h0 shape: {h0.shape}")  # (num_layers=1, batch=1, hidden=10)

with torch.no_grad():
    # Verify forward works
    out, h_out, ff = model(x, x_norm, h0)
    print(f"Output shape: {out.shape}")  # (1, 3) = [vx, vy, yaw_rate]
    print(f"FF shape:     {ff.shape}")   # (1, 9) = Pacejka params

    traced = torch.jit.trace(model, (x, x_norm, h0))

# ── Verify traced model ──────────────────────────────────────────────────────
with torch.no_grad():
    out2, h_out2, ff2 = traced(x, x_norm, h0)
    diff = (out - out2).abs().max().item()
    print(f"Max output diff (original vs traced): {diff:.2e}")
    assert diff < 1e-5, f"Trace verification failed: diff={diff}"

# ── Save ──────────────────────────────────────────────────────────────────────
traced.save(str(output_path))
print(f"\nSaved TorchScript model to: {output_path}")

# ── Verify scaler is loadable ────────────────────────────────────────────────
scaler_data = np.load(scaler_path, allow_pickle=True)
print(f"Scaler mean:  {scaler_data['mean']}")
print(f"Scaler scale: {scaler_data['scale']}")
print("\nDone! The model is ready for C++ LibTorch inference.")
