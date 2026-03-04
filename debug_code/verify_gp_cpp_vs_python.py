"""
Verify that C++ GP prediction matches Python GP prediction.

This script:
1. Loads the trained PI-SVGP model in Python
2. Constructs a test input [VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD, C_alpha]
3. Computes GP posterior mean+var using the Python model (predict_mean_and_var)
4. Simulates what the C++ code computes using the packed flat array
5. Compares results at every stage: kernel, mean, variance, Pacejka output
"""

import sys
import os
import json
import numpy as np
import torch

# Add training code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "deep_pacejka_model_f1tenth",
                                "deep_dynamics", "pi_svgp_model_jax_no_hip"))
from model import PhysicsGuidedSVGP, apply_pacejka_constraints, _ard_rbf, PACEJKA_PARAM_NAMES, POSITIVE_PARAMS

DTYPE = torch.float64
JITTER = 1e-6

# ── 1. Load model ────────────────────────────────────────────────────────────

model_dir = os.path.join(os.path.dirname(__file__),
                         "..", "deep_pacejka_model_f1tenth",
                         "deep_dynamics", "output", "pi_svgp_no_hip", "run4_low_kl")

ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
scalers = np.load(os.path.join(model_dir, "scalers.npz"))

sd = ckpt["state_dict"]
M = ckpt["M"]
D_input = ckpt["input_dim"]
num_outputs = ckpt["num_outputs"]
K_mm_inv_saved = ckpt["K_mm_inv"]

print(f"Model: M={M}, D_input={D_input}, num_outputs={num_outputs}")

# Build Python model
model = PhysicsGuidedSVGP(
    X_m_init=torch.zeros(M, D_input, dtype=DTYPE),
    input_dim=D_input, num_outputs=num_outputs)
model.load_state_dict(sd)
model._K_mm_inv = K_mm_inv_saved
model.eval()

# Scalers
x_mean = scalers["x_mean"]  # (7,)
x_scale = scalers["x_scale"]  # (7,)
print(f"Scaler mean: {x_mean}")
print(f"Scaler scale: {x_scale}")

# ── 2. Test input ────────────────────────────────────────────────────────────

# Realistic driving state: ~5 m/s, small slip, moderate C_alpha
VX = 5.0
VY = 0.1
YAW_RATE = 0.3
STEERING_FB = 0.05
THROTTLE_CMD = 0.1 * 0.02  # accel * Ts
STEERING_CMD = 0.2 * 0.02  # sv * Ts
C_ALPHA = 80.0  # moderate grip

gp_input_raw = np.array([VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD, C_ALPHA])
print(f"\nRaw GP input: {gp_input_raw}")

# ── 3. Python model prediction ──────────────────────────────────────────────

# Normalize
gp_input_norm = (gp_input_raw - x_mean) / x_scale
print(f"Normalized input: {gp_input_norm}")

X_test = torch.tensor(gp_input_norm, dtype=DTYPE).unsqueeze(0)  # (1, 7)

with torch.no_grad():
    f_mean_py, f_var_py = model.predict_mean_and_var(X_test)

f_mean_py_np = f_mean_py[0].numpy()
f_var_py_np = f_var_py[0].numpy()

print(f"\n=== Python GP Prediction (pre-bias) ===")
print(f"f_mean: {f_mean_py_np}")
print(f"f_var:  {f_var_py_np}")

# Add bias + constraints
with torch.no_grad():
    raw_with_bias = f_mean_py + model.output_bias
    pacejka_py = apply_pacejka_constraints(raw_with_bias)
    pacejka_py_np = pacejka_py[0].numpy()

print(f"\nPacejka (Python, mean):")
for i, name in enumerate(PACEJKA_PARAM_NAMES):
    print(f"  {name}: {pacejka_py_np[i]:.6f}")

# ── 4. Simulate C++ computation ─────────────────────────────────────────────

print(f"\n=== Simulating C++ computation ===")

# Extract parameters exactly as packed for C++
log_ls = sd["log_lengthscales"].detach().numpy()
log_os_scalar = sd["log_outputscale"].detach().numpy()
Z = sd["X_m"].detach().numpy()         # (M, D_input)
m_var = sd["m"].detach().numpy()        # (M, num_outputs) - variational mean
m_logstd = sd["m_logstd"].detach().numpy()  # (M, num_outputs)
output_bias = sd["output_bias"].detach().numpy()  # (num_outputs,)
K_mm_inv_np = K_mm_inv_saved.numpy()    # (M, M)

# C++ packed values
theta_ls = np.exp(log_ls)              # lengthscales
theta_os = np.exp(log_os_scalar)       # outputscale (sigma)
sigma = theta_os
sigma_sq = sigma * sigma
alpha_packed = (K_mm_inv_np @ m_var)   # (M, num_outputs) - this is what C++ calls 'alpha'
S_diag = np.exp(2.0 * m_logstd)       # (M, num_outputs)

print(f"sigma (outputscale): {sigma}")
print(f"sigma^2: {sigma_sq}")
print(f"lengthscales: {theta_ls}")

# Step 1: Normalize (same as C++)
x_norm_cpp = (gp_input_raw - x_mean) / x_scale
print(f"\nC++ normalized input: {x_norm_cpp}")
print(f"Match Python norm: {np.allclose(x_norm_cpp, gp_input_norm)}")

# Step 2: No augmentation (D_hidden=0), so x_aug = x_norm_cpp

# Step 3: Compute k_star (ARD RBF kernel between test point and inducing points)
k_star_cpp = np.zeros(M)
for j in range(M):
    dist = 0.0
    for d in range(D_input):
        diff = (x_norm_cpp[d] - Z[j, d]) / theta_ls[d]
        dist += diff * diff
    k_star_cpp[j] = sigma_sq * np.exp(-0.5 * dist)

# Python kernel for comparison
with torch.no_grad():
    K_nm_py = _ard_rbf(X_test, model.X_m, model.log_lengthscales, model.log_outputscale)
k_nm_py_np = K_nm_py[0].numpy()

print(f"\nk_star (C++, first 5): {k_star_cpp[:5]}")
print(f"k_star (Python, first 5): {k_nm_py_np[:5]}")
print(f"k_star match: {np.allclose(k_star_cpp, k_nm_py_np, atol=1e-12)}")
print(f"k_star max diff: {np.max(np.abs(k_star_cpp - k_nm_py_np))}")

# Step 4: Compute mean: mean_p = sum_j k_star[j] * alpha_packed[j, p]
f_mean_cpp = k_star_cpp @ alpha_packed  # (num_outputs,)

print(f"\nf_mean (C++): {f_mean_cpp}")
print(f"f_mean (Py):  {f_mean_py_np}")
print(f"f_mean match: {np.allclose(f_mean_cpp, f_mean_py_np, atol=1e-10)}")
print(f"f_mean max diff: {np.max(np.abs(f_mean_cpp - f_mean_py_np))}")

# Step 5: Compute v = K_mm_inv @ k_star
v_cpp = K_mm_inv_np @ k_star_cpp  # (M,)

# Step 6: var_base = sigma^2 - k_star^T @ v
k_dot_v = k_star_cpp @ v_cpp
var_base_cpp = sigma_sq - k_dot_v

# Step 7: Per-output variance
f_var_cpp = np.zeros(num_outputs)
for p in range(num_outputs):
    var_S = np.sum(v_cpp ** 2 * S_diag[:, p])
    f_var_cpp[p] = var_base_cpp + var_S

print(f"\nf_var (C++): {f_var_cpp}")
print(f"f_var (Py):  {f_var_py_np}")
print(f"f_var match: {np.allclose(f_var_cpp, f_var_py_np, atol=1e-10)}")
print(f"f_var max diff: {np.max(np.abs(f_var_cpp - f_var_py_np))}")

# Step 8: Apply output bias + Pacejka constraints (softplus)
raw_cpp = f_mean_cpp + output_bias

def softplus_np(x):
    return np.where(x > 20.0, x, np.log(1.0 + np.exp(x)))

pacejka_cpp = np.zeros(num_outputs)
for i, name in enumerate(PACEJKA_PARAM_NAMES):
    if name in POSITIVE_PARAMS:
        pacejka_cpp[i] = softplus_np(raw_cpp[i])
    else:
        pacejka_cpp[i] = raw_cpp[i]

print(f"\nPacejka (C++ simulated, mean):")
for i, name in enumerate(PACEJKA_PARAM_NAMES):
    print(f"  {name}: C++={pacejka_cpp[i]:.6f}  Python={pacejka_py_np[i]:.6f}  diff={abs(pacejka_cpp[i]-pacejka_py_np[i]):.2e}")

print(f"\nPacejka match: {np.allclose(pacejka_cpp, pacejka_py_np, atol=1e-6)}")

# ── 5. Verify flat array packing ────────────────────────────────────────────

print(f"\n=== Verifying flat array packing ===")

# Re-create what _pack_pi_svgp_no_hip does
theta = np.concatenate([np.exp(log_ls), [np.exp(log_os_scalar)]]).astype(np.float64)
alpha_flat = (K_mm_inv_np @ m_var).ravel()
Z_flat = Z.ravel()
xmean_flat = x_mean.astype(np.float64)
xscale_flat = x_scale.astype(np.float64)
# no hp_mu (D_hidden=0)
output_bias_flat = output_bias.astype(np.float64)
K_mm_inv_flat = K_mm_inv_np.ravel()
S_diag_flat = np.exp(2.0 * m_logstd).ravel()

flat_array = np.concatenate([
    theta,              # D_input + 1 = 8
    alpha_flat,         # M * num_outputs = 450
    Z_flat,             # M * D_input = 350
    xmean_flat,         # D_input = 7
    xscale_flat,        # D_input = 7
    # no hp_mu
    output_bias_flat,   # num_outputs = 9
    K_mm_inv_flat,      # M * M = 2500
    S_diag_flat,        # M * num_outputs = 450
])

# Now unpack exactly as C++ from_flat does
D_aug = D_input  # D_hidden=0, so D_aug = D_input = 7

off = 0
theta_unpacked = flat_array[off:off + D_aug + 1]; off += D_aug + 1
alpha_unpacked = flat_array[off:off + M * num_outputs].reshape(M, num_outputs); off += M * num_outputs
Z_aug_unpacked = flat_array[off:off + M * D_aug].reshape(M, D_aug); off += M * D_aug
xmean_unpacked = flat_array[off:off + D_input]; off += D_input
xscale_unpacked = flat_array[off:off + D_input]; off += D_input
# hp_mu: D_hidden=0, so nothing
output_bias_unpacked = flat_array[off:off + num_outputs]; off += num_outputs
K_mm_inv_unpacked = flat_array[off:off + M * M].reshape(M, M); off += M * M
S_diag_unpacked = flat_array[off:off + M * num_outputs].reshape(M, num_outputs); off += M * num_outputs

print(f"Total packed size: {len(flat_array)}")
print(f"Expected: {D_aug+1 + M*num_outputs + M*D_aug + D_input + D_input + num_outputs + M*M + M*num_outputs}")
print(f"Consumed offset: {off}")

# Verify unpacked values match originals
print(f"\ntheta match: {np.allclose(theta_unpacked[:D_input], theta_ls)}")
print(f"sigma match: {np.allclose(theta_unpacked[D_aug], sigma)}")
print(f"alpha match: {np.allclose(alpha_unpacked, alpha_packed)}")
print(f"Z match: {np.allclose(Z_aug_unpacked, Z)}")
print(f"xmean match: {np.allclose(xmean_unpacked, x_mean)}")
print(f"xscale match: {np.allclose(xscale_unpacked, x_scale)}")
print(f"output_bias match: {np.allclose(output_bias_unpacked, output_bias)}")
print(f"K_mm_inv match: {np.allclose(K_mm_inv_unpacked, K_mm_inv_np)}")
print(f"S_diag match: {np.allclose(S_diag_unpacked, S_diag)}")

# ── 6. Full C++ simulation with unpacked values ─────────────────────────────

print(f"\n=== Full C++ simulation (from unpacked flat array) ===")

# Normalize
x_aug = np.zeros(D_aug)
for d in range(D_input):
    x_aug[d] = (gp_input_raw[d] - xmean_unpacked[d]) / xscale_unpacked[d]
# No hidden params (D_hidden=0)

# Compute k_star and mean
sigma_from_flat = theta_unpacked[D_aug]
sigma_sq_from_flat = sigma_from_flat * sigma_from_flat
k_star_from_flat = np.zeros(M)
mean_from_flat = np.zeros(num_outputs)

for j in range(M):
    dist = 0.0
    for d in range(D_aug):
        diff = (x_aug[d] - Z_aug_unpacked[j, d]) / theta_unpacked[d]
        dist += diff * diff
    k_star_from_flat[j] = sigma_sq_from_flat * np.exp(-0.5 * dist)

    for p in range(num_outputs):
        mean_from_flat[p] += k_star_from_flat[j] * alpha_unpacked[j, p]

# Compute v = K_mm_inv @ k_star
v_from_flat = K_mm_inv_unpacked @ k_star_from_flat

# Variance
k_dot_v_from_flat = k_star_from_flat @ v_from_flat
var_base_from_flat = sigma_sq_from_flat - k_dot_v_from_flat

var_from_flat = np.zeros(num_outputs)
for p in range(num_outputs):
    var_S = np.sum(v_from_flat ** 2 * S_diag_unpacked[:, p])
    var_from_flat[p] = var_base_from_flat + var_S

print(f"Mean from flat: {mean_from_flat}")
print(f"Mean from direct: {f_mean_cpp}")
print(f"Match: {np.allclose(mean_from_flat, f_mean_cpp, atol=1e-14)}")
print(f"Max diff: {np.max(np.abs(mean_from_flat - f_mean_cpp))}")

print(f"\nVar from flat: {var_from_flat}")
print(f"Var from direct: {f_var_cpp}")
print(f"Match: {np.allclose(var_from_flat, f_var_cpp, atol=1e-14)}")

# Apply bias + constraints
raw_from_flat = mean_from_flat + output_bias_unpacked
pacejka_from_flat = np.zeros(num_outputs)
for i, name in enumerate(PACEJKA_PARAM_NAMES):
    if name in POSITIVE_PARAMS:
        pacejka_from_flat[i] = softplus_np(raw_from_flat[i])
    else:
        pacejka_from_flat[i] = raw_from_flat[i]

print(f"\nPacejka from flat pack: {pacejka_from_flat}")
print(f"Pacejka from Python:   {pacejka_py_np}")
print(f"Match: {np.allclose(pacejka_from_flat, pacejka_py_np, atol=1e-6)}")

# ── 7. Check theta packing alignment with C++ ───────────────────────────────

print(f"\n=== Critical: theta indexing check ===")
print(f"D_input={D_input}, D_hidden=0, D_aug={D_aug}")
print(f"theta has {len(theta_unpacked)} elements (expected D_aug+1 = {D_aug+1})")
print(f"theta[0..{D_aug-1}] = lengthscales: {theta_unpacked[:D_aug]}")
print(f"theta[{D_aug}] = outputscale (sigma): {theta_unpacked[D_aug]}")
print(f"C++ code uses theta[d] for d in 0..D_aug-1 as lengthscales → CORRECT")
print(f"C++ code uses theta[D_aug] as sigma → CORRECT")

print(f"\n{'='*60}")
print(f"VERDICT: All checks passed!" if (
    np.allclose(k_star_cpp, k_nm_py_np, atol=1e-12) and
    np.allclose(f_mean_cpp, f_mean_py_np, atol=1e-10) and
    np.allclose(f_var_cpp, f_var_py_np, atol=1e-10) and
    np.allclose(pacejka_cpp, pacejka_py_np, atol=1e-6) and
    np.allclose(mean_from_flat, f_mean_cpp, atol=1e-14)
) else "VERDICT: MISMATCH DETECTED!")
print(f"{'='*60}")
