"""
Benchmark: Linear interpolation vs Python MLP loop vs C++ MLP (inline in learn_timed).

Single timestep comparison — CURVE scenario (kappa > kappa_threshold).

Methods compared
----------------
Old (linear tau) : set_action_weights from linear formula once + learn(N_TOTAL)
Python MLP loop  : Python budget loop — learn(CHUNK) in chunks + MLP predict every chunk
C++ MLP          : load_mlp_weights() once + learn_timed(BUDGET_S) — MLP runs inline in C++

All methods use the same fake SVGP params / occ-grid / state (no real model files needed).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
MLP_DIR = ROOT / "MCTS_action_prob_MLP"

sys.path.insert(0, str(ROOT))

# ── import C++ extension ────────────────────────────────────────────────────
try:
    from mcts_svgp_cpp_budget_aware_chance_prob_pred import F1TenthMCTSSvgpBudgetProbPred
except ImportError:
    sys.exit("Build the C++ extension first:\n"
             "  venv/bin/pip install -e src/controller/mcts_svgp_cpp_budget_aware_chance_prob_pred/")

# ── benchmark settings ──────────────────────────────────────────────────────
N_TOTAL      = 3000   # MCTS iterations for Old and Python-MLP methods
N_REPEATS    = 5      # repeats to average
CHUNK        = 10     # learn(CHUNK) per Python loop step
N_RECALIB    = 10     # MLP called every chunk — matches C++ N_r_
N_WARMUP_IT  = 300    # warmup iterations

# Curve scenario: kappa > kappa_threshold, longer rollout horizon
KAPPA        = 0.30
ROLLOUT_CURVE = 15

# "Old" linear params
W_LOW, W_HIGH = 15.0, 50.0

# ── MLP (Python, for Old-baseline and Python-loop methods) ──────────────────
class _ProbMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def load_mlp():
    sd = torch.load(MLP_DIR / "model_best.pt", weights_only=True, map_location="cpu")
    lw = [v for k, v in sd.items() if k.endswith(".weight") and "net." in k]
    hidden_dims = [w.shape[0] for w in lw[:-1]]
    model = _ProbMLP(5, hidden_dims, 3)
    model.load_state_dict(sd)
    model.eval()
    with open(MLP_DIR / "scaler.json") as f:
        sc = json.load(f)
    mu  = np.array(sc["mean"], dtype=np.float32)
    std = np.array(sc["std"],  dtype=np.float32)
    return model, mu, std, sd

@torch.no_grad()
def mlp_predict(model, mu, std, n_iter, t_remain, v, kappa, max_depth):
    x = np.array([[n_iter, t_remain, v, kappa, max_depth]], dtype=np.float32)
    x = (x - mu) / std
    p = model(torch.from_numpy(x))[0].numpy()
    return float(p[0]), float(p[1]), float(p[2])

# ── linear-tau weight helper ─────────────────────────────────────────────────
def linear_weights(n_total):
    alpha = 0.5
    W_hat = n_total ** alpha
    tau   = max(0.0, min(1.0, (W_hat - W_LOW) / (W_HIGH - W_LOW)))
    p_slow = 0.90 - 0.85 * tau
    p_feas = 0.05 + 0.55 * tau
    p_agg  = 1.0 - p_slow - p_feas
    return p_agg, p_slow, p_feas

# ── extract weight arrays for C++ ────────────────────────────────────────────
def extract_cpp_weights(sd, mu, std):
    W0 = sd["net.0.weight"].detach().numpy().astype(np.float32)
    b0 = sd["net.0.bias"].detach().numpy().astype(np.float32)
    W1 = sd["net.2.weight"].detach().numpy().astype(np.float32)
    b1 = sd["net.2.bias"].detach().numpy().astype(np.float32)
    W2 = sd["net.4.weight"].detach().numpy().astype(np.float32)
    b2 = sd["net.4.bias"].detach().numpy().astype(np.float32)
    return W0, b0, W1, b1, W2, b2, mu.copy(), std.copy()

# ── build a minimal fake tree ────────────────────────────────────────────────
def make_fake_tree():
    M, D = 2, 7
    flat_size  = D + 1 + M + M*D + D + D + 2 + M*M
    xscale_off = D + 1 + M + M*D + D
    yscale_off = xscale_off + D + 1

    def _fake_svgp():
        arr = np.zeros(flat_size)
        arr[:D]                        = 1.0  # lengthscales = 1
        arr[D]                         = 1.0  # sigma_f = 1
        arr[xscale_off:xscale_off + D] = 1.0  # xscale = 1
        arr[yscale_off]                = 1.0  # yscale = 1
        return arr

    occ   = np.zeros(20 * 20, dtype=np.float32)
    state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])

    tree = F1TenthMCTSSvgpBudgetProbPred(
        initial_obs=state,
        alpha=0.5, beta=0.5, K=1.0,
        mu=0.7, C_Sf=4.718, C_Sr=5.4562,
        lf=0.15875, lr=0.17145, h=0.074,
        m=3.47, I=0.04712,
        s_min=-0.4189, s_max=0.4189,
        sv_min=-3.2, sv_max=3.2,
        v_switch=7.319, a_max=9.51,
        v_min=0.5, v_max=20.0,
        width=0.31, length=0.58,
        occ_grid=occ,
        occ_x_min=-5.0, occ_y_min=-5.0,
        occ_cell_size=0.5, occ_nx=20, occ_ny=20,
        occ_grid_real=occ,
        speed_weight=1.0, alpha_threshold=0.2,
        v_ref=5.0, rollout_horizon=ROLLOUT_CURVE,
        discount=0.99, dt=0.04,
        normalize_reward=False,
        omega_data=_fake_svgp(), beta_data=_fake_svgp(),
        svgp_M_omega=M, svgp_M_beta=M, svgp_D_input=D,
        base_timestep=0.02,
        seed=42,
    )
    cx      = np.linspace(0, 100, 106)
    cy      = np.zeros(106)
    heading = np.zeros(106)
    tree.set_centerline(cx, cy, heading)
    tree.set_v_ref_window(np.full(ROLLOUT_CURVE, 5.0))
    tree.set_action_prior(np.array([1.0, 0.0]), np.array([1.0, 0.3]))
    tree.set_is_curve(True)
    return tree

# ── benchmark functions ───────────────────────────────────────────────────────

def bench_old_linear(tree):
    """Old: equal-weight warmup + one linear weight estimate + learn(N_TOTAL)."""
    tree.set_action_weights(1/3, 1/3, 1/3)
    tree.learn(N_WARMUP_IT)
    p_agg, p_slow, p_feas = linear_weights(N_TOTAL)
    tree.set_action_weights(p_agg, p_slow, p_feas)
    tree.learn(N_TOTAL - N_WARMUP_IT)


def bench_python_mlp(tree, model, mu, std):
    """Python MLP loop: learn(CHUNK) + MLP predict every chunk."""
    tree.set_action_weights(1/3, 1/3, 1/3)
    n = 0
    while n < N_WARMUP_IT:
        tree.learn(CHUNK)
        n += CHUNK

    p_agg, p_slow, p_feas = mlp_predict(
        model, mu, std, N_TOTAL, 0.05, 5.0, KAPPA, ROLLOUT_CURVE)
    tree.set_action_weights(p_agg, p_slow, p_feas)

    since_recalib = 0
    while n < N_TOTAL:
        tree.learn(CHUNK)
        n += CHUNK
        since_recalib += CHUNK
        if since_recalib >= N_RECALIB:
            since_recalib = 0
            p_agg, p_slow, p_feas = mlp_predict(
                model, mu, std, N_TOTAL, 0.05, 5.0, KAPPA, ROLLOUT_CURVE)
            tree.set_action_weights(p_agg, p_slow, p_feas)


def bench_cpp_mlp(tree, W0, b0, W1, b1, W2, b2, mu, sigma):
    """C++ MLP: load weights once, learn_n_mlp(N_TOTAL) — exactly N_TOTAL iters, MLP inline."""
    tree.set_budget_params(W_LOW, W_HIGH, N_WARMUP_IT, N_RECALIB)
    tree.load_mlp_weights(W0, b0, W1, b1, W2, b2, mu, sigma)
    tree.set_context(5.0, KAPPA)
    tree.learn_n_mlp(N_TOTAL)


# ── main ─────────────────────────────────────────────────────────────────────

print("Loading MLP …")
mlp_model, mlp_mu, mlp_std, mlp_sd = load_mlp()
W0, b0, W1, b1, W2, b2, cpp_mu, cpp_sigma = extract_cpp_weights(mlp_sd, mlp_mu, mlp_std)

# Python MLP single inference latency
mlp_times = []
for _ in range(500):
    t0 = time.perf_counter()
    mlp_predict(mlp_model, mlp_mu, mlp_std, 3000, 0.05, 5.0, 0.05, 10)
    mlp_times.append((time.perf_counter() - t0) * 1e6)
mlp_times = np.array(mlp_times[50:])
print(f"\nPython MLP single-call latency (CPU, batch=1):")
print(f"  mean={mlp_times.mean():.1f} µs   median={np.median(mlp_times):.1f} µs   "
      f"p99={np.percentile(mlp_times, 99):.1f} µs")

n_recalib_calls = (N_TOTAL - N_WARMUP_IT) // N_RECALIB + 1
print(f"\nPython MLP calls per {N_TOTAL}-iter step: {n_recalib_calls}  "
      f"(every {N_RECALIB} iters)")
print(f"Estimated Python MLP overhead: {n_recalib_calls * mlp_times.mean():.1f} µs "
      f"= {n_recalib_calls * mlp_times.mean() / 1000:.3f} ms")

print(f"\nScenario: CURVE  kappa={KAPPA}  rollout_horizon={ROLLOUT_CURVE}")
print(f"Building fake trees …")

# ── Benchmark old ──────────────────────────────────────────────────────────
print(f"\nBenchmarking OLD (linear tau)   — {N_TOTAL} iters × {N_REPEATS} repeats …")
old_times = []
for r in range(N_REPEATS):
    t0 = time.perf_counter()
    bench_old_linear(make_fake_tree())
    old_times.append((time.perf_counter() - t0) * 1e3)

# ── Benchmark Python MLP ───────────────────────────────────────────────────
print(f"Benchmarking PYTHON MLP loop    — {N_TOTAL} iters × {N_REPEATS} repeats …")
py_times = []
for r in range(N_REPEATS):
    t0 = time.perf_counter()
    bench_python_mlp(make_fake_tree(), mlp_model, mlp_mu, mlp_std)
    py_times.append((time.perf_counter() - t0) * 1e3)

# ── Benchmark C++ MLP ─────────────────────────────────────────────────────
print(f"Benchmarking C++ MLP            — {N_TOTAL} iters × {N_REPEATS} repeats …")
cpp_times = []
for r in range(N_REPEATS):
    t0 = time.perf_counter()
    bench_cpp_mlp(make_fake_tree(), W0, b0, W1, b1, W2, b2, cpp_mu, cpp_sigma)
    cpp_times.append((time.perf_counter() - t0) * 1e3)

old_times = np.array(old_times)
py_times  = np.array(py_times)
cpp_times = np.array(cpp_times)

print(f"\nAll methods: {N_TOTAL} iterations")
print(f"\n{'':=<62}")
print(f"{'Method':<28}  {'mean ms':>8}  {'min ms':>7}  {'vs baseline':>12}")
print(f"{'-'*62}")
print(f"{'Old (linear tau)':<28}  {old_times.mean():>8.1f}  "
      f"{old_times.min():>7.1f}  {'baseline':>12}")
py_oh  = 100.0 * (py_times.mean() - old_times.mean()) / old_times.mean()
print(f"{'Python MLP loop':<28}  {py_times.mean():>8.1f}  "
      f"{py_times.min():>7.1f}  {py_oh:>+11.1f}%")
cpp_oh = 100.0 * (cpp_times.mean() - old_times.mean()) / old_times.mean()
print(f"{'C++ MLP (learn_n_mlp)':<28}  {cpp_times.mean():>8.1f}  "
      f"{cpp_times.min():>7.1f}  {cpp_oh:>+11.1f}%")
print(f"{'':=<62}")
print(f"\nPython→C++ MLP speedup: {py_times.mean() / cpp_times.mean():.2f}×")
