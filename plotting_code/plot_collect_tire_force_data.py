"""
Collect state transition data from the F1TENTH single-track dynamics.

For each target alpha in [0.1, 0.2, ..., alpha_max]:
  - Set state = [x=0, y=0, delta=0, v=V, psi=0, omega=0, beta=-alpha]
  - Action = [0, 0]  (no steering change, no acceleration)
  - Do one Euler step: s_{t+1} = s_t + dt * f(s_t, a_t)
  - Log (s_t, a_t, s_{t+1})

This matches the methodology of _compute_cornering_stiffness in the controller.

Usage:
    python plotting_code/collect_tire_force_data.py
    python plotting_code/collect_tire_force_data.py --v 10.0 --alpha-max 3.0
"""

import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702

import argparse
import math
import numpy as np
import yaml
import csv
from pathlib import Path
import matplotlib.pyplot as plt

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "dynamic_models",
    _REPO / "f1tenth_gym/gym/f110_gym/envs/dynamic_models.py",
)
_dm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dm)
vehicle_dynamics_st = _dm.vehicle_dynamics_st


def load_params():
    with open("params/vehicle_params.yaml") as f:
        return yaml.safe_load(f)["vehicle"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=float, default=5.0, help="Forward speed [m/s]")
    parser.add_argument("--dt", type=float, default=0.02, help="Euler step size [s]")
    parser.add_argument("--alpha-step", type=float, default=0.01, help="Alpha increment [rad]")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Max alpha [rad]")
    parser.add_argument("--mu", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0489],
                        help="Friction values to sweep (default: 0.3 0.5 0.7 1.0489)")
    parser.add_argument("--out", type=str, default="plotting_code/data/tire_force_transitions.csv")
    args = parser.parse_args()

    p = load_params()
    dt = args.dt
    v_fixed = args.v
    mu_values = sorted(args.mu)

    alpha_values = np.arange(0.0, args.alpha_max + 1e-9, args.alpha_step)

    lf = p["lf"]
    lr = p["lr"]
    mass = p["m"]
    iz = p["I"]
    action = np.array([p["sv_max"], p["a_max"]])

    # Store results per mu: {mu: (alphas, F_y_arr)}
    results = {}

    for mu in mu_values:
        F_y_list = []

        print(f"\n── μ = {mu} ──")
        print(f"{'alpha':>8s}  {'F_yr':>12s}  {'F_yf':>12s}  {'F_y':>12s}")
        print("-" * 50)

        for alpha in alpha_values:
            s_t = np.array([0.0, 0.0, 0.0, v_fixed, 0.0, 0.0, -alpha])

            xdot = vehicle_dynamics_st(
                s_t, action,
                mu, p["C_Sf"], p["C_Sr"], p["lf"], p["lr"], p["h"],
                p["m"], p["I"],
                p["s_min"], p["s_max"], p["sv_min"], p["sv_max"],
                p["v_switch"], p["a_max"], p["v_min"], p["v_max"],
            )

            s_tp1 = s_t + dt * xdot

            v_x_prev = s_t[3] * math.cos(s_t[6])
            v_y_prev = s_t[3] * math.sin(s_t[6])
            v_y_curr = s_tp1[3] * math.sin(s_tp1[6])

            v_y_dot = (v_y_curr - v_y_prev) / dt
            omega_dot = (s_tp1[5] - s_t[5]) / dt

            F_yr = (
                lf * mass * (v_y_dot + v_x_prev * s_t[5]) - iz * omega_dot
            ) / (lf + lr)

            delta = s_t[2]
            cos_delta = math.cos(delta)
            F_yf = (v_y_dot * mass - F_yr + mass * v_x_prev * s_t[5]) / cos_delta
            F_y = F_yr + F_yf

            F_y_list.append(F_y)
            print(f"{alpha:8.2f}  {F_yr:12.4f}  {F_yf:12.4f}  {F_y:12.4f}")

        results[mu] = np.array(F_y_list)

    # ── Plot Fy vs alpha for all mu values ────────────────────────────────
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(mu_values) - 1, 1)) for i in range(len(mu_values))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (mu, F_y_arr), color in zip(results.items(), colors):
        ax.plot(alpha_values, F_y_arr, "o-", color=color, linewidth=1.8,
                markersize=4, label=f"μ = {mu:.4g}")

    # Linear tangent from the mu_ref curve (slope from first nonzero point)
    mu_ref = max(mu_values)
    F_y_ref = results[mu_ref]
    nonzero = alpha_values > 0
    slope_ref = F_y_ref[nonzero][0] / alpha_values[nonzero][0]
    F_y_lin_ref = slope_ref * alpha_values

    ax.plot(alpha_values, F_y_lin_ref, "--", color="gray", linewidth=1.2,
            label=f"Linear (slope from μ={mu_ref:.4g})")

    # Find alpha_th for mu_ref: where F_y drops to 90% of its own linear tangent
    alpha_th = None
    for i in range(len(alpha_values)):
        if alpha_values[i] > 0 and F_y_lin_ref[i] > 0:
            ratio = F_y_ref[i] / F_y_lin_ref[i]
            if ratio < 0.9:
                if i > 0:
                    r_prev = F_y_ref[i-1] / F_y_lin_ref[i-1] if F_y_lin_ref[i-1] > 0 else 1.0
                    frac = (0.9 - r_prev) / (ratio - r_prev) if ratio != r_prev else 0.0
                    alpha_th = alpha_values[i-1] + frac * (alpha_values[i] - alpha_values[i-1])
                else:
                    alpha_th = alpha_values[i]
                break

    if alpha_th is not None:
        print(f"\n── 10% deviation threshold ──")
        print(f"  μ = {mu_ref:.4g}:  α_th = {alpha_th:.3f} rad ({math.degrees(alpha_th):.1f}°)")
        ax.axvline(alpha_th, color="tab:red", linestyle="-.", linewidth=1.4,
                   label=f"α_th = {alpha_th:.3f} rad (10% dev, μ={mu_ref:.2g})")

    ax.axhline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("Body slip angle  α  [rad]", fontsize=11)
    ax.set_ylabel("Lateral force  $F_y$  [N]", fontsize=11)
    ax.set_title(
        f"F1TENTH ST dynamics — Lateral tire force vs α\n"
        f"v = {v_fixed} m/s,  dt = {dt} s  |  vertical lines = 10% deviation α_th",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = Path("plotting_code/figs/Fy_vs_alpha.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
