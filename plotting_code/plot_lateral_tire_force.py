"""
Total lateral tire force Fy vs vehicle body slip angle alpha.

Method (Equation 11 of arXiv:2207.07920 / _compute_cornering_stiffness):
=======================================================================
From Newton's laws for the single-track model:

    m  * vy_dot  =  Fyr + Fyf*cos(delta) - m*vx*omega       ... (lateral)
    Iz * omega_dot  =  Fyf*cos(delta)*lf  - Fyr*lr           ... (yaw)

At delta=0, omega=0 this 2x2 system (the same as in _compute_cornering_stiffness)
gives:

    Fyr = ( m*vy_dot*lf - Iz*omega_dot ) / (lf + lr)
    Fyf = ( m*vy_dot*lr + Iz*omega_dot ) / (lf + lr)

where vy_dot and omega_dot are obtained directly from vehicle_dynamics_st
(the base gym dynamics).

The x-axis is the vehicle body slip angle:

    alpha = -arctan( vy / |vx| ) = -arctan( tan(beta) ) = -beta

(same definition used in _compute_cornering_stiffness: alpha_r = atan2(lr*omega-vy, |vx|)
which reduces to -arctan(vy/|vx|) when omega=0).

Since vehicle_dynamics_st uses a *linear* tire model without hard saturation,
the resulting Fy vs alpha curve is a straight line.  The dashed horizontal lines
show the friction-circle limit  ±mu*m*g  (the maximum physically possible Fy).

Usage:
    python run_exps/plot_lateral_tire_force.py
    python run_exps/plot_lateral_tire_force.py --mu 1.0
    python run_exps/plot_lateral_tire_force.py --v 5.0
    python run_exps/plot_lateral_tire_force.py --out figs/tire.pdf
"""

# ── Repo-root setup ───────────────────────────────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "dynamic_models",
    _REPO / "f1tenth_gym/gym/f110_gym/envs/dynamic_models.py",
)
_dm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dm)
vehicle_dynamics_st = _dm.vehicle_dynamics_st

VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
MU_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
CMAP = plt.cm.viridis


def load_params():
    with open(VEHICLE_PARAMS) as f:
        return yaml.safe_load(f)["vehicle"]


def compute_forces_from_sim(beta_rad, mu, v, p):
    """
    Compute Fyf, Fyr by calling vehicle_dynamics_st and solving the 2×2
    system from Eq. 11 (same as _compute_cornering_stiffness in controller.py).

    Conditions: delta=0, omega=0, a=0  (pure slip, no steering/yaw).

    Returns (Fyr, Fyf, Fy_total, alpha_deg)
    """
    m  = p["m"]
    lf = p["lf"]
    lr = p["lr"]
    I  = p["I"]

    # State: [x, y, delta=0, v, psi=0, omega=0, beta]
    x = np.array([0.0, 0.0, 0.0, float(v), 0.0, 0.0, float(beta_rad)])
    u = np.array([0.0, 0.0])  # zero sv, zero accel

    xdot = vehicle_dynamics_st(
        x, u,
        mu, p["C_Sf"], p["C_Sr"], lf, lr, p["h"], m, I,
        p["s_min"], p["s_max"], p["sv_min"], p["sv_max"],
        p["v_switch"], p["a_max"], p["v_min"], p["v_max"],
    )

    beta_dot  = xdot[6]  # d(beta)/dt
    omega_dot = xdot[5]  # d(omega)/dt

    # vy_dot via chain rule: vy = v*sin(beta) → vy_dot = v*cos(beta)*beta_dot  (a=0)
    vy_dot = v * math.cos(beta_rad) * beta_dot

    # 2×2 linear system (delta=0, omega=0):
    #   Fyr + Fyf       = m * vy_dot           ... lateral
    #   Fyf*lf - Fyr*lr = Iz * omega_dot       ... yaw
    #
    # Solution:
    Fyr = (m * vy_dot * lf - I * omega_dot) / (lf + lr)
    Fyf = (m * vy_dot * lr + I * omega_dot) / (lf + lr)

    Fy = Fyr + Fyf   # = m * vy_dot (check)

    vx = v * math.cos(beta_rad)
    vy = v * math.sin(beta_rad)
    alpha_rad = -math.atan2(vy, abs(vx))

    return Fyr, Fyf, Fy, alpha_rad


def main():
    parser = argparse.ArgumentParser(
        description="Plot total lateral tire force vs body slip angle (simulation-based, Eq. 11)"
    )
    parser.add_argument("--mu", type=float, nargs="+", default=None)
    parser.add_argument("--v", type=float, default=10.0,
                        help="Forward speed [m/s] for the simulation sweep (default: 10)")
    parser.add_argument("--alpha-range", type=float, default=1.22,
                        help="Upper bound of slip angle sweep in radians (default: 1.22 ≈ 70°)")
    parser.add_argument("--out", type=str,
                        default="plotting_code/figs/lateral_tire_force_vs_alpha.png")
    args = parser.parse_args()

    p = load_params()
    m  = p["m"]
    g  = 9.81
    lf = p["lf"]
    lr = p["lr"]

    mu_values = sorted(args.mu) if args.mu else MU_VALUES

    # alpha = -beta, so to get positive alpha we sweep negative beta
    beta_rad   = np.linspace(0.0, -args.alpha_range, 1000)

    colors = [CMAP(i / (len(mu_values) - 1)) for i in range(len(mu_values))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for mu, color in zip(mu_values, colors):
        alpha_arr, Fy_arr = [], []
        for beta in beta_rad:
            _, _, Fy, alpha_rad = compute_forces_from_sim(beta, mu, args.v, p)
            alpha_arr.append(alpha_rad)
            Fy_arr.append(Fy)
        ax.plot(alpha_arr, Fy_arr, color=color, linewidth=1.8, label=f"μ = {mu:.4g}")

        # Friction limit (horizontal dashed line — positive side only)
        Fy_max = mu * m * g
        ax.axhline(Fy_max, color=color, linestyle=":", linewidth=0.9, alpha=0.7)

    C_Sr = p["C_Sr"]
    C_Sf = p["C_Sf"]
    alpha_sat_r = 1.0 / C_Sr
    alpha_sat_f = 1.0 / C_Sf

    # Find the practical linear region boundary using mu=max (strongest nonlinearity visible).
    # Method: compute the tangent slope at alpha~0 from the simulation, then find where
    # the simulated Fy deviates from that tangent by more than DEVIATION_THRESHOLD.
    DEVIATION_THRESHOLD = 0.05   # 5% deviation from the linear tangent

    mu_ref = max(mu_values)
    alpha_ref, Fy_ref = [], []
    for beta in beta_rad:
        _, _, Fy, alpha_r = compute_forces_from_sim(beta, mu_ref, args.v, p)
        alpha_ref.append(alpha_r)
        Fy_ref.append(Fy)
    alpha_ref = np.array(alpha_ref)
    Fy_ref    = np.array(Fy_ref)

    # Linear slope from the first few points (small alpha)
    n_fit = max(5, len(alpha_ref) // 20)
    slope = np.polyfit(alpha_ref[:n_fit], Fy_ref[:n_fit], 1)[0]
    Fy_linear = slope * alpha_ref

    # First index where relative deviation exceeds threshold (skip alpha=0)
    rel_dev = np.abs(Fy_ref - Fy_linear) / (np.abs(Fy_linear) + 1e-9)
    nonlinear_idx = np.argmax(rel_dev[1:] > DEVIATION_THRESHOLD) + 1
    alpha_nonlinear = alpha_ref[nonlinear_idx] if nonlinear_idx > 0 else alpha_ref[-1]

    # Plot the tangent line
    ax.plot(alpha_ref, Fy_linear, color="black", linestyle="--", linewidth=1.2,
            label=f"linear tangent (slope={slope:.1f} N/rad)")

    ax.axvline(alpha_nonlinear, color="tab:red", linestyle="--", linewidth=1.4,
               label=f"5% nonlinear boundary ≈ {alpha_nonlinear:.3f} rad  ({int(round(math.degrees(alpha_nonlinear)))}°)")

    # Shade the linear region
    ax.axvspan(0.0, alpha_nonlinear, alpha=0.07, color="green")

    # Reference lines for candidate alpha_threshold values
    # ax.axvline(0.7, color="tab:blue",   linestyle="-.", linewidth=1.4,
    #            label="α = 0.700 rad")
    # ax.axvline(0.8, color="tab:purple", linestyle="-.", linewidth=1.4,
    #            label="α = 0.800 rad")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlim(left=0.0)

    ax.set_xlabel("Vehicle body slip angle  α = −arctan(vy / |vx|)  [rad]", fontsize=11)
    ax.set_ylabel("Total lateral force  Fy = Fyf + Fyr  [N]", fontsize=11)
    ax.set_title(
        "F1TENTH linear tire model  —  Fy vs body slip angle  (Eq. 11 / vehicle_dynamics_st)\n"
        f"v = {args.v} m/s,  δ = 0,  ω = 0  |  C_Sf = {C_Sf},  C_Sr = {C_Sr}  [1/rad]  |  "
        "dotted lines = friction limit  μmg",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Vehicle:  m={m} kg,  lf={lf} m,  lr={lr} m,  v={args.v} m/s")
    print(f"Rear  α_sat = 1/C_Sr = {alpha_sat_r:.4f} rad  ({math.degrees(alpha_sat_r):.2f}°)")
    print(f"Front α_sat = 1/C_Sf = {alpha_sat_f:.4f} rad  ({math.degrees(alpha_sat_f):.2f}°)")
    print(f"\nSaved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
