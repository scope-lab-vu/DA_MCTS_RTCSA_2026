"""
Collect per-step trajectory data from the best-tuned Stanley controller at
dt_s=0.02, in the same column schema as the existing
`MCTS_SVGP_dynaimcs_model/data/raw/traj_MPPI_*.csv` files (so downstream
preprocessing code can ingest both without changes), MINUS the columns the
MCTS controller fills but Stanley doesn't have a notion of:
    - pred_vx, pred_vy, pred_omega   (no model rollout)
    - Aa_action_list, Asv_action_list (no action sampling)
PLUS a `dt_s` column.

The dt=0.02 collection is intended for SVGP dynamics + action-prior
training that pairs with the original `da_mcts`
controller (which runs at dt_s=0.04 with svgp_base_timestep=0.02 → 2 SVGP
substeps per MCTS tick).

C_alpha, C_alpha_smooth, max_kappa, avg_kappa **are** computed here, with
the same semantics as the MCTS controller:
  - max_kappa / avg_kappa: max / mean of |κ| over a 30-step velocity-scaled
    lookahead along the centerline.
  - C_alpha: rear cornering stiffness estimated from two consecutive states
    via the lateral-force / slip-angle ratio. Clipped to [0, 104.89].
  - C_alpha_smooth: mean of the last 10 C_alpha samples.

Best Stanley params from Optuna sweep (n=1000, max_steps=6000) on
YasMarina, μ=1.0489. Trial #627 — 2 clean laps in 120 s sim, lap times
[41.58, 40.94], max_v=12.00, score=504.25:
  K_E=4.24  K_H=0.74  K_p_speed=1.79
  velocity_percentage=1.00       (full raceline speed)
  lookahead_min_m=2.34  lookahead_v_gain_s=0.08
  steering_limit=24.0

Output CSV columns (in this order):
  S_x, S_y, S_delta, S_vel, S_psi, S_omega, S_beta,
  A_a, A_sv,
  C_alpha, C_alpha_smooth,
  max_kappa, avg_kappa,
  dt_s

Usage:
  python run_exps/collect_stanley_dt0.02_data.py
  python run_exps/collect_stanley_dt0.02_data.py --n 5000 --track YasMarina
  python run_exps/collect_stanley_dt0.02_data.py --render
"""
# ── Repo-root setup ──────────────────────────────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.api import F1TenthAPI
from src.episode_logger import TRAJECTORY_COLUMNS
from src.track import Track
# Reuse the project's cornering-stiffness estimator + nominal constant so
# C_alpha matches the value the MCTS controller logs.
from src.controller.da_mcts.controller import (
    _compute_cornering_stiffness,
    _C_ALPHA_NOMINAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("collect_stanley_dt0.02")
for noisy in ("src.controller.stanley_method.controller",
              "src.controller", "src.api", "src.track",
              "src.vehicle_interface.gym_interface"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


BEST_PARAMS = {
    "K_E":                 4.24,
    "K_H":                 0.74,
    "K_p_speed":           1.79,
    "velocity_percentage": 1.00,
    "lookahead_min_m":     2.34,
    "lookahead_v_gain_s":  0.08,
    "steering_limit":      24.0,
}
DT_S = 0.02
# At dt_s=0.02 the gym integrator is stable in a single Euler step; no
# sub-stepping is needed (n_substeps = round(0.02/0.02) = 1).
PHYSICS_DT_S = 0.02

# Defaults from src/params.py (MCTS controller config), so our auxiliary
# fields match what the MCTS pipeline produces.
CURVATURE_LOOKAHEAD = 30
CALPHA_WINDOW = 10

# Drop columns the MPPI logger fills but Stanley has no semantics for.
DROP_COLS = ("pred_vx", "pred_vy", "pred_omega",
             "Aa_action_list", "Asv_action_list")
OUTPUT_COLS = [c for c in TRAJECTORY_COLUMNS if c not in DROP_COLS] + ["dt_s"]

BASE_CONFIG = Path("params/config_stanley_method.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")


def make_algo_config() -> Path:
    """Write a temp YAML config locked to the best Stanley dt=0.1 params."""
    cfg = yaml.safe_load(open(BASE_CONFIG))
    for k, v in BEST_PARAMS.items():
        cfg["controller"][k] = v
    cfg.setdefault("simulation", {})
    cfg["simulation"]["dt_s"] = DT_S
    cfg["simulation"]["physics_dt_s"] = PHYSICS_DT_S
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="stanley_collect_", dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def make_map_config(track: str, mu_id: str = "10489") -> Path:
    """Hardcode every templated path against the matching raceline directory."""
    base = Path(f"maps/config_{track}_map.yaml")
    cfg = yaml.safe_load(open(base))
    cfg["map_path"] = f"./{track}_{mu_id}/{track}_map"
    cfg["centerline_path"] = f"./{track}_{mu_id}/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_{mu_id}/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"map_collect_{track}_", dir=str(base.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def _state_finite(s) -> bool:
    return all(np.isfinite(getattr(s, k))
               for k in ("x", "y", "delta", "v", "psi", "omega", "beta"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000,
                   help="Target number of rows (one per control tick).")
    p.add_argument("--track", default="YasMarina")
    p.add_argument("--mu", type=float, default=1.0489)
    p.add_argument("--render", action="store_true",
                   help="Show the gym viewer at sim rate; the Stanley "
                        "lookahead waypoint is drawn as a yellow dot.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path. Default: "
                        "results/stanley_dt0.02_dataset/"
                        "dataset_<track>_mu<mu>.csv")
    args = p.parse_args()

    if args.out is None:
        mu_id = str(args.mu).replace(".", "")
        args.out = Path(
            f"results/stanley_dt0.02_dataset/dataset_{args.track}_mu{mu_id}.csv"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    algo_cfg = make_algo_config()
    map_cfg = make_map_config(args.track)

    log.info("=" * 70)
    log.info("Stanley dt=%.2f single-episode data collection: target=%d  out=%s",
             DT_S, args.n, args.out)
    log.info("=" * 70)
    log.info("params: %s", BEST_PARAMS)

    n_collected = 0
    early_exit_reason = None
    try:
        api = F1TenthAPI(VEHICLE_PARAMS, algo_cfg, map_cfg, mu_override=args.mu)
        waypoints = api.load_waypoints()
        if args.render:
            api.setup_rendering(waypoints)
            ctrl = api._controller   # for the lookahead highlight

        # Track for the κ lookahead.
        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        # Vehicle params for the C_alpha estimator.
        vp = api.vehicle_params
        lf, lr = float(vp.lf), float(vp.lr)
        mass, iz = float(vp.m), float(vp.I)

        # C_alpha rolling buffer (matches MCTS controller's window logic).
        calpha_buf: deque = deque(maxlen=CALPHA_WINDOW)

        # Use the project's EpisodeLogger so the schema matches the
        # MPPI traj_*.csv files. We then strip pred_*/Aa/Asv columns and
        # append dt_s before writing.
        logger = api.create_episode_logger()

        t0 = time.time()
        prev_state = None
        s = api.get_state()
        for t in range(args.n):
            tick_start = time.monotonic()

            # ── max_kappa / avg_kappa from a velocity-scaled lookahead ──
            kappas = track_obj.get_curvature_lookahead(
                position=np.array([s.x, s.y]),
                v=s.v,
                horizon=CURVATURE_LOOKAHEAD,
                dt=DT_S,
            )
            max_kappa = float(np.max(np.abs(kappas)))
            avg_kappa = float(np.mean(np.abs(kappas)))

            # ── C_alpha from finite-difference of two consecutive states ──
            if prev_state is None:
                c_alpha = _C_ALPHA_NOMINAL   # no prior tick yet → use nominal
            else:
                cs_raw = _compute_cornering_stiffness(
                    prev_state, s, DT_S, lf, lr, mass, iz,
                )
                c_alpha = float(np.clip(cs_raw, 0.0, _C_ALPHA_NOMINAL))
            calpha_buf.append(c_alpha)
            c_alpha_smooth = float(np.mean(calpha_buf))

            cmd = api.compute_action(s, waypoints)
            logger.log_step(
                s, cmd,
                c_alpha=c_alpha,
                c_alpha_smooth=c_alpha_smooth,
                max_kappa=max_kappa,
                avg_kappa=avg_kappa,
            )
            done = api.apply_action(cmd)
            sn = api.get_state()
            if not _state_finite(sn):
                early_exit_reason = "non-finite next-state (integrator divergence)"
                break
            n_collected += 1
            prev_state = s
            s = sn
            if args.render:
                if hasattr(ctrl, "get_lookahead_point"):
                    api.set_lookahead_point(ctrl.get_lookahead_point())
                api.render()
                sleep_left = DT_S - (time.monotonic() - tick_start)
                if sleep_left > 0:
                    time.sleep(sleep_left)
            if done:
                early_exit_reason = (
                    "gym signalled done"
                    + (" (collision)" if api.collision else " (lap counter / other)")
                )
                break
        wall = time.time() - t0

        # Build dataframe in the project's standard column order, drop the
        # MCTS-only columns, append dt_s.
        df = pd.DataFrame(logger._rows, columns=TRAJECTORY_COLUMNS)
        df = df.drop(columns=list(DROP_COLS))
        df["dt_s"] = DT_S
        df = df[OUTPUT_COLS]   # enforce final column order
        df.to_csv(args.out, index=False)
    finally:
        for path in (algo_cfg, map_cfg):
            try: _os.unlink(path)
            except OSError: pass

    log.info("=" * 70)
    if early_exit_reason:
        log.warning("Episode ended early after %d rows: %s",
                    n_collected, early_exit_reason)
        log.warning("If you wanted exactly %d, raise the per-episode lap "
                    "cap further in f1tenth_gym/.../f110_env.py.", args.n)
    log.info("Wrote %d rows × %d cols → %s  (%.1fs wall, sim_time = %.1fs)",
             len(df), df.shape[1], args.out, wall, n_collected * DT_S)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
