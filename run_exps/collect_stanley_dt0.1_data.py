"""
Collect (state, action, next_state) tuples from the best-tuned Stanley
controller at dt_s=0.1, for downstream model training. Single episode.

Best params from Optuna sweep (n=3000, max_steps=3000) on YasMarina,
μ=1.0489, gym lap-cap raised to 12 (= 6 laps). Trial #2415:
  K_E=0.96  K_H=0.44  K_p_speed=0.77
  velocity_percentage=0.91
  lookahead_min_m=1.13  lookahead_v_gain_s=0.33
  steering_limit=24.0
  → score 1110.81, **5 laps clean**, max_v=10.91, prog=599.9% in 3000 ticks

Output CSV columns:
  t, dt_s,
  S_x, S_y, S_delta, S_vel, S_psi, S_omega, S_beta,           # state_t (rear axle)
  A_acceleration, A_steering_vel,                              # action_t
  Sn_x, Sn_y, Sn_delta, Sn_vel, Sn_psi, Sn_omega, Sn_beta      # state_{t+1} (after dt_s = 0.1 s)

Usage:
  python run_exps/collect_stanley_dt0.1_data.py
  python run_exps/collect_stanley_dt0.1_data.py --n 5000 --track YasMarina
"""
# ── Repo-root setup ──────────────────────────────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import csv
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from src.api import F1TenthAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("collect_stanley_dt0.1")
for noisy in ("src.controller.stanley_method.controller",
              "src.controller", "src.api", "src.track",
              "src.vehicle_interface.gym_interface"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


BEST_PARAMS = {
    "K_E":                 0.96,
    "K_H":                 0.44,
    "K_p_speed":           0.77,
    "velocity_percentage": 0.91,
    "lookahead_min_m":     1.13,
    "lookahead_v_gain_s":  0.33,
    "steering_limit":      24.0,
}
DT_S = 0.10
PHYSICS_DT_S = 0.02

BASE_CONFIG = Path("params/config_stanley_method.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")

CSV_COLS = [
    "t", "dt_s",
    "S_x", "S_y", "S_delta", "S_vel", "S_psi", "S_omega", "S_beta",
    "A_acceleration", "A_steering_vel",
    "Sn_x", "Sn_y", "Sn_delta", "Sn_vel", "Sn_psi", "Sn_omega", "Sn_beta",
]


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
                   help="Target number of (s, a, s') tuples.")
    p.add_argument("--track", default="YasMarina")
    p.add_argument("--mu", type=float, default=1.0489)
    p.add_argument("--render", action="store_true",
                   help="Show the gym viewer at sim rate; the Stanley "
                        "lookahead waypoint is drawn as a yellow dot.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path. Default: "
                        "results/stanley_dt0.1_dataset/"
                        "dataset_<track>_mu<mu>.csv")
    args = p.parse_args()

    if args.out is None:
        mu_id = str(args.mu).replace(".", "")
        args.out = Path(
            f"results/stanley_dt0.1_dataset/dataset_{args.track}_mu{mu_id}.csv"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    algo_cfg = make_algo_config()
    map_cfg = make_map_config(args.track)

    log.info("=" * 70)
    log.info("Stanley dt=%.2f single-episode data collection: target=%d  out=%s",
             DT_S, args.n, args.out)
    log.info("=" * 70)
    log.info("params: %s", BEST_PARAMS)

    n_written = 0
    early_exit_reason = None
    try:
        api = F1TenthAPI(VEHICLE_PARAMS, algo_cfg, map_cfg, mu_override=args.mu)
        waypoints = api.load_waypoints()
        if args.render:
            api.setup_rendering(waypoints)
            ctrl = api._controller   # for the lookahead highlight

        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader()

            t0 = time.time()
            s = api.get_state()
            for t in range(args.n):
                tick_start = time.monotonic()
                cmd = api.compute_action(s, waypoints)
                done = api.apply_action(cmd)
                sn = api.get_state()
                if not _state_finite(sn):
                    early_exit_reason = "non-finite next-state (integrator divergence)"
                    break
                w.writerow({
                    "t": t, "dt_s": DT_S,
                    "S_x": s.x, "S_y": s.y, "S_delta": s.delta, "S_vel": s.v,
                    "S_psi": s.psi, "S_omega": s.omega, "S_beta": s.beta,
                    "A_acceleration": cmd.acceleration, "A_steering_vel": cmd.steering_vel,
                    "Sn_x": sn.x, "Sn_y": sn.y, "Sn_delta": sn.delta, "Sn_vel": sn.v,
                    "Sn_psi": sn.psi, "Sn_omega": sn.omega, "Sn_beta": sn.beta,
                })
                n_written += 1
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
    finally:
        for path in (algo_cfg, map_cfg):
            try: _os.unlink(path)
            except OSError: pass

    log.info("=" * 70)
    if early_exit_reason:
        log.warning("Episode ended early after %d tuples: %s",
                    n_written, early_exit_reason)
        log.warning("If you wanted exactly %d, increase the per-episode lap "
                    "cap in the gym (you mentioned you'd raised it).", args.n)
    log.info("Wrote %d tuples → %s  (%.1fs wall, sim_time = %.1fs)",
             n_written, args.out, wall, n_written * DT_S)
    log.info("=" * 70)


if __name__ == "__main__":
    main()