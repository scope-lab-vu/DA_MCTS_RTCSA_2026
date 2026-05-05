"""
Random hyperparameter search for the stanley_method controller.

Sweeps K_E, K_H, K_p_speed, velocity_lookahead_m on YasMarina + μ=1.0489
(or whatever you set). Each candidate runs a single fixed-length episode;
the candidate is scored by progress + lap-completion − collision penalty.
Results land in `results/stanley_hp_search/` as a CSV plus a top-10 print.

Usage:
    python run_exps/run_stanley_hyperparam_search.py            # default 30 trials
    python run_exps/run_stanley_hyperparam_search.py --n 60     # 60 trials
    python run_exps/run_stanley_hyperparam_search.py --max-steps 2000
    python run_exps/run_stanley_hyperparam_search.py --track Sepang
"""
# ── Repo-root setup ──────────────────────────────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("stanley_hp_search")
log.setLevel(logging.INFO)

# Silence noisy per-step logs from the controller / API.
for noisy in ("src.controller.stanley_method.controller",
              "src.controller", "src.api", "src.track",
              "src.vehicle_interface.gym_interface"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Search space ──────────────────────────────────────────────────────────
# Each candidate = independent uniform sample from each list.
SEARCH_SPACE = {
    "K_E":                 [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "K_H":                 [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
    "K_p_speed":           [0.3, 0.5, 0.7, 1.0, 1.5],
    "velocity_percentage": [1.0],  # locked at full raceline speed
}


BASE_CONFIG = Path("params/config_stanley_method.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results/stanley_hp_search")


def make_temp_config(params: dict) -> Path:
    """Write a temp config yaml that overrides the gain knobs."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for k, v in params.items():
        cfg["controller"][k] = v
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="stanley_hp_", dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_fixed_map_config(track: str, mu_id: str = "10489") -> Path:
    """Hardcode all paths in the map config for a chosen mu variant (mirrors the
    pattern in the other run_exp scripts)."""
    map_config = Path(f"maps/config_{track}_map.yaml")
    with open(map_config) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_{mu_id}/{track}_map"
    cfg["centerline_path"] = f"./{track}_{mu_id}/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_{mu_id}/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"map_stanley_hp_{track}_", dir=str(map_config.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_trial(params: dict, track: str, mu: float, max_steps: int) -> dict:
    """Run one episode and return scoring + telemetry."""
    tmp_algo = make_temp_config(params)
    tmp_map = make_fixed_map_config(track)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, tmp_algo, tmp_map, mu_override=mu)
        waypoints = api.load_waypoints()
        track_obj = Track(api.map_config.centerline_path)
        dt_s = api.algo_config.simulation.dt_s

        prev_progress = None
        accumulated = 0.0
        prev_laps = 0
        lap_step_start = 0
        lap_times: list[float] = []
        max_speed = 0.0
        speeds: list[float] = []
        reason = "max_steps"
        step = 0
        for step in range(max_steps):
            state = api.get_state()
            cmd = api.compute_action(state, waypoints)
            done = api.apply_action(cmd)
            speeds.append(state.v); max_speed = max(max_speed, state.v)

            pos = np.array([state.x, state.y])
            _, _, progress = track_obj.get_nearest_point(pos)
            if prev_progress is None:
                prev_progress = progress
            d = progress - prev_progress
            if d < -track_obj.total_length * 0.5:   d += track_obj.total_length
            elif d > track_obj.total_length * 0.5:  d -= track_obj.total_length
            accumulated += d
            prev_progress = progress

            laps = int(accumulated / track_obj.total_length)
            if laps > prev_laps:
                for _ in range(laps - prev_laps):
                    lap_times.append(round((step - lap_step_start) * dt_s, 3))
                    lap_step_start = step
                prev_laps = laps

            if done:
                reason = "collision" if api.collision else "done"
                break

        progress_pct = (accumulated / track_obj.total_length) * 100.0
        collided = (reason == "collision")
        # Score: heavy penalty for collisions; reward progress + lap completions.
        score = progress_pct + 200.0 * len(lap_times) - (500.0 if collided else 0.0)
        return {
            **params,
            "steps": step + 1,
            "reason": reason,
            "collision": collided,
            "laps": len(lap_times),
            "progress_pct": progress_pct,
            "max_v": max_speed,
            "mean_v": float(np.mean(speeds)) if speeds else 0.0,
            "lap_times_s": lap_times,
            "score": score,
        }
    finally:
        for p in (tmp_algo, tmp_map):
            try: _os.unlink(p)
            except OSError: pass


def main():
    parser = argparse.ArgumentParser(
        description="Random hyperparameter search for stanley_method.")
    parser.add_argument("--n", type=int, default=30, help="Number of random trials.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--track", default="YasMarina")
    parser.add_argument("--mu", type=float, default=1.0489)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"\n{'#':>3}  {'K_E':>5}  {'K_H':>5}  {'K_p_v':>6}  {'v_pct':>5}    "
          f"{'reason':<10}  {'laps':>4}  {'prog%':>6}  {'max_v':>6}  {'score':>7}  {'secs':>5}")
    print("-" * 110)
    for i in range(1, args.n + 1):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        t0 = time.time()
        try:
            row = run_trial(params, args.track, args.mu, args.max_steps)
        except Exception as e:
            log.warning("trial %d crashed: %s", i, e)
            row = {**params, "reason": f"err:{type(e).__name__}",
                   "laps": 0, "progress_pct": 0.0, "max_v": 0.0, "mean_v": 0.0,
                   "score": -1e9, "collision": True, "steps": 0, "lap_times_s": []}
        secs = time.time() - t0
        row["wall_secs"] = round(secs, 1)
        row["trial"] = i
        rows.append(row)
        print(f"{i:>3d}  {params['K_E']:>5.2f}  {params['K_H']:>5.2f}  "
              f"{params['K_p_speed']:>6.2f}  {params['velocity_percentage']:>5.2f}    "
              f"{row['reason']:<10}  {row['laps']:>4d}  {row['progress_pct']:>5.1f}%  "
              f"{row['max_v']:>6.2f}  {row['score']:>+7.1f}  {secs:>5.1f}s")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_csv = RESULTS_DIR / f"hp_search_{args.track}_n{args.n}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nfull results → {out_csv}")

    print("\n" + "=" * 110)
    print(f"TOP 10 BY SCORE  (track={args.track}, mu={args.mu}, max_steps={args.max_steps})")
    print("=" * 110)
    cols = ["trial", "K_E", "K_H", "K_p_speed", "velocity_percentage",
            "reason", "laps", "progress_pct", "max_v", "score"]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
