"""
Run the stanley_method controller across multiple tracks and friction
variants. One episode per (track, mu); per-step state + action are logged
to a CSV under `results/stanley_method_<track>_<mu_id>/`. Final summary
table prints to stdout.

For each (track, μ) the script loads the raceline + map from the directory
that *matches* μ (e.g. `YasMarina_08642/` for μ=0.8642). Available racelines
are pre-generated at μ ∈ {1.0489, 0.8642, 0.4947}; passing a μ without a
matching directory will skip that combination with a warning.

This differs from the convention in run_da_mcts_*_experiments.py, which
intentionally pins the raceline to μ=1.0489 for *all* friction values to
test the MCTS planner's tolerance to friction mismatch. Stanley has no
adaptation, so we want the raceline matched to the gym's friction.

Usage:
    python run_exps/run_stanley_method_experiments.py                     # default sweep
    python run_exps/run_stanley_method_experiments.py --track YasMarina   # single track
    python run_exps/run_stanley_method_experiments.py --mu 1.0489 0.7     # subset of mu
    python run_exps/run_stanley_method_experiments.py --max-steps 3000    # longer episodes
    python run_exps/run_stanley_method_experiments.py --render            # with visualisation
"""
# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("stanley_method_experiments")

DEFAULT_TRACKS = ["YasMarina", "Sepang", "Shanghai"]
DEFAULT_MUS = [1.0489]   # extend with [1.0489, 0.8642, 0.4947] etc. for friction sweeps

BASE_CONFIG = Path("params/config_stanley_method.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results")


def mu_to_id(mu: float) -> str:
    """Match the directory naming used elsewhere in the repo: drop the
    decimal point and zero-pad. 1.0489 → '10489', 0.8642 → '08642',
    0.4947 → '04947'."""
    return str(mu).replace(".", "")


def make_fixed_map_config(map_config_path: Path, track: str, mu: float) -> Path:
    """Resolve every templated path against the raceline directory that
    matches `mu` (e.g. ./YasMarina_08642/...). Returns the temp config path."""
    mu_id = mu_to_id(mu)
    with open(map_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_{mu_id}/{track}_map"
    cfg["centerline_path"] = f"./{track}_{mu_id}/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_{mu_id}/"
        f"traj_race_cl_centerline_mintime_mue{mu}_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"map_stanley_{track}_{mu_id}_", dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def run_episode(track: str, mu: float, max_steps: int, render: bool) -> dict:
    """Run one episode; return summary dict (and write per-step CSV)."""
    map_config_path = Path(f"maps/config_{track}_map.yaml")
    if not map_config_path.exists():
        log.warning("Map config not found: %s — skipping", map_config_path)
        return {"track": track, "mu": mu, "status": "SKIPPED",
                "reason": "no map config"}

    mu_id = mu_to_id(mu)
    track_dir = Path(f"maps/{track}_{mu_id}")
    wpt_file = track_dir / f"traj_race_cl_centerline_mintime_mue{mu}_{track}_map.csv"
    if not track_dir.exists():
        log.warning("No raceline directory for μ=%.4f at %s — skipping. "
                    "Available μ values: 1.0489, 0.8642, 0.4947.",
                    mu, track_dir)
        return {"track": track, "mu": mu, "status": "SKIPPED",
                "reason": f"no raceline dir for mu={mu}"}
    if not wpt_file.exists():
        log.warning("Raceline CSV missing at %s — skipping.", wpt_file)
        return {"track": track, "mu": mu, "status": "SKIPPED",
                "reason": f"no raceline CSV for mu={mu}"}

    log.info("=" * 60)
    log.info("EXPERIMENT: %s  mu=%.4f  raceline=%s", track, mu, track_dir.name)
    log.info("=" * 60)

    fixed_map_config = make_fixed_map_config(map_config_path, track, mu)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, BASE_CONFIG, fixed_map_config, mu_override=mu)
        waypoints = api.load_waypoints()
        track_obj = Track(api.map_config.centerline_path)
        if render:
            api.setup_rendering(waypoints)

        episode_logger = api.create_episode_logger()
        dt_s = api.algo_config.simulation.dt_s

        prev_progress = None
        accumulated = 0.0
        prev_laps = 0
        lap_step_start = 0
        lap_times: list[float] = []
        max_v = 0.0
        speeds: list[float] = []
        step_times: list[float] = []
        reason = "max_steps"
        step = 0

        try:
            for step in range(max_steps):
                t0 = time.monotonic()
                state = api.get_state()
                cmd = api.compute_action(state, waypoints)
                episode_logger.log_step(state, cmd)
                done = api.apply_action(cmd)
                step_times.append(time.monotonic() - t0)

                speeds.append(state.v); max_v = max(max_v, state.v)

                if render:
                    # Forward the Stanley lookahead point so it gets drawn
                    # as a yellow highlight in the gym renderer.
                    ctrl = api._controller
                    if hasattr(ctrl, "get_lookahead_point"):
                        api.set_lookahead_point(ctrl.get_lookahead_point())
                    api.render()

                # Lap / progress tracking (matches the other run_exp scripts).
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

                if step % 500 == 0 and step > 0:
                    pct = (accumulated / track_obj.total_length) * 100
                    log.info("[%s mu=%.4f] step=%d  v=%.2f  progress=%.1f%%  laps=%d",
                             track, mu, step, state.v, pct, prev_laps)

                if done:
                    reason = "collision" if api.collision else "done"
                    break

                if render:
                    sleep_time = dt_s - step_times[-1]
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.4f] Interrupted at step %d", track, mu, step)

        progress_pct = (accumulated / track_obj.total_length) * 100
        mu_id = mu_to_id(mu)
        output_dir = str(RESULTS_DIR / f"stanley_method_{track}_{mu_id}")
        config_name = f"{BASE_CONFIG.stem}"

        extra_meta = {
            "lap_times_s": lap_times,
            "collision": (reason == "collision"),
            "max_v_mps": max_v,
            "mean_v_mps": float(np.mean(speeds)) if speeds else 0.0,
        }
        csv_path = api.save_episode(
            episode_logger,
            output_dir=output_dir,
            reason=reason,
            config_name=config_name,
            extra_metadata=extra_meta,
        )

        return {
            "track": track,
            "mu": mu,
            "steps": step + 1,
            "reason": reason,
            "collision": (reason == "collision"),
            "laps": prev_laps,
            "progress_pct": progress_pct,
            "max_v": max_v,
            "mean_v": float(np.mean(speeds)) if speeds else 0.0,
            "lap_times": lap_times,
            "avg_step_ms": float(np.mean(step_times)) * 1000 if step_times else 0.0,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        try: os.unlink(fixed_map_config)
        except OSError: pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", nargs="+", default=DEFAULT_TRACKS,
                   help="One or more track names (default: %(default)s)")
    p.add_argument("--mu", nargs="+", type=float, default=DEFAULT_MUS,
                   help="One or more friction coefficients (default: %(default)s)")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="Max simulation steps per episode (default: %(default)s)")
    p.add_argument("--render", action="store_true",
                   help="Visualise the gym environment at sim rate")
    args = p.parse_args()

    rows: list[dict] = []
    for track in args.track:
        for mu in args.mu:
            r = run_episode(track, mu, args.max_steps, args.render)
            rows.append(r)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY  (config: " + str(BASE_CONFIG) + ")")
    print("=" * 100)
    print(f"{'track':<12} {'mu':>6}  {'reason':<12} {'laps':>4}  {'prog%':>6}  "
          f"{'max_v':>5}  {'mean_v':>6}  {'steps':>5}  {'lap_times_s'}")
    print("-" * 100)
    for r in rows:
        if r.get("status") != "OK":
            print(f"{r['track']:<12} {r['mu']:>6.4f}  SKIPPED       — — — — —   "
                  f"({r.get('reason','?')})")
            continue
        print(f"{r['track']:<12} {r['mu']:>6.4f}  {r['reason']:<12} {r['laps']:>4d}  "
              f"{r['progress_pct']:>5.1f}%  {r['max_v']:>5.2f}  {r['mean_v']:>6.2f}  "
              f"{r['steps']:>5d}  {r['lap_times']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
