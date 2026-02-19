"""
Batch experiment runner for MCTS DPW C++ on F1TENTH.

Runs all combinations of tracks and friction coefficients, with multiple
trials per combination using different random seeds.  Saves trajectory CSVs
to results/<track>_<mu_id>/ and prints per-trial + aggregate summary tables.

Usage:
    python run_experiments.py                       # 9 combos x 10 trials
    python run_experiments.py --num-trials 5        # 9 combos x 5 trials
    python run_experiments.py --render              # with visualisation
    python run_experiments.py --max-steps 2000      # limit episode length
"""

import argparse
import logging
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("experiments")

# ── Experiment grid ──────────────────────────────────────────────────────────

TRACKS = ["Shanghai", "Sepang", ]
MU_CONFIGS = [
    # (mu, task_index)  — task_index selects the GP friction head
    (0.8642, 1),
    (1.0489, 0),
]

VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
DEFAULT_ALGO_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp_new_model.yaml")
RESULTS_DIR = Path("results")


def mu_to_id(mu: float) -> str:
    return str(mu).replace(".", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_config_with_task_index(base_path: Path, task_index: int, seed: int = 42) -> Path:
    """Load base YAML, override svgp_task_index and seed, return temp file path."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["svgp_task_index"] = task_index
    cfg["simulation"]["seed"] = seed
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="exp_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(
    track: str,
    mu: float,
    task_index: int,
    seed: int,
    trial: int,
    max_steps: int,
    render: bool,
    algo_config: Path = DEFAULT_ALGO_CONFIG,
) -> dict:
    """Run a single trial. Returns a summary dict."""
    mu_id = mu_to_id(mu)
    map_config_path = Path(f"maps/config_{track}_map.yaml")

    if not map_config_path.exists():
        log.warning("Map config not found: %s — skipping", map_config_path)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no map config"}

    track_dir = Path(f"maps/{track}_{mu_id}")
    if not track_dir.exists():
        log.warning("Track directory not found: %s — skipping", track_dir)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no track dir"}

    set_seed(seed)

    log.info("=" * 60)
    log.info("EXPERIMENT: %s  mu=%.4f  task=%d  trial=%d  seed=%d",
             track, mu, task_index, trial, seed)
    log.info("=" * 60)

    config_path = _make_config_with_task_index(algo_config, task_index, seed)
    try:
        api = F1TenthAPI(VEHICLE_PARAMS, config_path, map_config_path, mu_override=mu)
        waypoints = api.load_waypoints()

        # Load track centerline for progress computation.
        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        if render:
            api.setup_rendering(waypoints)

        episode_logger = api.create_episode_logger()
        dt_s = api.algo_config.simulation.dt_s

        step = 0
        reason = "max_steps"
        step_times = []
        start_progress = None
        prev_progress = 0.0
        laps_completed = 0
        accumulated_progress = 0.0

        try:
            while step < max_steps:
                t0 = time.monotonic()

                state = api.get_state()
                cmd = api.compute_action(state, waypoints)
                episode_logger.log_step(state, cmd)
                done = api.apply_action(cmd)

                if render:
                    api.render()

                elapsed = time.monotonic() - t0
                step_times.append(elapsed)
                step += 1

                # Track progress along centerline.
                pos = np.array([state.x, state.y])
                _, _, progress = track_obj.get_nearest_point(pos)
                if start_progress is None:
                    start_progress = progress
                    prev_progress = progress

                delta_p = progress - prev_progress
                if delta_p < -track_obj.total_length * 0.5:
                    delta_p += track_obj.total_length
                elif delta_p > track_obj.total_length * 0.5:
                    delta_p -= track_obj.total_length
                accumulated_progress += delta_p
                prev_progress = progress

                laps_completed = int(accumulated_progress / track_obj.total_length)

                if step % 100 == 0:
                    pct = (accumulated_progress / track_obj.total_length) * 100
                    log.info(
                        "[%s mu=%.4f t%d] step=%d  v=%.2f  progress=%.1f%%  dt=%.4fs",
                        track, mu, trial, step, state.v, pct, elapsed,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info("[%s mu=%.4f t%d] Finished at step %d (%s)",
                             track, mu, trial, step, reason)
                    break

                sleep_time = dt_s - elapsed
                if sleep_time > 0 and render:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.4f t%d] Interrupted at step %d", track, mu, trial, step)

        # Save trajectory.
        output_dir = str(RESULTS_DIR / f"{track}_{mu_id}")
        csv_path = api.save_episode(episode_logger, output_dir=output_dir, reason=reason)

        avg_dt = np.mean(step_times) if step_times else 0.0
        progress_pct = (accumulated_progress / track_obj.total_length) * 100
        collided = reason == "collision"

        return {
            "track": track,
            "mu": mu,
            "trial": trial,
            "seed": seed,
            "steps": step,
            "reason": reason,
            "collision": collided,
            "laps": laps_completed,
            "progress_pct": progress_pct,
            "track_length": track_obj.total_length,
            "avg_step_ms": avg_dt * 1000,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        os.unlink(config_path)


def main():
    parser = argparse.ArgumentParser(description="MCTS DPW batch experiments")
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG, help="Algorithm config YAML")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--num-trials", type=int, default=5, help="Trials per (track, mu) combo")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed (trial i uses base_seed + i)")
    parser.add_argument("--render", action="store_true", help="Enable visualisation")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    combos = len(TRACKS) * len(MU_CONFIGS)
    total = combos * args.num_trials
    idx = 0

    for track in TRACKS:
        for mu, task_index in MU_CONFIGS:
            for trial in range(1, args.num_trials + 1):
                idx += 1
                seed = args.base_seed + trial - 1
                log.info("\n>>> Run %d/%d: %s  mu=%.4f  task=%d  trial=%d  seed=%d",
                         idx, total, track, mu, task_index, trial, seed)
                result = run_experiment(track, mu, task_index, seed, trial,
                                        args.max_steps, args.render, args.algo_config)
                results.append(result)

    # ── Per-trial table ──────────────────────────────────────────────────
    w = 110
    print("\n" + "=" * w)
    print("PER-TRIAL RESULTS")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9}"
    )
    print("-" * w)

    for r in results:
        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<12} {r['mu']:>7.4f} {r['trial']:>6d} {r['seed']:>6d} "
                f"{'---':>7} {'---':<10} {'---':>5} {'---':>9} {r.get('reason',''):<12} {'---':>9}"
            )
            continue

        col_str = "YES" if r["collision"] else "No"
        print(
            f"{r['track']:<12} {r['mu']:>7.4f} {r['trial']:>6d} {r['seed']:>6d} "
            f"{r['steps']:>7d} {col_str:<10} {r['laps']:>5d} "
            f"{r['progress_pct']:>8.1f}% {r['reason']:<12} {r['avg_step_ms']:>8.2f}ms"
        )

    # ── Aggregate table ──────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in results:
        if r["status"] != "SKIPPED":
            grouped[(r["track"], r["mu"])].append(r)

    print("\n" + "=" * w)
    print("AGGREGATE SUMMARY  (per track + mu)")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'Trials':>7} {'Collisions':>11} "
        f"{'Col Rate':>9} {'Avg Laps':>9} {'Avg Prog':>9} {'Min Prog':>9} "
        f"{'Max Prog':>9} {'Avg ms':>9}"
    )
    print("-" * w)

    total_trials = 0
    total_collisions = 0

    for track in TRACKS:
        for mu, _ in MU_CONFIGS:
            key = (track, mu)
            trials = grouped.get(key, [])
            if not trials:
                print(f"{track:<12} {mu:>7.4f}  --- SKIPPED ---")
                continue

            n = len(trials)
            n_col = sum(1 for t in trials if t["collision"])
            col_rate = n_col / n * 100
            avg_laps = np.mean([t["laps"] for t in trials])
            avg_prog = np.mean([t["progress_pct"] for t in trials])
            min_prog = np.min([t["progress_pct"] for t in trials])
            max_prog = np.max([t["progress_pct"] for t in trials])
            avg_ms = np.mean([t["avg_step_ms"] for t in trials])

            total_trials += n
            total_collisions += n_col

            print(
                f"{track:<12} {mu:>7.4f} {n:>7d} {n_col:>11d} "
                f"{col_rate:>8.0f}% {avg_laps:>9.1f} {avg_prog:>8.1f}% "
                f"{min_prog:>8.1f}% {max_prog:>8.1f}% {avg_ms:>8.2f}ms"
            )

    print("=" * w)
    safe = total_trials - total_collisions
    print(
        f"\nTotal trials: {total_trials}  |  "
        f"Collisions: {total_collisions} ({total_collisions/max(total_trials,1)*100:.0f}%)  |  "
        f"Collision-free: {safe} ({safe/max(total_trials,1)*100:.0f}%)"
    )


if __name__ == "__main__":
    main()
