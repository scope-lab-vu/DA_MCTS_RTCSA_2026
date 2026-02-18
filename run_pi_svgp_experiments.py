"""
Run MPPI + PI-SVGP experiments for all track/friction combinations.

Usage:
    python run_pi_svgp_experiments.py                  # 9 combos, no render
    python run_pi_svgp_experiments.py --render          # with visualisation
    python run_pi_svgp_experiments.py --max-steps 5000  # limit episode length
"""

import argparse
import logging
import os
import random
import tempfile
import time
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
log = logging.getLogger("pi_svgp_experiments")

TRACKS = ["Sepang", "Shanghai", "YasMarina" ]

MU_CONFIGS = [
    # (mu, task_index)
    (0.4947, 2),
    (0.8642, 1),
    (1.0489, 0),
]

BASE_CONFIG = Path("params/config_mppi_pi_svgp.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results")


def mu_to_id(mu: float) -> str:
    return str(mu).replace(".", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config_with_task_index(base_path: Path, task_index: int) -> Path:
    """Load base YAML, override task_index, return temp file path."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"]["task_index"] = task_index
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="pi_svgp_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, task_index, seed, max_steps, render) -> dict:
    """Run a single experiment. Returns summary dict."""
    mu_id = mu_to_id(mu)
    map_config_path = Path(f"maps/config_{track}_map.yaml")

    if not map_config_path.exists():
        log.warning("Map config not found: %s — skipping", map_config_path)
        return {"track": track, "mu": mu, "status": "SKIPPED", "reason": "no map config"}

    track_dir = Path(f"maps/{track}_{mu_id}")
    if not track_dir.exists():
        log.warning("Track directory not found: %s — skipping", track_dir)
        return {"track": track, "mu": mu, "status": "SKIPPED", "reason": "no track dir"}

    set_seed(seed)

    log.info("=" * 60)
    log.info("EXPERIMENT: %s  mu=%.4f  task_index=%d  seed=%d", track, mu, task_index, seed)
    log.info("=" * 60)

    # Create config with correct task_index
    config_path = make_config_with_task_index(BASE_CONFIG, task_index)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, config_path, map_config_path, mu_override=mu)
        waypoints = api.load_waypoints()

        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        if render:
            api.setup_rendering(waypoints)

        episode_logger = api.create_episode_logger()
        dt_s = api.algo_config.simulation.dt_s

        step = 0
        reason = "max_steps"
        step_times = []
        prev_progress = None
        accumulated_progress = 0.0

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

            # Track progress
            pos = np.array([state.x, state.y])
            _, _, progress = track_obj.get_nearest_point(pos)
            if prev_progress is None:
                prev_progress = progress

            delta_p = progress - prev_progress
            if delta_p < -track_obj.total_length * 0.5:
                delta_p += track_obj.total_length
            elif delta_p > track_obj.total_length * 0.5:
                delta_p -= track_obj.total_length
            accumulated_progress += delta_p
            prev_progress = progress

            if step % 100 == 0:
                pct = (accumulated_progress / track_obj.total_length) * 100
                log.info(
                    "[%s mu=%.4f] step=%d  v=%.2f  progress=%.1f%%  dt=%.4fs",
                    track, mu, step, state.v, pct, elapsed,
                )

            if done:
                reason = "collision" if api.collision else "done"
                log.info("[%s mu=%.4f] Finished at step %d (%s)", track, mu, step, reason)
                break

            sleep_time = dt_s - elapsed
            if sleep_time > 0 and render:
                time.sleep(sleep_time)

        # Save trajectory
        output_dir = str(RESULTS_DIR / f"pi_svgp_{track}_{mu_id}")
        csv_path = api.save_episode(episode_logger, output_dir=output_dir, reason=reason)

        avg_dt = np.mean(step_times) if step_times else 0.0
        progress_pct = (accumulated_progress / track_obj.total_length) * 100
        laps = int(accumulated_progress / track_obj.total_length)

        return {
            "track": track,
            "mu": mu,
            "task_index": task_index,
            "steps": step,
            "reason": reason,
            "collision": reason == "collision",
            "laps": laps,
            "progress_pct": progress_pct,
            "avg_step_ms": avg_dt * 1000,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        os.unlink(config_path)


def main():
    parser = argparse.ArgumentParser(description="MPPI + PI-SVGP batch experiments")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    total = len(TRACKS) * len(MU_CONFIGS)
    idx = 0

    for track in TRACKS:
        for mu, task_index in MU_CONFIGS:
            idx += 1
            log.info("\n>>> Run %d/%d: %s  mu=%.4f  task=%d", idx, total, track, mu, task_index)
            result = run_experiment(track, mu, task_index, args.seed, args.max_steps, args.render)
            results.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    w = 100
    print("\n" + "=" * w)
    print("PI-SVGP EXPERIMENT RESULTS")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'Task':>5} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9}"
    )
    print("-" * w)

    total_runs = 0
    total_collisions = 0

    for r in results:
        if r["status"] == "SKIPPED":
            print(f"{r['track']:<12} {r['mu']:>7.4f}  --- SKIPPED ({r.get('reason','')}) ---")
            continue

        total_runs += 1
        if r["collision"]:
            total_collisions += 1

        col_str = "YES" if r["collision"] else "No"
        print(
            f"{r['track']:<12} {r['mu']:>7.4f} {r['task_index']:>5d} {r['steps']:>7d} "
            f"{col_str:<10} {r['laps']:>5d} {r['progress_pct']:>8.1f}% "
            f"{r['reason']:<12} {r['avg_step_ms']:>8.2f}ms"
        )

    print("=" * w)
    safe = total_runs - total_collisions
    print(
        f"\nTotal: {total_runs}  |  "
        f"Collisions: {total_collisions} ({total_collisions/max(total_runs,1)*100:.0f}%)  |  "
        f"Collision-free: {safe} ({safe/max(total_runs,1)*100:.0f}%)"
    )


if __name__ == "__main__":
    main()
