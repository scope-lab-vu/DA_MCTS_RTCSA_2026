"""
Run MCTS DPW C++ (known dynamics) at arbitrary friction values.

The raceline / map / centerline come from the nearest training anchor
{1.0489, 0.8642, 0.4947}, but the gym simulates with the requested mu.

Usage:
    python run_exp_known_dyn_friction.py --mu 0.1
    python run_exp_known_dyn_friction.py --mu 0.3 --track Sepang --render
    python run_exp_known_dyn_friction.py --mu 0.8642 --max-steps 5000 --num-trials 3
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

# ── Constants ────────────────────────────────────────────────────────────────

ANCHORS = [1.0489, 0.8642, 0.4947]

VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
DEFAULT_ALGO_CONFIG = Path("params/config_mcts_dpw_cpp_known_dyn.yaml")
RESULTS_DIR = Path("results")


def mu_to_id(mu: float) -> str:
    return str(mu).replace(".", "")


def nearest_anchor(mu: float) -> float:
    """Map arbitrary mu to the nearest training anchor."""
    return min(ANCHORS, key=lambda a: abs(a - mu))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_map_config(base_path: Path, track: str,
                     anchor_mu: float) -> Path:
    """Create a temp map config with paths resolved to the anchor's directory.

    Template variables ({track_name}, {mu_id}, {mu}) are expanded here so
    that any subsequent mu_override in the API only affects vehicle params /
    gym friction, not the file paths.
    """
    anchor_mu_id = mu_to_id(anchor_mu)

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve all path templates to the anchor values, then make absolute
    # so that load_map_config (which resolves relative to config_dir) works
    # even when the temp config lives in /tmp/.
    base_dir = base_path.parent
    path_keys = ["map_path", "wpt_path", "centerline_path"]
    for key in path_keys:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = (
                cfg[key]
                .replace("{track_name}", track)
                .replace("{mu_id}", anchor_mu_id)
                .replace("{mu}", str(anchor_mu))
            )
            cfg[key] = str((base_dir / cfg[key]).resolve())

    # Also resolve track_name / mu_id fields themselves
    cfg["track_name"] = track
    cfg["mu_id"] = anchor_mu_id

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="map_cfg_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def _make_algo_config(base_path: Path, seed: int) -> Path:
    """Create a temp algo config with the seed overridden."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["seed"] = seed
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="algo_cfg_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(
    track: str,
    mu: float,
    anchor_mu: float,
    seed: int,
    trial: int,
    max_steps: int,
    render: bool,
    algo_config: Path = DEFAULT_ALGO_CONFIG,
) -> dict:
    """Run a single trial. Returns a summary dict."""
    anchor_mu_id = mu_to_id(anchor_mu)
    actual_mu_id = mu_to_id(mu)
    map_config_base = Path(f"maps/config_{track}_map.yaml")

    if not map_config_base.exists():
        log.warning("Map config not found: %s — skipping", map_config_base)
        return {"track": track, "mu": mu, "anchor_mu": anchor_mu,
                "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no map config"}

    track_dir = Path(f"maps/{track}_{anchor_mu_id}")
    if not track_dir.exists():
        log.warning("Track directory not found: %s — skipping", track_dir)
        return {"track": track, "mu": mu, "anchor_mu": anchor_mu,
                "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no track dir"}

    set_seed(seed)

    log.info("=" * 70)
    log.info("EXPERIMENT: %s  gym_mu=%.4f  anchor_mu=%.4f  trial=%d  seed=%d",
             track, mu, anchor_mu, trial, seed)
    log.info("=" * 70)

    # Temp configs: map paths point to anchor, algo config gets seed
    map_config_path = _make_map_config(map_config_base, track, anchor_mu)
    algo_config_path = _make_algo_config(algo_config, seed)

    try:
        # mu_override sets gym friction + vehicle params;
        # map paths are already resolved to anchor so they're unaffected.
        api = F1TenthAPI(VEHICLE_PARAMS, algo_config_path, map_config_path,
                         mu_override=mu)
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
        start_progress = None
        prev_progress = 0.0
        laps_completed = 0
        prev_laps = 0
        lap_step_start = 0
        lap_times_list: list = []
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

                # Track progress along centerline
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

                if laps_completed > prev_laps:
                    for _ in range(laps_completed - prev_laps):
                        lap_time_s = (step - lap_step_start) * dt_s
                        lap_times_list.append(round(lap_time_s, 3))
                        lap_step_start = step
                    prev_laps = laps_completed

                if step % 1 == 0:
                    pct = (accumulated_progress / track_obj.total_length) * 100
                    log.info(
                        "[%s mu=%.4f anchor=%.4f t%d] step=%d  v=%.2f  "
                        "progress=%.1f%%  dt=%.4fs",
                        track, mu, anchor_mu, trial, step, state.v, pct,
                        elapsed,
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
            log.info("[%s mu=%.4f t%d] Interrupted at step %d",
                     track, mu, trial, step)

        # Reconcile lap count against gym
        if reason == "done":
            try:
                gym_env = api._vehicle._env
                gym_laps = int(gym_env.lap_counts[gym_env.ego_idx])
            except AttributeError:
                gym_laps = laps_completed + 1
            while len(lap_times_list) < gym_laps:
                lap_time_s = (step - lap_step_start) * dt_s
                lap_times_list.append(round(lap_time_s, 3))
                lap_step_start = step
            laps_completed = max(laps_completed, gym_laps)

        # Save trajectory
        output_dir = str(
            RESULTS_DIR / f"known_dyn_{track}_gym{actual_mu_id}_ref{anchor_mu_id}"
        )
        config_name = algo_config.stem
        extra_meta = {
            "gym_mu": mu,
            "anchor_mu": anchor_mu,
            "collision": reason == "collision",
            "lap_times_s": lap_times_list,
        }
        csv_path = api.save_episode(
            episode_logger,
            output_dir=output_dir,
            reason=reason,
            config_name=config_name,
            extra_metadata=extra_meta,
        )

        avg_dt = np.mean(step_times) if step_times else 0.0
        progress_pct = (accumulated_progress / track_obj.total_length) * 100

        return {
            "track": track,
            "mu": mu,
            "anchor_mu": anchor_mu,
            "trial": trial,
            "seed": seed,
            "steps": step,
            "reason": reason,
            "collision": reason == "collision",
            "laps": laps_completed,
            "progress_pct": progress_pct,
            "track_length": track_obj.total_length,
            "avg_step_ms": avg_dt * 1000,
            "lap_times": lap_times_list,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        os.unlink(map_config_path)
        os.unlink(algo_config_path)


def main():
    parser = argparse.ArgumentParser(
        description="MCTS DPW C++ (known dynamics) at arbitrary friction"
    )
    parser.add_argument("--mu", type=float, required=True,
                        help="Gym friction coefficient (e.g. 0.1, 0.5, 1.0489)")
    parser.add_argument("--track", type=str, default="YasMarina",
                        help="Track name (default: YasMarina)")
    parser.add_argument("--algo-config", type=Path,
                        default=DEFAULT_ALGO_CONFIG,
                        help="Algorithm config YAML")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of trials")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed (trial i uses base_seed + i)")
    parser.add_argument("--render", action="store_true",
                        help="Enable visualisation")
    args = parser.parse_args()

    anchor_mu = nearest_anchor(args.mu)
    log.info("Requested mu=%.4f -> nearest anchor mu=%.4f (raceline/map)",
             args.mu, anchor_mu)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for trial in range(1, args.num_trials + 1):
        seed = args.base_seed + trial - 1
        log.info("\n>>> Trial %d/%d: %s  gym_mu=%.4f  anchor=%.4f  seed=%d",
                 trial, args.num_trials, args.track, args.mu, anchor_mu, seed)
        result = run_experiment(
            args.track, args.mu, anchor_mu, seed, trial,
            args.max_steps, args.render, args.algo_config,
        )
        results.append(result)

    # ── Results table ────────────────────────────────────────────────────
    w = 120
    print("\n" + "=" * w)
    print(f"RESULTS: {args.track}  gym_mu={args.mu}  anchor_mu={anchor_mu}")
    print("=" * w)
    print(
        f"{'Trial':>6} {'Seed':>6} {'Steps':>7} {'Collision':<10} "
        f"{'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'Lap times (s)'}"
    )
    print("-" * w)

    for r in results:
        if r["status"] == "SKIPPED":
            print(f"{r['trial']:>6d} {r['seed']:>6d}  --- SKIPPED ({r['reason']}) ---")
            continue
        col_str = "YES" if r["collision"] else "No"
        lap_str = "  ".join(f"{t:.1f}s" for t in r.get("lap_times", []))
        print(
            f"{r['trial']:>6d} {r['seed']:>6d} {r['steps']:>7d} {col_str:<10} "
            f"{r['laps']:>5d} {r['progress_pct']:>8.1f}% {r['reason']:<12} "
            f"{r['avg_step_ms']:>8.2f}ms  [{lap_str}]"
        )

    # ── Aggregate ────────────────────────────────────────────────────────
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        n = len(ok_results)
        n_col = sum(1 for r in ok_results if r["collision"])
        avg_prog = np.mean([r["progress_pct"] for r in ok_results])
        avg_laps = np.mean([r["laps"] for r in ok_results])
        avg_ms = np.mean([r["avg_step_ms"] for r in ok_results])
        print(f"\n  Trials: {n}  |  Collisions: {n_col}/{n}  |  "
              f"Avg progress: {avg_prog:.1f}%  |  Avg laps: {avg_laps:.1f}  |  "
              f"Avg ms/step: {avg_ms:.2f}")


if __name__ == "__main__":
    main()
