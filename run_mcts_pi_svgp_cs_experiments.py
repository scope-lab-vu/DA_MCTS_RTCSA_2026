"""
Run MCTS DPW + PI-SVGP + Cornering Stiffness experiments for all track/friction
combinations.

The MCTSPiSvgpCppCsController estimates rear cornering stiffness on-the-fly
from consecutive state observations, uses it to update the PI-SVGP task latents
via rolling-window interpolation, and adapts the MCTS speed reference
accordingly.  All MCTS rollouts run in C++.

Race line: always 1.0489 (high-grip) for every experiment.
Friction:  varied via mu_override in the gym (the only thing that changes).

Usage:
    python run_mcts_pi_svgp_cs_experiments.py                  # all combos, no render
    python run_mcts_pi_svgp_cs_experiments.py --render          # with visualisation
    python run_mcts_pi_svgp_cs_experiments.py --max-steps 5000  # limit episode length
"""

import argparse
import logging
import math
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
log = logging.getLogger("mcts_pi_svgp_cs_experiments")

TRACKS = ["Sepang", "Shanghai", "YasMarina"]
# TRACKS = ["YasMarina"]


# (mu, task_index) — task_index is always 0; the CS controller adapts online.
MU_CONFIGS = [
    (1.0, 0),
    (0.9, 0),
    (0.8, 0),
    (0.7, 0),
    (0.6, 0),
    (0.5, 0),
    (0.4, 0),
    (0.3, 0),
    (0.2, 0),
    (0.1, 0),
]

BASE_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp_cs.yaml")
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


def make_map_config_all_10489_files(map_config_path: Path, track: str) -> Path:
    """Return a temp map config YAML with ALL file paths hardcoded to mu=1.0489.

    map_path, centerline_path, and wpt_path all point to the {track}_10489
    directory.  This means any friction value (including novel ones like 0.3
    that have no dedicated map directory) can be tested: the track geometry
    and raceline are always the 1.0489 variants, while the vehicle physics mu
    is set via mu_override in F1TenthAPI.

    The temp file is written into the same directory as the original map config
    so that the relative paths resolve correctly.
    """
    with open(map_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_10489/{track}_map"
    cfg["centerline_path"] = f"./{track}_10489/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_10489/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix=f"map_cs_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, task_index, seed, trial, max_steps, render) -> dict:
    """Run a single experiment. Returns summary dict."""
    mu_id = mu_to_id(mu)
    map_config_path = Path(f"maps/config_{track}_map.yaml")

    if not map_config_path.exists():
        log.warning("Map config not found: %s — skipping", map_config_path)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no map config"}

    track_dir_10489 = Path(f"maps/{track}_10489")
    if not track_dir_10489.exists():
        log.warning("Track directory not found: %s — skipping", track_dir_10489)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "status": "SKIPPED", "reason": "no track dir"}

    set_seed(seed)

    log.info("=" * 60)
    log.info(
        "EXPERIMENT: %s  mu=%.4f  task_index=%d  trial=%d  seed=%d",
        track, mu, task_index, trial, seed,
    )
    log.info("=" * 60)

    # All map/centerline/raceline files come from the 1.0489 directory.
    # mu_override in F1TenthAPI still sets vehicle physics to the desired mu.
    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, track)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, BASE_CONFIG, fixed_map_config_path, mu_override=mu)
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
        prev_laps = 0
        lap_step_start = 0
        lap_times_list: list = []

        # Access controller to log CS values periodically
        controller = api._controller

        try:
            while step < max_steps:
                t0 = time.monotonic()

                state = api.get_state()
                cmd = api.compute_action(state, waypoints)

                # MCTS dynamics are internal to C++ — no RMSE prediction available.
                episode_logger.log_step(
                    state, cmd,
                    float("nan"), float("nan"), float("nan"),
                )
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

                laps_completed = int(accumulated_progress / track_obj.total_length)

                if laps_completed > prev_laps:
                    for _ in range(laps_completed - prev_laps):
                        lap_time_s = (step - lap_step_start) * dt_s
                        lap_times_list.append(round(lap_time_s, 3))
                        lap_step_start = step
                    prev_laps = laps_completed

                if step % 100 == 0:
                    pct = (accumulated_progress / track_obj.total_length) * 100
                    cs = (
                        controller.get_cornering_stiffness()
                        if hasattr(controller, "get_cornering_stiffness")
                        else float("nan")
                    )
                    c_alpha_smooth = (
                        controller.get_smoothed_calpha()
                        if hasattr(controller, "get_smoothed_calpha")
                        else float("nan")
                    )
                    log.info(
                        "[%s mu=%.4f] step=%d  v=%.2f  progress=%.1f%%  "
                        "C_alpha=%.1f (smooth=%.1f)  dt=%.4fs",
                        track, mu, step, state.v, pct, cs, c_alpha_smooth, elapsed,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info(
                        "[%s mu=%.4f] Finished at step %d (%s)", track, mu, step, reason,
                    )
                    break

                sleep_time = dt_s - elapsed
                if sleep_time > 0 and render:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.4f] Interrupted at step %d", track, mu, step)

        laps = int(accumulated_progress / track_obj.total_length)

        # Post-loop: reconcile against gym's authoritative lap count.
        if reason == "done":
            try:
                gym_env = api._vehicle._env
                gym_laps = int(gym_env.lap_counts[gym_env.ego_idx])
            except AttributeError:
                gym_laps = laps + 1
            while len(lap_times_list) < gym_laps:
                lap_time_s = (step - lap_step_start) * dt_s
                lap_times_list.append(round(lap_time_s, 3))
                lap_step_start = step
            laps = max(laps, gym_laps)

        output_dir = str(RESULTS_DIR / f"mcts_pi_svgp_cs_{track}_{mu_id}")
        config_name = f"{BASE_CONFIG.stem}_t{trial}_s{seed}"
        extra_meta = {
            "trial": trial,
            "seed": seed,
            "lap_times_s": lap_times_list,
            "collision": reason == "collision",
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
            "task_index": task_index,
            "trial": trial,
            "seed": seed,
            "steps": step,
            "reason": reason,
            "collision": reason == "collision",
            "laps": laps,
            "progress_pct": progress_pct,
            "avg_step_ms": avg_dt * 1000,
            "lap_times": lap_times_list,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        os.unlink(fixed_map_config_path)


def main():
    parser = argparse.ArgumentParser(
        description="MCTS DPW + PI-SVGP + Cornering Stiffness batch experiments"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=10, help="Trials per (track, mu) combo")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed (trial i uses base_seed + i - 1)")
    parser.add_argument("--render", action="store_true")
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
                log.info(
                    "\n>>> Run %d/%d: %s  mu=%.4f  task=%d  trial=%d  seed=%d",
                    idx, total, track, mu, task_index, trial, seed,
                )
                result = run_experiment(
                    track, mu, task_index, seed, trial, args.max_steps, args.render,
                )
                results.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    w = 120
    print("\n" + "=" * w)
    print("MCTS + PI-SVGP + CORNERING STIFFNESS EXPERIMENT RESULTS")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'Task':>5} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'Lap times (s)'}"
    )
    print("-" * w)

    total_runs = 0
    total_collisions = 0

    for r in results:
        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<12} {r['mu']:>7.4f}  --- SKIPPED ({r.get('reason','')}) ---"
            )
            continue

        total_runs += 1
        if r["collision"]:
            total_collisions += 1

        col_str = "YES" if r["collision"] else "No"
        lap_str = "  ".join(f"{t:.1f}s" for t in r.get("lap_times", []))
        print(
            f"{r['track']:<12} {r['mu']:>7.4f} {r['task_index']:>5d} "
            f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d} {r['steps']:>7d} "
            f"{col_str:<10} {r['laps']:>5d} {r['progress_pct']:>8.1f}% "
            f"{r['reason']:<12} {r['avg_step_ms']:>8.2f}ms "
            f" [{lap_str}]"
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
