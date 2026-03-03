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
import json
import logging
import os
import random
import signal
import subprocess
import sys
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
log = logging.getLogger("mcts_known_dyn_experiments")

# Only re-run the experiments that CRASHED (segfault) in the previous run.
# Format: (track, mu, lane_width)
RETRY_EXPERIMENTS = [
    # # Sepang crashes
    # ("Sepang", 1.0, 0.5),
    # ("Sepang", 1.0, 1.4),
    # ("Sepang", 1.0, 1.5),
    ("Sepang", 0.7, 0.7),
    # ("Sepang", 0.7, 1.5),
    # ("Sepang", 0.4, 1.0),
    # ("Sepang", 0.3, 1.2),
    # # Shanghai crashes
    # ("Shanghai", 0.9, 0.6),
    # ("Shanghai", 0.8, 0.9),
    # ("Shanghai", 0.8, 1.0),
    # ("Shanghai", 0.8, 1.5),
    # ("Shanghai", 0.6, 0.7),
    # ("Shanghai", 0.5, 1.1),
    # ("Shanghai", 0.3, 0.6),
    # ("Shanghai", 0.3, 1.0),
    # # YasMarina crashes
    # ("YasMarina", 1.0, 1.2),
    # ("YasMarina", 0.9, 0.5),
    # ("YasMarina", 0.7, 1.5),
    # ("YasMarina", 0.6, 1.4),
    # ("YasMarina", 0.5, 1.5),
    # ("YasMarina", 0.4, 0.5),
    ("YasMarina", 0.4, 1.3),
    # ("YasMarina", 0.3, 1.3),
]



BASE_CONFIG = Path("params/config_mcts_known_dyn_action_prior_cpp.yaml")
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


def make_config_with_lane_width(lane_width: float) -> Path:
    """Return a temp algo config YAML with the given real_track_width."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["real_track_width"] = lane_width
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"config_lw{lane_width}_",
        dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, task_index, seed, trial, max_steps, render, lane_width=None) -> dict:
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

    lw_str = f"  lane_width={lane_width:.1f}" if lane_width is not None else ""
    log.info("=" * 60)
    log.info(
        "EXPERIMENT: %s  mu=%.4f  task_index=%d  trial=%d  seed=%d%s",
        track, mu, task_index, trial, seed, lw_str,
    )
    log.info("=" * 60)

    # All map/centerline/raceline files come from the 1.0489 directory.
    # mu_override in F1TenthAPI still sets vehicle physics to the desired mu.
    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, track)

    # If lane_width is specified, create a temp algo config with that value.
    algo_config_path = BASE_CONFIG
    tmp_algo_config_path = None
    if lane_width is not None:
        tmp_algo_config_path = make_config_with_lane_width(lane_width)
        algo_config_path = tmp_algo_config_path

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, algo_config_path, fixed_map_config_path, mu_override=mu)
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
                cs_raw = (
                    controller.get_cornering_stiffness()
                    if hasattr(controller, "get_cornering_stiffness")
                    else float("nan")
                )
                cs_smooth = (
                    controller.get_smoothed_calpha()
                    if hasattr(controller, "get_smoothed_calpha")
                    else float("nan")
                )
                max_kappa = getattr(controller, "_max_kappa_lookahead", float("nan"))
                avg_kappa = getattr(controller, "_avg_kappa_lookahead", float("nan"))
                episode_logger.log_step(
                    state, cmd,
                    float("nan"), float("nan"), float("nan"),
                    c_alpha=cs_raw,
                    c_alpha_smooth=cs_smooth,
                    max_kappa=max_kappa,
                    avg_kappa=avg_kappa,
                )
                done = api.apply_action(cmd)

                if render:
                    # Extract MCTS planned trajectory for visualization
                    traj_steps = getattr(controller, 'get_best_trajectory', lambda: [])()
                    if traj_steps:
                        xy = np.array([[s['state'][0], s['state'][1]]
                                       for s in traj_steps if len(s['state']) >= 2])
                        if len(xy) > 0:
                            api.set_planned_trajectory(xy)
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

                if step % 1000 == 0:
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

        lw_tag = f"_lw{lane_width}" if lane_width is not None else ""
        output_dir = str(RESULTS_DIR / f"mcts_pi_svgp_cs_{track}_{mu_id}{lw_tag}")
        config_name = f"{BASE_CONFIG.stem}_t{trial}_s{seed}"
        extra_meta = {
            "trial": trial,
            "seed": seed,
            "lane_width": lane_width,
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
            "lane_width": lane_width,
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
        if tmp_algo_config_path is not None:
            try:
                os.unlink(tmp_algo_config_path)
            except OSError:
                pass


def run_experiment_in_subprocess(track, mu, task_index, seed, trial, max_steps, render, lane_width=None) -> dict:
    """Run a single experiment in a child process to isolate segfaults."""
    result_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="mcts_result_",
    )
    result_file.close()
    result_path = result_file.name

    cmd = [
        sys.executable, __file__,
        "--_subprocess",
        "--track", track,
        "--mu", str(mu),
        "--task-index", str(task_index),
        "--seed", str(seed),
        "--trial", str(trial),
        "--max-steps", str(max_steps),
        "--result-file", result_path,
    ]
    if render:
        cmd.append("--render")
    if lane_width is not None:
        cmd.extend(["--lane-width", str(lane_width)])

    try:
        proc = subprocess.run(cmd, timeout=max_steps * 2)
        retcode = proc.returncode
    except subprocess.TimeoutExpired:
        retcode = -99

    # Try to read the result JSON written by the child.
    try:
        with open(result_path) as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    # Child crashed before writing results.
    if retcode == -signal.SIGSEGV or retcode == -11:
        reason = "segfault"
    elif retcode == -signal.SIGABRT or retcode == -6:
        reason = "sigabrt"
    elif retcode == -99:
        reason = "timeout"
    else:
        reason = f"crash (exit {retcode})"

    log.error(
        "[%s mu=%.4f lw=%s trial=%d] Subprocess died: %s",
        track, mu, lane_width, trial, reason,
    )
    return {
        "track": track, "mu": mu, "task_index": task_index,
        "trial": trial, "seed": seed, "lane_width": lane_width,
        "steps": 0, "reason": reason, "collision": False,
        "laps": 0, "progress_pct": 0.0, "avg_step_ms": 0.0,
        "lap_times": [], "csv": "", "status": "CRASHED",
    }


def _subprocess_main():
    """Entry point for subprocess: run one experiment and write result to JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--_subprocess", action="store_true")
    parser.add_argument("--track", required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lane-width", type=float, default=None)
    args = parser.parse_args()

    result = run_experiment(
        args.track, args.mu, args.task_index,
        args.seed, args.trial, args.max_steps, args.render,
        lane_width=args.lane_width,
    )
    with open(args.result_file, "w") as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(
        description="MCTS DPW + PI-SVGP + Cornering Stiffness batch experiments"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per (track, mu) combo")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed (trial i uses base_seed + i - 1)")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    total = len(RETRY_EXPERIMENTS) * args.num_trials
    idx = 0
    task_index = 0  # always 0

    for track, mu, lane_width in RETRY_EXPERIMENTS:
        for trial in range(1, args.num_trials + 1):
            idx += 1
            seed = args.base_seed + trial - 1
            log.info(
                "\n>>> Run %d/%d: %s  mu=%.4f  lw=%.1f  task=%d  trial=%d  seed=%d",
                idx, total, track, mu, lane_width, task_index, trial, seed,
            )
            result = run_experiment_in_subprocess(
                track, mu, task_index, seed, trial, args.max_steps, args.render,
                lane_width=lane_width,
            )
            results.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    w = 130
    print("\n" + "=" * w)
    print("MCTS + PI-SVGP + CORNERING STIFFNESS EXPERIMENT RESULTS")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'LW':>5} {'Task':>5} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'Lap times (s)'}"
    )
    print("-" * w)

    total_runs = 0
    total_collisions = 0
    total_crashes = 0

    for r in results:
        lw = r.get("lane_width")
        lw_str = f"{lw:.1f}" if lw is not None else "  -"

        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<12} {r['mu']:>7.4f} {lw_str:>5}  "
                f"--- SKIPPED ({r.get('reason','')}) ---"
            )
            continue

        if r["status"] == "CRASHED":
            total_crashes += 1
            print(
                f"{r['track']:<12} {r['mu']:>7.4f} {lw_str:>5} {r['task_index']:>5d} "
                f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d}  "
                f"--- CRASHED ({r['reason']}) ---"
            )
            continue

        total_runs += 1
        if r["collision"]:
            total_collisions += 1

        col_str = "YES" if r["collision"] else "No"
        lap_str = "  ".join(f"{t:.1f}s" for t in r.get("lap_times", []))
        print(
            f"{r['track']:<12} {r['mu']:>7.4f} {lw_str:>5} {r['task_index']:>5d} "
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
        f"Collision-free: {safe} ({safe/max(total_runs,1)*100:.0f}%)  |  "
        f"Crashes: {total_crashes}"
    )


if __name__ == "__main__":
    # Dispatch: subprocess mode runs a single isolated experiment.
    if "--_subprocess" in sys.argv:
        _subprocess_main()
    else:
        main()
