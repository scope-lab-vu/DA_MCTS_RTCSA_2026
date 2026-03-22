"""
Run Localization-Budget-Aware MCTS experiments.

Runs the particle filter localization in-loop, then gives MCTS the remaining
time budget (minus overhead).  Sweeps over tracks, friction coefficients,
and number of trials — recording both localization error and driving performance.

Usage:
    python run_localize_budget_experiments.py                         # all combos
    python run_localize_budget_experiments.py --render                # with visualisation
    python run_localize_budget_experiments.py --track YasMarina       # single track
    python run_localize_budget_experiments.py --mu 1.0 0.5            # subset of mu values
    python run_localize_budget_experiments.py --max-steps 5000        # limit episode length
"""

# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

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
log = logging.getLogger("localize_budget_experiments")

TRACKS = ["Sepang", "YasMarina"]

MU_CONFIGS = [
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
]

BASE_CONFIG = Path("params/config_mcts_svgp_budget_prob_pred.yaml")
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
    """Return a temp map config YAML with ALL file paths hardcoded to mu=1.0489."""
    with open(map_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_10489/{track}_map"
    cfg["centerline_path"] = f"./{track}_10489/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_10489/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"map_localize_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_config_localize_budget(overhead_s: float = 0.010) -> Path:
    """Return a temp algo config YAML with localize_budget enabled."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["localize_budget"] = True
    cfg["controller"]["localize_budget_overhead_s"] = overhead_s
    # Disable dynamic budget since localize_budget takes priority
    cfg["controller"]["dynamic_budget"] = False
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="config_localize_budget_",
        dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, seed, trial, max_steps, render, overhead_s) -> dict:
    """Run a single experiment with PF localization + MCTS. Returns summary dict."""
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
        "EXPERIMENT: %s  mu=%.4f  localize_budget  trial=%d  seed=%d",
        track, mu, trial, seed,
    )
    log.info("=" * 60)

    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, track)
    tmp_algo_config_path = make_config_localize_budget(overhead_s)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, tmp_algo_config_path, fixed_map_config_path, mu_override=mu)
        waypoints = api.load_waypoints()

        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        if render:
            api.setup_rendering(waypoints)

        episode_logger = api.create_episode_logger()
        dt_s = api.algo_config.simulation.dt_s
        dt_ms = dt_s * 1000.0

        # --- Initialize particle filter ---
        from src.localize.particle_filter import ParticleFilter, PFConfig
        from src.localize.particle_filter.pf_logger import PFLogger

        map_yaml = str(
            Path(api.map_config.map_path).parent
            / (Path(api.map_config.map_path).name + ".yaml")
        )
        pf_config = PFConfig(max_particles=1500, range_method='rm')
        pf = ParticleFilter(pf_config, map_yaml)
        pf.set_pose(api.map_config.sx, api.map_config.sy, api.map_config.stheta)
        scan_angles = api.get_scan_angles()

        pf_log_path = str(
            RESULTS_DIR / f"pf_log_{track}_{mu_id}_t{trial}_s{seed}.csv"
        )
        pf_logger = PFLogger(output_path=pf_log_path)
        log.info("Particle filter initialized with %d particles", pf_config.max_particles)

        step = 0
        reason = "max_steps"
        step_times = []
        pf_durations = []
        mcts_budgets = []
        weight_values = []
        prev_state = None
        prev_progress = None
        accumulated_progress = 0.0
        prev_laps = 0
        lap_step_start = 0
        lap_times_list: list = []

        controller = api._controller

        try:
            while step < max_steps:
                t0 = time.monotonic()

                state = api.get_state()

                # --- Particle filter localization ---
                scan = api.get_scan()
                if prev_state is not None:
                    dx_global = state.x - prev_state.x
                    dy_global = state.y - prev_state.y
                    dtheta = state.psi - prev_state.psi
                    cos_prev = np.cos(-prev_state.psi)
                    sin_prev = np.sin(-prev_state.psi)
                    dx_local = cos_prev * dx_global - sin_prev * dy_global
                    dy_local = sin_prev * dx_global + cos_prev * dy_global
                    odom_delta = np.array([dx_local, dy_local, dtheta])
                else:
                    odom_delta = np.array([0.0, 0.0, 0.0])

                pf_pose, pf_duration_ms = pf.update(scan, odom_delta, scan_angles)
                remaining_budget_ms = dt_ms - pf_duration_ms
                remaining_s = remaining_budget_ms / 1000.0

                pf_logger.log(
                    gt_x=state.x, gt_y=state.y, gt_psi=state.psi,
                    pf_x=pf_pose[0], pf_y=pf_pose[1], pf_psi=pf_pose[2],
                    pf_duration_ms=pf_duration_ms,
                    dt_ms=dt_ms,
                )
                pf_durations.append(pf_duration_ms)

                prev_state = state

                # --- MCTS planning with remaining budget ---
                cmd = api.compute_action(state, waypoints, remaining_budget_s=remaining_s)

                # Track actual MCTS budget used
                actual_mcts_budget = max(remaining_s - overhead_s, 0.001)
                mcts_budgets.append(actual_mcts_budget * 1000)

                cs_raw = (
                    controller.get_cornering_stiffness()
                    if hasattr(controller, "get_cornering_stiffness")
                    else float("nan")
                )
                max_kappa = getattr(controller, "_max_kappa_lookahead", float("nan"))
                avg_kappa = getattr(controller, "_avg_kappa_lookahead", float("nan"))

                weights = controller.get_last_weights() if hasattr(controller, "get_last_weights") else (1/3, 1/3, 1/3)
                weight_values.append(weights)

                episode_logger.log_step(
                    state, cmd,
                    float("nan"), float("nan"), float("nan"),
                    c_alpha=cs_raw,
                    max_kappa=max_kappa,
                    avg_kappa=avg_kappa,
                )
                done = api.apply_action(cmd)

                if render:
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
                    avg_pf = np.mean(pf_durations[-100:])
                    avg_mcts = np.mean(mcts_budgets[-100:])
                    log.info(
                        "[%s mu=%.4f] step=%d  v=%.2f  progress=%.1f%%  "
                        "pf=%.1fms  mcts_budget=%.1fms",
                        track, mu, step, state.v, pct, avg_pf, avg_mcts,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info(
                        "[%s mu=%.4f] Finished at step %d (%s)",
                        track, mu, step, reason,
                    )
                    break

                sleep_time = dt_s - elapsed
                if sleep_time > 0 and render:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.4f] Interrupted at step %d", track, mu, step)

        pf_logger.close()

        laps = int(accumulated_progress / track_obj.total_length)

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

        output_dir = str(RESULTS_DIR / f"localize_mcts_{track}_{mu_id}")
        config_name = f"{BASE_CONFIG.stem}_localize_t{trial}_s{seed}"

        if weight_values:
            avg_p_agg = float(np.mean([w[0] for w in weight_values]))
            avg_p_slow = float(np.mean([w[1] for w in weight_values]))
            avg_p_feas = float(np.mean([w[2] for w in weight_values]))
        else:
            avg_p_agg = avg_p_slow = avg_p_feas = 1 / 3

        # Compute localization RMSE
        import csv, math
        with open(pf_log_path) as f:
            pf_rows = list(csv.DictReader(f))
        n = len(pf_rows) if pf_rows else 1
        rmse_x = math.sqrt(sum((float(r['gt_x']) - float(r['pf_x']))**2 for r in pf_rows) / n)
        rmse_y = math.sqrt(sum((float(r['gt_y']) - float(r['pf_y']))**2 for r in pf_rows) / n)
        rmse_psi = math.sqrt(sum(((float(r['gt_psi']) - float(r['pf_psi']) + math.pi) % (2 * math.pi) - math.pi)**2 for r in pf_rows) / n)

        extra_meta = {
            "trial": trial,
            "seed": seed,
            "localize_budget": True,
            "avg_pf_duration_ms": float(np.mean(pf_durations)) if pf_durations else 0.0,
            "avg_mcts_budget_ms": float(np.mean(mcts_budgets)) if mcts_budgets else 0.0,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_psi": rmse_psi,
            "avg_p_agg": avg_p_agg,
            "avg_p_slow": avg_p_slow,
            "avg_p_feas": avg_p_feas,
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
            "trial": trial,
            "seed": seed,
            "avg_pf_ms": float(np.mean(pf_durations)) if pf_durations else 0.0,
            "avg_mcts_budget_ms": float(np.mean(mcts_budgets)) if mcts_budgets else 0.0,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_psi": rmse_psi,
            "avg_p_agg": avg_p_agg,
            "avg_p_slow": avg_p_slow,
            "avg_p_feas": avg_p_feas,
            "steps": step,
            "reason": reason,
            "collision": reason == "collision",
            "laps": laps,
            "progress_pct": progress_pct,
            "avg_step_ms": avg_dt * 1000,
            "lap_times": lap_times_list,
            "csv": csv_path,
            "pf_log": pf_log_path,
            "status": "OK",
        }
    finally:
        os.unlink(fixed_map_config_path)
        if tmp_algo_config_path is not None:
            try:
                os.unlink(tmp_algo_config_path)
            except OSError:
                pass


def run_experiment_in_subprocess(track, mu, seed, trial, max_steps, render, overhead_s) -> dict:
    """Run a single experiment in a child process to isolate segfaults."""
    result_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="localize_result_",
    )
    result_file.close()
    result_path = result_file.name

    cmd = [
        sys.executable, __file__,
        "--_subprocess",
        "--track", track,
        "--mu", str(mu),
        "--seed", str(seed),
        "--trial", str(trial),
        "--max-steps", str(max_steps),
        "--result-file", result_path,
        "--overhead", str(overhead_s),
    ]
    if render:
        cmd.append("--render")

    try:
        proc = subprocess.run(cmd, timeout=max_steps * 2)
        retcode = proc.returncode
    except subprocess.TimeoutExpired:
        retcode = -99

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

    if retcode == -signal.SIGSEGV or retcode == -11:
        reason = "segfault"
    elif retcode == -signal.SIGABRT or retcode == -6:
        reason = "sigabrt"
    elif retcode == -99:
        reason = "timeout"
    else:
        reason = f"crash (exit {retcode})"

    log.error(
        "[%s mu=%.4f trial=%d] Subprocess died: %s",
        track, mu, trial, reason,
    )
    return {
        "track": track, "mu": mu,
        "trial": trial, "seed": seed,
        "avg_pf_ms": 0.0, "avg_mcts_budget_ms": 0.0,
        "rmse_x": 0.0, "rmse_y": 0.0, "rmse_psi": 0.0,
        "avg_p_agg": 0.0, "avg_p_slow": 0.0, "avg_p_feas": 0.0,
        "steps": 0, "reason": reason, "collision": False,
        "laps": 0, "progress_pct": 0.0, "avg_step_ms": 0.0,
        "lap_times": [], "csv": "", "pf_log": "",
        "status": "CRASHED",
    }


def _subprocess_main():
    """Entry point for subprocess: run one experiment and write result to JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--_subprocess", action="store_true")
    parser.add_argument("--track", required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--overhead", type=float, default=0.010)
    args = parser.parse_args()

    result = run_experiment(
        args.track, args.mu,
        args.seed, args.trial, args.max_steps, args.render,
        overhead_s=args.overhead,
    )
    with open(args.result_file, "w") as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(
        description="Localization-Budget-Aware MCTS experiments"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=10, help="Trials per combo")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries for crashed/segfaulted trials")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--track", type=str, nargs="+", default=None, help="Filter by track")
    parser.add_argument("--mu", type=float, nargs="+", default=None, help="Filter by mu")
    parser.add_argument("--overhead", type=float, default=0.010,
                        help="MCTS overhead in seconds (subtracted from remaining budget)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tracks = args.track if args.track else TRACKS
    mus = args.mu if args.mu else MU_CONFIGS

    results = []
    total = len(tracks) * len(mus) * args.num_trials
    idx = 0

    for trial in range(1, args.num_trials + 1):
        seed = args.base_seed + trial - 1
        for track in tracks:
            for mu in mus:
                idx += 1
                log.info(
                    "\n>>> Run %d/%d: %s  mu=%.4f  localize_budget  trial=%d  seed=%d",
                    idx, total, track, mu, trial, seed,
                )
                result = run_experiment_in_subprocess(
                    track, mu, seed, trial, args.max_steps, args.render,
                    overhead_s=args.overhead,
                )
                # Retry crashed trials
                retries = 0
                while result.get("status") == "CRASHED" and retries < args.max_retries:
                    retries += 1
                    log.warning(
                        "Retry %d/%d for %s mu=%.4f trial=%d (was: %s)",
                        retries, args.max_retries, track, mu,
                        trial, result.get("reason", "?"),
                    )
                    result = run_experiment_in_subprocess(
                        track, mu, seed, trial, args.max_steps, args.render,
                        overhead_s=args.overhead,
                    )
                results.append(result)
                print_summary_table(results)

    # Final summary
    print_summary_table(results)


def print_summary_table(results):
    """Print summary table of all completed results so far."""
    w = 165
    print("\n" + "=" * w)
    print("LOCALIZATION-BUDGET-AWARE MCTS EXPERIMENT RESULTS")
    print("=" * w)
    print(
        f"{'Track':<14} {'mu':>7} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'PF ms':>7} {'MCTS ms':>8} "
        f"{'RMSE_x':>8} {'RMSE_y':>8} {'RMSE_ψ':>8}  {'Lap times (s)'}"
    )
    print("-" * w)

    total_runs = 0
    total_collisions = 0
    total_crashes = 0

    for r in results:
        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<14} {r['mu']:>7.4f}  "
                f"--- SKIPPED ({r.get('reason','')}) ---"
            )
            continue

        if r["status"] == "CRASHED":
            total_crashes += 1
            print(
                f"{r['track']:<14} {r['mu']:>7.4f} "
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
            f"{r['track']:<14} {r['mu']:>7.4f} "
            f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d} {r['steps']:>7d} "
            f"{col_str:<10} {r['laps']:>5d} {r['progress_pct']:>8.1f}% "
            f"{r['reason']:<12} {r['avg_step_ms']:>8.2f}ms "
            f"{r['avg_pf_ms']:>6.1f}ms {r['avg_mcts_budget_ms']:>7.1f}ms "
            f"{r['rmse_x']:>8.4f} {r['rmse_y']:>8.4f} {r['rmse_psi']:>8.4f}  [{lap_str}]"
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
    if "--_subprocess" in sys.argv:
        _subprocess_main()
    else:
        main()
