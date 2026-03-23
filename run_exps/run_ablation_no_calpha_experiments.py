"""
Run ablation experiments: uniform action sampling + no C_alpha SVGP input.

Same sweep as run_budget_aware_prob_pred_experiments.py but with:
  - uniform_action_sampling: true  (no GP action prior)
  - no_c_alpha_input: true         (6D SVGP dynamics, no C_alpha feature)
  - SVGP models: *_no_c_alpha_* variants

Usage:
    python run_exps/run_ablation_uniform_no_calpha_experiments.py
    python run_exps/run_ablation_uniform_no_calpha_experiments.py --track Sepang --mu 1.0 0.5
    python run_exps/run_ablation_uniform_no_calpha_experiments.py --time-budget 0.020
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
log = logging.getLogger("ablation_uniform_no_calpha")

TRACKS = ["Sepang", "YasMarina"]

MU_CONFIGS = [
    1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
]

TIME_BUDGETS = [
    # 0.001,
    # 0.005,
    # 0.010,
    # 0.015,
    # 0.020,
    # 0.025,
    0.030,
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
        prefix=f"map_ablation_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_ablation_config(time_budget: float) -> Path:
    """Return a temp algo config with ablation flags enabled."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["mcts_time_budget"] = time_budget
    cfg["controller"]["no_c_alpha_input"] = True
    # Use no-c_alpha SVGP models (6D input)
    cfg["controller"]["svgp_omega_model_path"] = (
        "models/svgp/krasser_svgp_omega_YasMarina_VAR5_jax_Ts0.020_MPPI_data_ard_kernel_no_c_alpha_M_11.npz"
    )
    cfg["controller"]["svgp_beta_model_path"] = (
        "models/svgp/krasser_svgp_beta_YasMarina_VAR6_jax_Ts0.020_MPPI_data_ard_kernel_no_c_alpha_M_12.npz"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"config_ablation_tb{int(time_budget*1000)}ms_",
        dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, seed, trial, max_steps, render, time_budget) -> dict:
    """Run a single ablation experiment. Returns summary dict."""
    mu_id = mu_to_id(mu)
    map_config_path = Path(f"maps/config_{track}_map.yaml")

    if not map_config_path.exists():
        log.warning("Map config not found: %s — skipping", map_config_path)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "time_budget_ms": time_budget * 1000,
                "status": "SKIPPED", "reason": "no map config"}

    track_dir_10489 = Path(f"maps/{track}_10489")
    if not track_dir_10489.exists():
        log.warning("Track directory not found: %s — skipping", track_dir_10489)
        return {"track": track, "mu": mu, "trial": trial, "seed": seed,
                "time_budget_ms": time_budget * 1000,
                "status": "SKIPPED", "reason": "no track dir"}

    set_seed(seed)

    log.info("=" * 60)
    log.info(
        "ABLATION: %s  mu=%.4f  budget=%dms  trial=%d  seed=%d",
        track, mu, int(time_budget * 1000), trial, seed,
    )
    log.info("=" * 60)

    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, track)
    tmp_algo_config_path = make_ablation_config(time_budget)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, tmp_algo_config_path, fixed_map_config_path, mu_override=mu)
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
        weight_values = []
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
                cmd = api.compute_action(state, waypoints)

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
                    recent = weight_values[-100:] if weight_values else [(1/3, 1/3, 1/3)]
                    avg_p_agg = np.mean([w[0] for w in recent])
                    avg_p_slow = np.mean([w[1] for w in recent])
                    avg_p_feas = np.mean([w[2] for w in recent])
                    log.info(
                        "[%s mu=%.4f %dms] step=%d  v=%.2f  progress=%.1f%%  "
                        "p_agg=%.3f p_slow=%.3f p_feas=%.3f  dt=%.4fs",
                        track, mu, int(time_budget * 1000), step,
                        state.v, pct, avg_p_agg, avg_p_slow, avg_p_feas, elapsed,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info(
                        "[%s mu=%.4f %dms] Finished at step %d (%s)",
                        track, mu, int(time_budget * 1000), step, reason,
                    )
                    break

                sleep_time = dt_s - elapsed
                if sleep_time > 0 and render:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.4f %dms] Interrupted at step %d",
                     track, mu, int(time_budget * 1000), step)

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

        tb_tag = f"_tb{int(time_budget*1000)}ms"
        output_dir = str(RESULTS_DIR / f"ablation_uniform_no_calpha_{track}_{mu_id}{tb_tag}")
        config_name = f"{BASE_CONFIG.stem}_t{trial}_s{seed}"

        if weight_values:
            avg_p_agg = float(np.mean([w[0] for w in weight_values]))
            avg_p_slow = float(np.mean([w[1] for w in weight_values]))
            avg_p_feas = float(np.mean([w[2] for w in weight_values]))
        else:
            avg_p_agg = avg_p_slow = avg_p_feas = 1/3

        extra_meta = {
            "trial": trial,
            "seed": seed,
            "time_budget_ms": time_budget * 1000,
            "avg_p_agg": avg_p_agg,
            "avg_p_slow": avg_p_slow,
            "avg_p_feas": avg_p_feas,
            "lap_times_s": lap_times_list,
            "collision": reason == "collision",
            "ablation": "uniform_action_sampling + no_c_alpha_input",
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
            "time_budget_ms": time_budget * 1000,
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
            "status": "OK",
        }
    finally:
        os.unlink(fixed_map_config_path)
        if tmp_algo_config_path is not None:
            try:
                os.unlink(tmp_algo_config_path)
            except OSError:
                pass


def run_experiment_in_subprocess(track, mu, seed, trial, max_steps, render, time_budget) -> dict:
    """Run a single experiment in a child process to isolate segfaults."""
    result_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="ablation_result_",
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
        "--time-budget", str(time_budget),
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
        "[%s mu=%.4f %dms trial=%d] Subprocess died: %s",
        track, mu, int(time_budget * 1000), trial, reason,
    )
    return {
        "track": track, "mu": mu,
        "trial": trial, "seed": seed,
        "time_budget_ms": time_budget * 1000,
        "avg_p_agg": 0.0, "avg_p_slow": 0.0, "avg_p_feas": 0.0,
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--time-budget", type=float, required=True)
    args = parser.parse_args()

    result = run_experiment(
        args.track, args.mu,
        args.seed, args.trial, args.max_steps, args.render,
        time_budget=args.time_budget,
    )
    with open(args.result_file, "w") as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: uniform action sampling + no C_alpha SVGP input"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-trials", type=int, default=10, help="Trials per combo")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries for crashed/segfaulted trials")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--track", type=str, nargs="+", default=None, help="Filter by track")
    parser.add_argument("--mu", type=float, nargs="+", default=None, help="Filter by mu")
    parser.add_argument("--time-budget", type=float, nargs="+", default=None,
                        dest="filter_tb", help="Filter by time budget(s) in seconds")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tracks = args.track if args.track else TRACKS
    mus = args.mu if args.mu else MU_CONFIGS
    tbs = args.filter_tb if args.filter_tb else TIME_BUDGETS

    results = []
    total = len(tracks) * len(mus) * len(tbs) * args.num_trials
    idx = 0

    for trial in range(1, args.num_trials + 1):
        seed = args.base_seed + trial - 1
        for track in tracks:
            for mu in mus:
                for tb in tbs:
                    idx += 1
                    log.info(
                        "\n>>> Run %d/%d: %s  mu=%.4f  budget=%dms  trial=%d  seed=%d",
                        idx, total, track, mu, int(tb * 1000), trial, seed,
                    )
                    result = run_experiment_in_subprocess(
                        track, mu, seed, trial, args.max_steps, args.render,
                        time_budget=tb,
                    )
                    retries = 0
                    while result.get("status") == "CRASHED" and retries < args.max_retries:
                        retries += 1
                        log.warning(
                            "Retry %d/%d for %s mu=%.4f %dms trial=%d (was: %s)",
                            retries, args.max_retries, track, mu,
                            int(tb * 1000), trial, result.get("reason", "?"),
                        )
                        result = run_experiment_in_subprocess(
                            track, mu, seed, trial, args.max_steps, args.render,
                            time_budget=tb,
                        )
                    results.append(result)
                    print_summary_table(results)

    print_summary_table(results)


def print_summary_table(results):
    """Print summary table of all completed results so far."""
    w = 145
    print("\n" + "=" * w)
    print("ABLATION: UNIFORM ACTION SAMPLING + NO C_ALPHA SVGP INPUT")
    print("=" * w)
    print(
        f"{'Track':<14} {'mu':>7} {'Budget':>8} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'p_agg':>6} {'p_slow':>7} {'p_feas':>7}  {'Lap times (s)'}"
    )
    print("-" * w)

    total_runs = 0
    total_collisions = 0
    total_crashes = 0

    for r in results:
        tb_str = f"{r['time_budget_ms']:.0f}ms"

        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<14} {r['mu']:>7.4f} {tb_str:>8}  "
                f"--- SKIPPED ({r.get('reason','')}) ---"
            )
            continue

        if r["status"] == "CRASHED":
            total_crashes += 1
            print(
                f"{r['track']:<14} {r['mu']:>7.4f} {tb_str:>8} "
                f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d}  "
                f"--- CRASHED ({r['reason']}) ---"
            )
            continue

        total_runs += 1
        if r["collision"]:
            total_collisions += 1

        col_str = "YES" if r["collision"] else "No"
        lap_str = "  ".join(f"{t:.1f}s" for t in r.get("lap_times", []))
        p_agg = r.get("avg_p_agg", 1/3)
        p_slow = r.get("avg_p_slow", 1/3)
        p_feas = r.get("avg_p_feas", 1/3)
        print(
            f"{r['track']:<14} {r['mu']:>7.4f} {tb_str:>8} "
            f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d} {r['steps']:>7d} "
            f"{col_str:<10} {r['laps']:>5d} {r['progress_pct']:>8.1f}% "
            f"{r['reason']:<12} {r['avg_step_ms']:>8.2f}ms "
            f"{p_agg:>6.3f} {p_slow:>7.3f} {p_feas:>7.3f}  [{lap_str}]"
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
