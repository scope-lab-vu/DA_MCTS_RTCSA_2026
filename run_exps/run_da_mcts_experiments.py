"""
Run Budget-Aware MLP-Prob-Pred MCTS experiments sweeping over mcts_time_budget values.

Combines GP variance sampling at chance nodes (stochastic outcomes) with
budget-aware action sampling where action weights (p_agg, p_slow, p_feas) are
predicted by a small MLP running inline in C++.

Sweeps time budgets to characterise how the MLP-predicted action weights
interact with the stochastic chance-node expansion on driving performance.

Usage:
    python run_da_mcts_experiments.py                        # all combos, no render
    python run_da_mcts_experiments.py --render               # with visualisation
    python run_da_mcts_experiments.py --time-budget 0.030    # single budget
    python run_da_mcts_experiments.py --track YasMarina       # single track
    python run_da_mcts_experiments.py --mu 1.0 0.3           # subset of mu values
    python run_da_mcts_experiments.py --max-steps 5000       # limit episode length

EXTRA (contention study) -- controlled with environment variables, which the
child processes inherit:
    DAMCTS_MODE=da|anytime|fixed   planner arm (default da)
        da       full DA-MCTS: timed search + MLP-adapted sampling weights
        anytime  timed search, weights stay 1/3 each (warmup set huge)
        fixed    fixed work: exactly DAMCTS_N_ITERS iterations, no time limit
    DAMCTS_N_ITERS=1000            iterations for fixed mode
    DAMCTS_LATE=1|0                late-command model on (default) / off
    DAMCTS_TAG=name                label for the condition (e.g. heavy_s1)
    DAMCTS_FAKE_DELAY_MS=0         test only: extra sleep on every 10th plan
    DAMCTS_WARMUP_DECISIONS=40     first N decisions: no late model, excluded from stats
"""

# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

import argparse
import csv
import json
import logging
import math
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
log = logging.getLogger("budget_aware_prob_pred_experiments")

# TRACKS = ["YasMarina"]
TRACKS = ["Sepang", "YasMarina"]
# TRACKS = ["Sepang"]


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
# Sweep over time budgets (seconds)
# TIME_BUDGETS = [
#     0.001,  # tight budget — will surely fail
#     0.003,
#     0.006,
#     0.009,
#     0.012,
#     0.015,
#     0.018,
#     0.021,
#     0.024,
#     0.027,
#     0.030,
# ]


TIME_BUDGETS = [
    0.001,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
]

BASE_CONFIG = Path("params/config_da_mcts.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results")

# ── Contention-study settings (read from environment) ────────────────────
MODE = os.environ.get("DAMCTS_MODE", "da").lower()
N_ITERS = int(os.environ.get("DAMCTS_N_ITERS", "1000"))
LATE_MODEL = os.environ.get("DAMCTS_LATE", "1") == "1"
TAG = os.environ.get("DAMCTS_TAG", "")
FAKE_DELAY_S = float(os.environ.get("DAMCTS_FAKE_DELAY_MS", "0")) / 1000.0
# The first planning calls include one-time setup (tree build, cache warm-up; measured
# at ~4 s for the very first real plan). They are not deadline misses: the late model is
# not applied to them and they are excluded from the deadline statistics.
WARMUP_DECISIONS = int(os.environ.get("DAMCTS_WARMUP_DECISIONS", "40"))
if MODE not in ("da", "anytime", "fixed"):
    raise SystemExit(f"DAMCTS_MODE must be da, anytime or fixed (got '{MODE}')")


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
        prefix=f"map_prob_pred_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_config_with_time_budget(time_budget: float) -> Path:
    """Return a temp algo config YAML with the given mcts_time_budget.

    The planner arm (MODE) also changes the config:
      da       -> nothing else changes
      anytime  -> budget_n_warmup is made huge, so the MLP never takes over and
                  the sampling weights stay at 1/3 each
      fixed    -> mcts_time_budget=0 and n_simulations=N_ITERS (fixed work)
    """
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["mcts_time_budget"] = time_budget

    if MODE == "anytime":
        if "budget_n_warmup" not in cfg["controller"]:
            raise KeyError(
                "budget_n_warmup not found under 'controller' in the config. "
                "Find the real key name with: grep -n warmup params/config_da_mcts.yaml"
            )
        cfg["controller"]["budget_n_warmup"] = 1_000_000
    elif MODE == "fixed":
        cfg["controller"]["mcts_time_budget"] = 0.0
        cfg["controller"]["n_simulations"] = N_ITERS

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"config_prob_pred_tb{int(time_budget*1000)}ms_",
        dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_experiment(track, mu, seed, trial, max_steps, render, time_budget) -> dict:
    """Run a single experiment. Returns summary dict."""
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
        "EXPERIMENT: %s  mu=%.4f  budget=%dms  trial=%d  seed=%d  mode=%s  late_model=%s  tag=%s",
        track, mu, int(time_budget * 1000), trial, seed, MODE, LATE_MODEL, TAG or "-",
    )
    log.info("=" * 60)

    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, track)
    tmp_algo_config_path = make_config_with_time_budget(time_budget)

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
        weight_values = []  # list of (p_agg, p_slow, p_feas) tuples
        prev_progress = None
        accumulated_progress = 0.0
        prev_laps = 0
        lap_step_start = 0
        lap_times_list: list = []

        # timing log: one row per planning decision
        # (decision, sim_step, wall_time, plan_time_s, n_iters, budget_s, met_deadline, late_steps, in_warmup)
        timing_rows = []
        decision_idx = 0
        prev_cmd = None  # last command that was applied; held while a plan is late
        applied_sv = []   # every command actually applied (incl. held ones), for smoothness
        applied_acc = []

        controller = api._controller

        try:
            while step < max_steps:
                t0 = time.monotonic()

                state = api.get_state()

                # ---- time the whole planning call (Python prep + C++ search + extraction)
                controller._last_n_sims = -1   # stays -1 if the controller returns early (startup)
                t_wall = time.time()
                t_plan0 = time.perf_counter()
                cmd = api.compute_action(state, waypoints)
                if FAKE_DELAY_S > 0 and decision_idx % 10 == 5:
                    time.sleep(FAKE_DELAY_S)   # test hook: pretend the plan was slow
                plan_time = time.perf_counter() - t_plan0
                decision_idx += 1

                n_iters = getattr(controller, "_last_n_sims", -1)
                budget_s = getattr(controller, "_last_budget_s", float("nan"))

                # ---- late-command model: a plan that overruns dt_s means the car keeps
                # driving on the PREVIOUS command for the extra periods.
                in_warmup = decision_idx <= WARMUP_DECISIONS
                late_steps = (
                    max(0, math.ceil(plan_time / dt_s) - 1)
                    if (LATE_MODEL and not in_warmup) else 0
                )
                timing_rows.append((
                    decision_idx, step, t_wall, plan_time, n_iters, budget_s,
                    int(plan_time <= dt_s), late_steps, int(in_warmup),
                ))

                if prev_cmd is None:
                    prev_cmd = type(cmd)(acceleration=0.0, steering_vel=0.0)
                cmds_to_apply = [prev_cmd] * late_steps + [cmd]
                prev_cmd = cmd

                cs_raw = (
                    controller.get_cornering_stiffness()
                    if hasattr(controller, "get_cornering_stiffness")
                    else float("nan")
                )
                max_kappa = getattr(controller, "_max_kappa_lookahead", float("nan"))
                avg_kappa = getattr(controller, "_avg_kappa_lookahead", float("nan"))

                weights = controller.get_last_weights() if hasattr(controller, "get_last_weights") else (1/3, 1/3, 1/3)
                weight_values.append(weights)

                done = False
                for k, c in enumerate(cmds_to_apply):
                    if k > 0:
                        state = api.get_state()   # state before each held step

                    episode_logger.log_step(
                        state, c,
                        float("nan"), float("nan"), float("nan"),
                        c_alpha=cs_raw,
                        max_kappa=max_kappa,
                        avg_kappa=avg_kappa,
                    )
                    done = api.apply_action(c)
                    applied_sv.append(float(getattr(c, "steering_vel", float("nan"))))
                    applied_acc.append(float(getattr(c, "acceleration", float("nan"))))

                    if render:
                        traj_steps = getattr(controller, 'get_best_trajectory', lambda: [])()
                        if traj_steps:
                            xy = np.array([[s['state'][0], s['state'][1]]
                                           for s in traj_steps if len(s['state']) >= 2])
                            if len(xy) > 0:
                                api.set_planned_trajectory(xy)
                        api.render()

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
                            "p_agg=%.3f p_slow=%.3f p_feas=%.3f  plan=%.4fs",
                            track, mu, int(time_budget * 1000), step,
                            state.v, pct, avg_p_agg, avg_p_slow, avg_p_feas, plan_time,
                        )

                    if done or step >= max_steps:
                        break

                elapsed = time.monotonic() - t0
                step_times.append(elapsed)

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

        # ---- save the per-decision timing log
        timing_dir = RESULTS_DIR / "timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = timing_dir / (
            f"timing_{MODE}_{TAG or 'none'}_{track}_mu{mu}_tb{int(time_budget*1000)}ms"
            f"_t{trial}_s{seed}.csv"
        )
        with open(timing_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["decision", "sim_step", "wall_time", "plan_time_s",
                        "n_iters", "budget_s", "met_deadline", "late_steps", "in_warmup"])
            w.writerows(timing_rows)

        # skip startup steps (n_iters == -1) and the warm-up window
        valid = [r for r in timing_rows if r[4] >= 0 and not r[8]]
        nan = float("nan")
        if valid:
            plan_ms = np.array([r[3] for r in valid]) * 1000.0
            hit_rate = float(np.mean([r[6] for r in valid]))
            plan_p95_ms = float(np.percentile(plan_ms, 95))
            plan_p99_ms = float(np.percentile(plan_ms, 99))
            plan_mean_ms = float(np.mean(plan_ms))
            plan_median_ms = float(np.median(plan_ms))
            plan_std_ms = float(np.std(plan_ms))
            plan_max_ms = float(np.max(plan_ms))
            severe_miss_rate = float(np.mean(plan_ms > 2.0 * dt_s * 1000.0))  # > 2 periods
            mean_iters = float(np.mean([r[4] for r in valid]))
        else:
            hit_rate = plan_p95_ms = plan_p99_ms = mean_iters = nan
            plan_mean_ms = plan_median_ms = plan_std_ms = plan_max_ms = severe_miss_rate = nan
        late_steps_total = int(sum(r[7] for r in timing_rows))

        # smoothness: mean absolute change of the applied command between consecutive steps
        def _mean_abs_diff(v):
            v = np.asarray(v, dtype=float)
            return float(np.nanmean(np.abs(np.diff(v)))) if len(v) > 1 else nan
        steer_rate_change = _mean_abs_diff(applied_sv)
        accel_change = _mean_abs_diff(applied_acc)

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
        if MODE != "da":
            tb_tag += f"_{MODE}"
        if TAG:
            tb_tag += f"_{TAG}"
        output_dir = str(RESULTS_DIR / f"prob_pred_mcts_{track}_{mu_id}{tb_tag}")
        config_name = f"{BASE_CONFIG.stem}_t{trial}_s{seed}"

        # Compute average weights over the episode
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
            "mode": MODE,
            "tag": TAG,
            "late_model": LATE_MODEL,
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
            "mode": MODE,
            "tag": TAG,
            "deadline_hit_rate": hit_rate,
            "plan_p95_ms": plan_p95_ms,
            "plan_p99_ms": plan_p99_ms,
            "mean_iters": mean_iters,
            "late_steps_total": late_steps_total,
            "plan_mean_ms": plan_mean_ms,
            "plan_median_ms": plan_median_ms,
            "plan_std_ms": plan_std_ms,
            "plan_max_ms": plan_max_ms,
            "severe_miss_rate": severe_miss_rate,
            "steer_rate_change": steer_rate_change,
            "accel_change": accel_change,
            "completed": reason == "done",
            "n_decisions": len(valid),
            "timing_csv": str(timing_path),
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
        mode="w", suffix=".json", delete=False, prefix="prob_pred_result_",
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
        description="Budget-Aware MLP-Prob-Pred MCTS experiments: sweep over time budgets"
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
                    # Retry crashed trials
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

    # Final summary
    print_summary_table(results)
    append_summary_csv(results)


def append_summary_csv(results):
    """Append one row per finished episode to results/contention_summary.csv."""
    path = RESULTS_DIR / "contention_summary.csv"
    cols = ["mode", "tag", "track", "mu", "time_budget_ms", "trial", "seed",
            "status", "reason", "collision", "completed", "laps", "steps", "progress_pct",
            "best_lap_s", "mean_lap_s", "deadline_hit_rate", "plan_mean_ms",
            "plan_median_ms", "plan_std_ms", "plan_p95_ms", "plan_p99_ms", "plan_max_ms",
            "severe_miss_rate", "mean_iters", "n_decisions", "late_steps_total", "steer_rate_change",
            "accel_change", "late_model"]
    if path.exists():
        with open(path, newline="") as f:
            old_header = next(csv.reader(f), [])
        if old_header != cols:   # never mix two layouts in one file
            path.rename(path.with_name(f"contention_summary.{int(time.time())}.old.csv"))
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(cols)
        for r in results:
            laps_t = r.get("lap_times", [])
            w.writerow([
                r.get("mode", MODE), r.get("tag", TAG), r.get("track"), r.get("mu"),
                r.get("time_budget_ms"), r.get("trial"), r.get("seed"),
                r.get("status"), r.get("reason"), r.get("collision"),
                r.get("completed", ""), r.get("laps"), r.get("steps"),
                r.get("progress_pct"),
                min(laps_t) if laps_t else "",
                float(np.mean(laps_t)) if laps_t else "",
                r.get("deadline_hit_rate", ""), r.get("plan_mean_ms", ""),
                r.get("plan_median_ms", ""), r.get("plan_std_ms", ""),
                r.get("plan_p95_ms", ""), r.get("plan_p99_ms", ""),
                r.get("plan_max_ms", ""), r.get("severe_miss_rate", ""),
                r.get("mean_iters", ""), r.get("n_decisions", ""), r.get("late_steps_total", ""),
                r.get("steer_rate_change", ""), r.get("accel_change", ""),
                LATE_MODEL,
            ])


def print_summary_table(results):
    """Print summary table of all completed results so far."""
    w = 165
    print("\n" + "=" * w)
    print(f"BUDGET-AWARE MLP-PROB-PRED MCTS EXPERIMENT RESULTS  (mode={MODE}, late_model={LATE_MODEL}, tag={TAG or '-'})")
    print("=" * w)
    print(
        f"{'Track':<14} {'mu':>7} {'Budget':>8} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'p_agg':>6} {'p_slow':>7} {'p_feas':>7} {'Hit%':>6} {'Iters':>7}  {'Lap times (s)'}"
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
        hit = r.get("deadline_hit_rate", float("nan")) * 100
        iters = r.get("mean_iters", float("nan"))
        print(
            f"{r['track']:<14} {r['mu']:>7.4f} {tb_str:>8} "
            f"{r.get('trial', 1):>6d} {r.get('seed', 0):>6d} {r['steps']:>7d} "
            f"{col_str:<10} {r['laps']:>5d} {r['progress_pct']:>8.1f}% "
            f"{r['reason']:<12} {r['avg_step_ms']:>8.2f}ms "
            f"{p_agg:>6.3f} {p_slow:>7.3f} {p_feas:>7.3f} {hit:>5.1f}% {iters:>7.0f}  [{lap_str}]"
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
