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


# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
import math
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

# TRACKS = ["YasMarina", "Sepang"]
TRACKS = ["YasMarina"]

MU_CONFIGS = [
    # (mu, task_index)  — task_index selects the GP friction head
    (1.0489, 0),
    # (0.8642, 1),
    # (0.4947, 2),
]

VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
# DEFAULT_ALGO_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp.yaml")
DEFAULT_ALGO_CONFIG = Path("params/config_mppi_pi_svgp.yaml")

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


def _predict_single_step(dynamics, state, cmd) -> tuple:
    """Predict next (vx, vy, omega) using batched_dynamics. Returns (nan,nan,nan) on failure."""
    try:
        dt = getattr(dynamics, "_dt_torch", None)
        device = getattr(dynamics, "_device", None)
        dtype = getattr(dynamics, "_dtype", None)
        if dt is None or device is None or dtype is None:
            return float("nan"), float("nan"), float("nan")
        s = torch.tensor(
            [state.x, state.y, state.delta, state.v, state.psi, state.omega, state.beta],
            device=device, dtype=dtype,
        ).unsqueeze(0)  # (1, 7)
        a = torch.tensor(
            [cmd.acceleration, cmd.steering_vel],
            device=device, dtype=dtype,
        ).unsqueeze(0)  # (1, 2)
        with torch.no_grad():
            next_s = dynamics.batched_dynamics(s, a, 0)[0]  # (7,)
        pred_v = float(next_s[3])
        pred_beta = float(next_s[6])
        pred_vx = pred_v * math.cos(pred_beta)
        pred_vy = pred_v * math.sin(pred_beta)
        pred_omega = float(next_s[5])
        return pred_vx, pred_vy, pred_omega
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _make_config_with_task_index(base_path: Path, task_index: int) -> Path:
    """Load base YAML, override svgp_task_index, return temp file path."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["svgp_task_index"] = task_index
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

    config_path = _make_config_with_task_index(algo_config, task_index)
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

        # Ensure dynamics model is set up for single-step RMSE prediction.
        dynamics = api._dynamics
        if hasattr(dynamics, "setup_torch") and getattr(dynamics, "_dt_torch", None) is None:
            dynamics.setup_torch(dt_s, "cpu", torch.float32)

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

        # RMSE accumulators for dynamics prediction quality
        rmse_vx_sq = 0.0
        rmse_vy_sq = 0.0
        rmse_omega_sq = 0.0
        rmse_n = 0

        try:
            while step < max_steps:
                t0 = time.monotonic()

                state = api.get_state()
                cmd = api.compute_action(state, waypoints)

                # Predict next state BEFORE stepping the simulator
                pred_vx, pred_vy, pred_omega = _predict_single_step(dynamics, state, cmd)

                episode_logger.log_step(state, cmd, pred_vx, pred_vy, pred_omega)
                done = api.apply_action(cmd)

                # Actual next state (already updated inside apply_action)
                next_state = api.get_state()
                actual_vx = next_state.v * math.cos(next_state.beta)
                actual_vy = next_state.v * math.sin(next_state.beta)
                actual_omega = next_state.omega

                if not math.isnan(pred_vx):
                    rmse_vx_sq += (pred_vx - actual_vx) ** 2
                    rmse_vy_sq += (pred_vy - actual_vy) ** 2
                    rmse_omega_sq += (pred_omega - actual_omega) ** 2
                    rmse_n += 1

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

                # Record per-lap times when a new lap is completed
                if laps_completed > prev_laps:
                    for _ in range(laps_completed - prev_laps):
                        lap_time_s = (step - lap_step_start) * dt_s
                        lap_times_list.append(round(lap_time_s, 3))
                        lap_step_start = step
                    prev_laps = laps_completed

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

        # Compute RMSE
        rmse_vx = math.sqrt(rmse_vx_sq / rmse_n) if rmse_n > 0 else float("nan")
        rmse_vy = math.sqrt(rmse_vy_sq / rmse_n) if rmse_n > 0 else float("nan")
        rmse_omega = math.sqrt(rmse_omega_sq / rmse_n) if rmse_n > 0 else float("nan")

        # Post-loop: reconcile against gym's authoritative lap count.
        # The gym terminates when the car enters its finish zone (dist <= 0.1 m
        # from start), which can be slightly before accumulated_progress crosses
        # N * total_length (e.g. 199.9% → int = 1 even though 2 laps are done).
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

        # Save trajectory (filename: traj_{config}_{track}_{mu_id}_{timestamp}.csv)
        output_dir = str(RESULTS_DIR / f"{track}_{mu_id}")
        config_name = algo_config.stem
        extra_meta = {
            "rmse_vx": f"{rmse_vx:.6f}",
            "rmse_vy": f"{rmse_vy:.6f}",
            "rmse_omega": f"{rmse_omega:.6f}",
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
            "rmse_vx": rmse_vx,
            "rmse_vy": rmse_vy,
            "rmse_omega": rmse_omega,
            "lap_times": lap_times_list,
            "csv": csv_path,
            "status": "OK",
        }
    finally:
        os.unlink(config_path)


def main():
    parser = argparse.ArgumentParser(description="MCTS DPW batch experiments")
    parser.add_argument("--algo-config", type=Path, default=DEFAULT_ALGO_CONFIG, help="Algorithm config YAML")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per (track, mu) combo")
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
    w = 140
    print("\n" + "=" * w)
    print("PER-TRIAL RESULTS")
    print("=" * w)
    print(
        f"{'Track':<12} {'mu':>7} {'Trial':>6} {'Seed':>6} {'Steps':>7} "
        f"{'Collision':<10} {'Laps':>5} {'Progress':>9} {'Reason':<12} {'ms/step':>9} "
        f"{'RMSE_vx':>9} {'RMSE_vy':>9} {'RMSE_w':>9} {'Lap times (s)'}"
    )
    print("-" * w)

    for r in results:
        if r["status"] == "SKIPPED":
            print(
                f"{r['track']:<12} {r['mu']:>7.4f} {r['trial']:>6d} {r['seed']:>6d} "
                f"{'---':>7} {'---':<10} {'---':>5} {'---':>9} {r.get('reason',''):<12} {'---':>9} "
                f"{'---':>9} {'---':>9} {'---':>9}"
            )
            continue

        col_str = "YES" if r["collision"] else "No"
        rmse_vx = r.get("rmse_vx", float("nan"))
        rmse_vy = r.get("rmse_vy", float("nan"))
        rmse_w = r.get("rmse_omega", float("nan"))
        lap_str = "  ".join(f"{t:.1f}s" for t in r.get("lap_times", []))
        print(
            f"{r['track']:<12} {r['mu']:>7.4f} {r['trial']:>6d} {r['seed']:>6d} "
            f"{r['steps']:>7d} {col_str:<10} {r['laps']:>5d} "
            f"{r['progress_pct']:>8.1f}% {r['reason']:<12} {r['avg_step_ms']:>8.2f}ms "
            f"{rmse_vx:>9.4f} {rmse_vy:>9.4f} {rmse_w:>9.4f}  [{lap_str}]"
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
        f"{'Max Prog':>9} {'Avg ms':>9} {'RMSE_vx':>9} {'RMSE_vy':>9} {'RMSE_w':>9}"
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

            rmse_vx_vals = [t["rmse_vx"] for t in trials if not math.isnan(t.get("rmse_vx", float("nan")))]
            rmse_vy_vals = [t["rmse_vy"] for t in trials if not math.isnan(t.get("rmse_vy", float("nan")))]
            rmse_w_vals  = [t["rmse_omega"] for t in trials if not math.isnan(t.get("rmse_omega", float("nan")))]
            avg_rmse_vx = float(np.mean(rmse_vx_vals)) if rmse_vx_vals else float("nan")
            avg_rmse_vy = float(np.mean(rmse_vy_vals)) if rmse_vy_vals else float("nan")
            avg_rmse_w  = float(np.mean(rmse_w_vals))  if rmse_w_vals  else float("nan")

            total_trials += n
            total_collisions += n_col

            print(
                f"{track:<12} {mu:>7.4f} {n:>7d} {n_col:>11d} "
                f"{col_rate:>8.0f}% {avg_laps:>9.1f} {avg_prog:>8.1f}% "
                f"{min_prog:>8.1f}% {max_prog:>8.1f}% {avg_ms:>8.2f}ms "
                f"{avg_rmse_vx:>9.4f} {avg_rmse_vy:>9.4f} {avg_rmse_w:>9.4f}"
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
