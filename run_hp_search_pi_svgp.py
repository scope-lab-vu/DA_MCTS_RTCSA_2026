"""
Optuna hyperparameter search for MCTS DPW + PI-SVGP C++ controller.

Each Optuna trial evaluates a parameter combination across all 3 friction
levels on YasMarina.  The objective maximises average track progress while
penalising collisions.

Tuned parameters:
    n_simulations   : {500, 750, 1000}
    speed_weight    : multiples of 5 in [5, 50]
    alpha_dpw       : [0.1, 1.0]
    beta_dpw        : [0.1, 1.0]

Usage:
    python run_hp_search_pi_svgp.py                    # 50 trials, no render
    python run_hp_search_pi_svgp.py --n-trials 100     # more trials
    python run_hp_search_pi_svgp.py --render           # visualise each run
    python run_hp_search_pi_svgp.py --max-steps 5000   # longer episodes
"""

import argparse
import logging
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("hp_search")

# ── Constants ────────────────────────────────────────────────────────────────

N_SIMULATIONS_CHOICES = [500, 750, 1000]

MU_CONFIGS = [
    # (mu_value, svgp_task_index)
    (1.0489, 0),
    (0.8642, 1),
    (0.4947, 2),
]

BASE_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
TRACK = "YasMarina"
MAP_CONFIG_PATH = Path("maps/config_{}_map.yaml".format(TRACK))

COLLISION_PENALTY = 50.0  # subtracted from objective per collision


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(task_index, n_sim, speed_weight, alpha_dpw, beta_dpw):
    """Create a temp YAML with the given hyperparameters."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["controller"]["n_simulations"] = n_sim
    cfg["controller"]["speed_weight"] = speed_weight
    cfg["controller"]["alpha_dpw"] = float(alpha_dpw)
    cfg["controller"]["beta_dpw"] = float(beta_dpw)
    cfg["controller"]["svgp_task_index"] = task_index

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="hp_search_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


# ── Single friction run ──────────────────────────────────────────────────────

def run_single_friction(mu, task_index, seed, max_steps, render,
                        n_sim, speed_weight, alpha_dpw, beta_dpw):
    """Run one friction config. Returns result dict."""
    config_path = make_config(task_index, n_sim, speed_weight, alpha_dpw, beta_dpw)
    try:
        set_seed(seed)
        api = F1TenthAPI(VEHICLE_PARAMS, config_path, MAP_CONFIG_PATH, mu_override=mu)
        waypoints = api.load_waypoints()
        track_obj = Track(api.map_config.centerline_path)

        if render:
            api.setup_rendering(waypoints)

        dt_s = api.algo_config.simulation.dt_s
        step = 0
        prev_progress = None
        accumulated_progress = 0.0
        collision = False

        while step < max_steps:
            state = api.get_state()
            cmd = api.compute_action(state, waypoints)
            api.apply_action(cmd)

            if render:
                api.render()

            step += 1

            # Track progress.
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

            if api.collision:
                collision = True
                break

            if render:
                elapsed = 0.0  # no timing overhead
                sleep_time = dt_s - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        progress_pct = (accumulated_progress / track_obj.total_length) * 100
        lap_time_s = step * dt_s  # gym (simulation) time used

        return {
            "progress_pct": progress_pct,
            "collision": collision,
            "steps": step,
            "lap_time_s": lap_time_s,
        }

    finally:
        os.unlink(config_path)


# ── Optuna objective ─────────────────────────────────────────────────────────

def make_objective(max_steps, seed, render):
    """Create an Optuna objective closure."""

    def objective(trial):
        # Sample hyperparameters.
        n_sim = trial.suggest_categorical("n_simulations", N_SIMULATIONS_CHOICES)
        speed_weight = trial.suggest_int("speed_weight", 1, 10) * 5  # 5,10,...,50
        alpha_dpw = trial.suggest_float("alpha_dpw", 0.1, 1.0, step=0.05)
        beta_dpw = trial.suggest_float("beta_dpw", 0.1, 1.0, step=0.05)

        log.info(
            "Trial %d: n_sim=%d  sw=%d  a=%.2f  b=%.2f",
            trial.number, n_sim, speed_weight, alpha_dpw, beta_dpw,
        )

        total_progress = 0.0
        num_collisions = 0
        all_lap_times = []

        for mu, task_index in MU_CONFIGS:
            result = run_single_friction(
                mu, task_index, seed, max_steps, render,
                n_sim, speed_weight, alpha_dpw, beta_dpw,
            )

            total_progress += result["progress_pct"]
            if result["collision"]:
                num_collisions += 1
            all_lap_times.append(result["lap_time_s"])

            col_str = "CRASH" if result["collision"] else "OK"
            log.info(
                "  mu=%.4f: %s  progress=%.1f%%  lap_time=%.1fs",
                mu, col_str, result["progress_pct"], result["lap_time_s"],
            )

        # Objective: average progress across 3 frictions - collision penalty.
        # Maximising progress over a fixed step budget is equivalent to
        # minimising lap time: more progress = faster car.
        avg_progress = total_progress / len(MU_CONFIGS)
        avg_lap_time = float(np.mean(all_lap_times))
        score = avg_progress - COLLISION_PENALTY * num_collisions

        trial.set_user_attr("avg_progress", round(avg_progress, 2))
        trial.set_user_attr("num_collisions", num_collisions)
        trial.set_user_attr("avg_lap_time_s", round(avg_lap_time, 2))

        log.info(
            "  Trial %d: score=%.1f  progress=%.1f%%  collisions=%d  lap_time=%.1fs",
            trial.number, score, avg_progress, num_collisions, avg_lap_time,
        )
        return score

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HP search for MCTS PI-SVGP C++",
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--study-name", type=str,
                        default="mcts_pi_svgp_hp_search")
    parser.add_argument(
        "--db", type=str, default=None,
        help="Optuna storage URL (e.g. sqlite:///hp_search.db). "
             "Default: sqlite in results/hp_search_pi_svgp/.",
    )
    args = parser.parse_args()

    storage = args.db or "sqlite:///results/hp_search_pi_svgp/{}.db".format(
        args.study_name,
    )
    os.makedirs("results/hp_search_pi_svgp", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(args.max_steps, args.seed, args.render)
    study.optimize(objective, n_trials=args.n_trials)

    # ── Summary ──────────────────────────────────────────────────────────
    w = 90
    print("\n" + "=" * w)
    print("OPTUNA HP SEARCH COMPLETE")
    print("=" * w)

    best_t = study.best_trial
    best_sw = best_t.params["speed_weight"] * 5
    print("\nBest trial: #{}".format(best_t.number))
    print("  Score:         {:.1f}".format(study.best_value))
    print("  Avg progress:  {:.1f}%".format(
        best_t.user_attrs.get("avg_progress", 0)))
    print("  Collisions:    {}".format(
        best_t.user_attrs.get("num_collisions", "?")))
    print("  Avg lap time:  {:.1f}s".format(
        best_t.user_attrs.get("avg_lap_time_s", 0)))
    print("  Params:")
    print("    n_simulations: {}".format(best_t.params["n_simulations"]))
    print("    speed_weight:  {}".format(best_sw))
    print("    alpha_dpw:     {}".format(best_t.params["alpha_dpw"]))
    print("    beta_dpw:      {}".format(best_t.params["beta_dpw"]))

    # Top 10 trials.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value, reverse=True)

    print("\nTop 10 trials (of {} completed):".format(len(completed)))
    header = (
        "  {:>4} {:>7} {:>6} {:>4} {:>6} {:>6}"
        " {:>7} {:>4} {:>8}"
    ).format("#", "Score", "n_sim", "sw", "a_dpw", "b_dpw", "Prog%", "Col", "LapTime")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in completed[:10]:
        p = t.params
        sw = p.get("speed_weight", 0) * 5
        ua = t.user_attrs
        line = (
            "  {:>4} {:>7.1f} {:>6} {:>4} {:>6.2f} {:>6.2f}"
            " {:>6.1f}% {:>4} {:>7.1f}s"
        ).format(
            t.number, t.value, p["n_simulations"], sw,
            p["alpha_dpw"], p["beta_dpw"],
            ua.get("avg_progress", 0),
            ua.get("num_collisions", "?"),
            ua.get("avg_lap_time_s", 0),
        )
        print(line)

    # Print best config as YAML snippet for easy copy-paste.
    print("\n" + "=" * w)
    print("BEST CONFIG (paste into YAML):")
    print("=" * w)
    print("controller:")
    print("  n_simulations: {}".format(best_t.params["n_simulations"]))
    print("  speed_weight: {}".format(best_sw))
    print("  alpha_dpw: {}".format(best_t.params["alpha_dpw"]))
    print("  beta_dpw: {}".format(best_t.params["beta_dpw"]))
    print("\nStudy saved to: {}".format(storage))


if __name__ == "__main__":
    main()
