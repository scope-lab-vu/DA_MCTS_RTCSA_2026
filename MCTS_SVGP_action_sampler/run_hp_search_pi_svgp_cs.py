"""
Optuna hyperparameter search for MCTS DPW + PI-SVGP C++ CS controller.

Each Optuna trial evaluates a parameter combination across all 3 friction
levels on YasMarina.  The objective maximises average track progress while
penalising collisions.

Tuned parameters (6):
    exploration_constant : [0.10, 3.00]  step 0.01
    alpha_dpw            : [0.1,  1.0]   step 0.1
    beta_dpw             : [0.1,  1.0]   step 0.1
    discount             : [0.80, 1.00]  step 0.01
    speed_weight         : [1.0, 100.0]  step 0.1
    alpha_threshold      : [0.00, 2.00]  step 0.01

Usage:
    python run_hp_search_pi_svgp_cs.py                      # 10000 trials
    python run_hp_search_pi_svgp_cs.py --n-trials 10        # quick test
    python run_hp_search_pi_svgp_cs.py --render              # visualise
    python run_hp_search_pi_svgp_cs.py --max-steps 5000      # longer episodes
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
log = logging.getLogger("hp_search_cs")

# ── Constants ────────────────────────────────────────────────────────────────

MU_CONFIGS = [
    # (mu_value, svgp_task_index)
    (1.0489, 0),
    (0.8642, 1),
    (0.4947, 2),
]

ANCHORS = [1.0489, 0.8642, 0.4947]

BASE_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp_cs.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
TRACK = "YasMarina"
MAP_CONFIG_BASE = Path("maps/config_{}_map.yaml".format(TRACK))

COLLISION_PENALTY = 50.0  # subtracted from objective per collision


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mu_to_id(mu: float) -> str:
    return str(mu).replace(".", "")


def nearest_anchor(mu: float) -> float:
    """Map arbitrary mu to the nearest training anchor."""
    return min(ANCHORS, key=lambda a: abs(a - mu))


def _make_map_config(base_path: Path, track: str, anchor_mu: float) -> Path:
    """Create a temp map config with paths resolved to the anchor directory.

    Template variables ({track_name}, {mu_id}, {mu}) are expanded here so
    that load_map_config resolves paths correctly even from /tmp/.
    """
    anchor_mu_id = mu_to_id(anchor_mu)

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

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

    cfg["track_name"] = track
    cfg["mu_id"] = anchor_mu_id

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="map_cfg_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_algo_config(
    task_index,
    exploration_constant,
    alpha_dpw,
    beta_dpw,
    discount,
    speed_weight,
    alpha_threshold,
):
    """Create a temp YAML with the given hyperparameters."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["controller"]["svgp_task_index"] = task_index
    cfg["controller"]["exploration_constant"] = float(exploration_constant)
    cfg["controller"]["alpha_dpw"] = float(alpha_dpw)
    cfg["controller"]["beta_dpw"] = float(beta_dpw)
    cfg["controller"]["discount"] = float(discount)
    cfg["controller"]["speed_weight"] = float(speed_weight)
    cfg["controller"]["alpha_threshold"] = float(alpha_threshold)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="hp_cs_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


# ── Single friction run ──────────────────────────────────────────────────────

def run_single_friction(
    mu, task_index, seed, max_steps, render,
    exploration_constant, alpha_dpw, beta_dpw,
    discount, speed_weight, alpha_threshold,
):
    """Run one friction config. Returns result dict."""
    anchor_mu = nearest_anchor(mu)

    config_path = make_algo_config(
        task_index, exploration_constant, alpha_dpw, beta_dpw,
        discount, speed_weight, alpha_threshold,
    )
    map_config_path = _make_map_config(MAP_CONFIG_BASE, TRACK, anchor_mu)

    try:
        set_seed(seed)
        api = F1TenthAPI(
            VEHICLE_PARAMS, config_path, map_config_path, mu_override=mu,
        )
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
                sleep_time = dt_s
                if sleep_time > 0:
                    time.sleep(sleep_time)

        progress_pct = (accumulated_progress / track_obj.total_length) * 100
        lap_time_s = step * dt_s

        return {
            "progress_pct": progress_pct,
            "collision": collision,
            "steps": step,
            "lap_time_s": lap_time_s,
        }

    finally:
        os.unlink(config_path)
        os.unlink(map_config_path)


# ── Optuna objective ─────────────────────────────────────────────────────────

def make_objective(max_steps, seed, render):
    """Create an Optuna objective closure."""

    def objective(trial):
        # Sample hyperparameters.
        exploration_constant = trial.suggest_float(
            "exploration_constant", 0.10, 3.00, step=0.01,
        )
        alpha_dpw = trial.suggest_float(
            "alpha_dpw", 0.1, 1.0, step=0.1,
        )
        beta_dpw = trial.suggest_float(
            "beta_dpw", 0.1, 1.0, step=0.1,
        )
        discount = trial.suggest_float(
            "discount", 0.80, 1.00, step=0.01,
        )
        speed_weight = trial.suggest_float(
            "speed_weight", 1.0, 100.0, step=0.1,
        )
        alpha_threshold = trial.suggest_float(
            "alpha_threshold", 0.00, 2.00, step=0.01,
        )

        log.info(
            "Trial %d: ec=%.2f  a=%.1f  b=%.1f  d=%.2f  sw=%.1f  at=%.2f",
            trial.number,
            exploration_constant, alpha_dpw, beta_dpw,
            discount, speed_weight, alpha_threshold,
        )

        total_progress = 0.0
        num_collisions = 0
        all_lap_times = []

        for mu, task_index in MU_CONFIGS:
            result = run_single_friction(
                mu, task_index, seed, max_steps, render,
                exploration_constant, alpha_dpw, beta_dpw,
                discount, speed_weight, alpha_threshold,
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
        avg_progress = total_progress / len(MU_CONFIGS)
        avg_lap_time = float(np.mean(all_lap_times))
        score = avg_progress - COLLISION_PENALTY * num_collisions

        trial.set_user_attr("avg_progress", round(avg_progress, 2))
        trial.set_user_attr("num_collisions", num_collisions)
        trial.set_user_attr("avg_lap_time_s", round(avg_lap_time, 2))

        log.info(
            "  Trial %d: score=%.1f  progress=%.1f%%  collisions=%d  "
            "lap_time=%.1fs",
            trial.number, score, avg_progress, num_collisions, avg_lap_time,
        )
        return score

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HP search for MCTS PI-SVGP C++ CS controller",
    )
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--study-name", type=str,
        default="mcts_pi_svgp_cs_hp_search",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Optuna storage URL (e.g. sqlite:///hp_search.db). "
             "Default: sqlite in results/hp_search_pi_svgp_cs/.",
    )
    args = parser.parse_args()

    storage = args.db or "sqlite:///results/hp_search_pi_svgp_cs/{}.db".format(
        args.study_name,
    )
    os.makedirs("results/hp_search_pi_svgp_cs", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(args.max_steps, args.seed, args.render)
    study.optimize(objective, n_trials=args.n_trials)

    # ── Summary ──────────────────────────────────────────────────────────
    w = 110
    print("\n" + "=" * w)
    print("OPTUNA HP SEARCH COMPLETE  (MCTS PI-SVGP C++ CS)")
    print("=" * w)

    best_t = study.best_trial
    print("\nBest trial: #{}".format(best_t.number))
    print("  Score:         {:.1f}".format(study.best_value))
    print("  Avg progress:  {:.1f}%".format(
        best_t.user_attrs.get("avg_progress", 0)))
    print("  Collisions:    {}".format(
        best_t.user_attrs.get("num_collisions", "?")))
    print("  Avg lap time:  {:.1f}s".format(
        best_t.user_attrs.get("avg_lap_time_s", 0)))
    print("  Params:")
    print("    exploration_constant: {}".format(
        best_t.params["exploration_constant"]))
    print("    alpha_dpw:            {}".format(
        best_t.params["alpha_dpw"]))
    print("    beta_dpw:             {}".format(
        best_t.params["beta_dpw"]))
    print("    discount:             {}".format(
        best_t.params["discount"]))
    print("    speed_weight:         {}".format(
        best_t.params["speed_weight"]))
    print("    alpha_threshold:      {}".format(
        best_t.params["alpha_threshold"]))

    # Top 10 trials.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value, reverse=True)

    print("\nTop 10 trials (of {} completed):".format(len(completed)))
    header = (
        "  {:>4} {:>7}"
        " {:>5} {:>5} {:>5} {:>5} {:>6} {:>5}"
        " {:>7} {:>4} {:>8}"
    ).format(
        "#", "Score",
        "ec", "a_dpw", "b_dpw", "disc", "sw", "a_th",
        "Prog%", "Col", "LapTime",
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in completed[:10]:
        p = t.params
        ua = t.user_attrs
        line = (
            "  {:>4} {:>7.1f}"
            " {:>5.2f} {:>5.1f} {:>5.1f} {:>5.2f} {:>6.1f} {:>5.2f}"
            " {:>6.1f}% {:>4} {:>7.1f}s"
        ).format(
            t.number, t.value,
            p["exploration_constant"], p["alpha_dpw"], p["beta_dpw"],
            p["discount"], p["speed_weight"], p["alpha_threshold"],
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
    print("  exploration_constant: {}".format(
        best_t.params["exploration_constant"]))
    print("  alpha_dpw: {}".format(best_t.params["alpha_dpw"]))
    print("  beta_dpw: {}".format(best_t.params["beta_dpw"]))
    print("  discount: {}".format(best_t.params["discount"]))
    print("  speed_weight: {}".format(best_t.params["speed_weight"]))
    print("  alpha_threshold: {}".format(best_t.params["alpha_threshold"]))
    print("\nStudy saved to: {}".format(storage))


if __name__ == "__main__":
    main()
