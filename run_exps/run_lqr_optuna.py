"""
Optuna-based hyperparameter tuning for the lqr controller (faithful port
of f1tenth_planning's LQRPlanner).

State x = [e_cg, ė_cg, θ_e, θ̇_e]; Q is diagonal, R is scalar. The
reference's defaults (q1=0.999, q3=0.0066, r=0.75) are tuned for f110-gym's
`(steer, speed)` interface at 100 Hz; on our `(accel, sv)` 25 Hz interface
they overcommand and saturate the rate-limited `sv` channel. This sweep
finds a basin that respects the actuation envelope.

All params are searched on a 0.01 grid (step=0.01) so the best trial
reproduces exactly when written to YAML.

Score = progress_pct + 100·laps + max_v
        (collision/NaN trials still rank below clean runs via the
         100·laps gap.)

Usage:
    python run_exps/run_lqr_optuna.py                     # 50 trials
    python run_exps/run_lqr_optuna.py --n 1000 --max-steps 3000
    python run_exps/run_lqr_optuna.py --track Sepang
    python run_exps/run_lqr_optuna.py --resume            # extend existing study
"""
# ── Repo-root setup ──────────────────────────────────────────────────────
import os as _os, sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
for noisy in (
    "src.controller.lqr.controller",
    "src.controller", "src.api", "src.track",
    "src.vehicle_interface.gym_interface",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_CONFIG = Path("params/config_lqr.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")


def make_temp_config(params: dict, dt_s: float | None = None) -> Path:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for k, v in params.items():
        cfg["controller"][k] = v
    if dt_s is not None:
        cfg.setdefault("simulation", {})
        cfg["simulation"]["dt_s"] = float(dt_s)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="lqr_optuna_", dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def make_fixed_map_config(track: str, mu_id: str = "10489") -> Path:
    map_config = Path(f"maps/config_{track}_map.yaml")
    with open(map_config) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_{mu_id}/{track}_map"
    cfg["centerline_path"] = f"./{track}_{mu_id}/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_{mu_id}/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"map_lqr_optuna_{track}_", dir=str(map_config.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def run_trial_episode(params: dict, track: str, mu: float, max_steps: int,
                      dt_s: float | None = None) -> dict:
    tmp_algo = make_temp_config(params, dt_s=dt_s)
    tmp_map = make_fixed_map_config(track)
    try:
        api = F1TenthAPI(VEHICLE_PARAMS, tmp_algo, tmp_map, mu_override=mu)
        waypoints = api.load_waypoints()
        track_obj = Track(api.map_config.centerline_path)
        dt_s = api.algo_config.simulation.dt_s

        accumulated = 0.0
        prev_progress = None
        prev_laps = 0
        lap_step_start = 0
        lap_times: list[float] = []
        max_speed = 0.0
        speeds: list[float] = []
        reason = "max_steps"
        step = 0
        for step in range(max_steps):
            state = api.get_state()
            if not np.isfinite(state.psi) or not np.isfinite(state.v):
                reason = "nan"
                break
            cmd = api.compute_action(state, waypoints)
            done = api.apply_action(cmd)
            speeds.append(state.v); max_speed = max(max_speed, state.v)

            pos = np.array([state.x, state.y])
            _, _, progress = track_obj.get_nearest_point(pos)
            if prev_progress is None:
                prev_progress = progress
            d = progress - prev_progress
            if d < -track_obj.total_length * 0.5:   d += track_obj.total_length
            elif d > track_obj.total_length * 0.5:  d -= track_obj.total_length
            accumulated += d
            prev_progress = progress

            laps = int(accumulated / track_obj.total_length)
            if laps > prev_laps:
                for _ in range(laps - prev_laps):
                    lap_times.append(round((step - lap_step_start) * dt_s, 3))
                    lap_step_start = step
                prev_laps = laps

            if done:
                reason = "collision" if api.collision else "done"
                break

        progress_pct = (accumulated / track_obj.total_length) * 100.0
        score = progress_pct + 100.0 * len(lap_times) + max_speed
        return {
            "score": score,
            "progress_pct": progress_pct,
            "laps": len(lap_times),
            "max_v": max_speed,
            "mean_v": float(np.mean(speeds)) if speeds else 0.0,
            "reason": reason,
            "collision": (reason == "collision"),
            "steps": step + 1,
            "lap_times_s": lap_times,
        }
    finally:
        for p in (tmp_algo, tmp_map):
            try: _os.unlink(p)
            except OSError: pass


def make_progress_callback(every: int):
    """Return an Optuna callback that prints the current best trial every
    `every` completed trials. `study.optimize(callbacks=[cb])` invokes it
    after each trial sequentially (TPE is single-threaded by default), so
    this is safe."""
    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len(study.trials)
        if every <= 0 or n_done % every != 0:
            return
        try:
            best = study.best_trial
        except ValueError:
            return  # no successful trials yet
        ua = best.user_attrs
        print(
            f"\n  ── progress @ {n_done} trials → BEST #{best.number}  "
            f"score={best.value:+.2f}  "
            f"reason={ua.get('reason', '?')}  laps={ua.get('laps', 0)}  "
            f"prog={ua.get('progress_pct', 0):.1f}%  "
            f"max_v={ua.get('max_v', 0):.2f}  "
            f"lap_times={ua.get('lap_times_s', [])}\n"
            f"     params={best.params}\n"
        )
    return cb


def make_objective(track: str, mu: float, max_steps: int,
                   dt_s: float | None = None):
    def objective(trial: optuna.Trial) -> float:
        # All params on a 0.01 grid so best params reproduce exactly when
        # written to YAML (avoids the precision-edge problem we hit with
        # Stanley before this script switched to step=0.01).
        params = {
            # Q diag: q_1 weights e_cg, q_3 weights θ_e. q_2/q_4 (rate
            # weights) are usually 0 in the reference; keep small ranges
            # to give Optuna room without dominating the search.
            "matrix_q_1":          trial.suggest_float("matrix_q_1", 0.01, 5.0,  step=0.01),
            "matrix_q_2":          trial.suggest_float("matrix_q_2", 0.0,  1.0,  step=0.01),
            "matrix_q_3":          trial.suggest_float("matrix_q_3", 0.01, 5.0,  step=0.01),
            "matrix_q_4":          trial.suggest_float("matrix_q_4", 0.0,  1.0,  step=0.01),
            # R: scalar control penalty. Reference default is 0.75; we
            # need higher R to soften commands at our 25 Hz with sv saturation.
            "matrix_r":            trial.suggest_float("matrix_r",   0.5, 100.0, step=0.5),
            # Speed-loop knobs (our adaptation; reference uses gym's
            # built-in (steer, speed) interface).
            "K_p_speed":           trial.suggest_float("K_p_speed",            0.1, 3.0, step=0.01),
            "velocity_percentage": trial.suggest_float("velocity_percentage",  0.3, 1.0, step=0.01),
            "steering_limit":      24.0,   # locked
        }
        t0 = time.time()
        try:
            r = run_trial_episode(params, track, mu, max_steps, dt_s=dt_s)
        except Exception as e:
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            return -1e9
        secs = time.time() - t0
        for k in ("progress_pct", "laps", "max_v", "mean_v",
                  "reason", "collision", "steps", "lap_times_s"):
            trial.set_user_attr(k, r[k])
        trial.set_user_attr("wall_secs", round(secs, 2))
        print(f"  trial {trial.number:>3d}  Q=({params['matrix_q_1']:.2f},"
              f"{params['matrix_q_2']:.2f},{params['matrix_q_3']:.2f},"
              f"{params['matrix_q_4']:.2f}) R={params['matrix_r']:.2f} "
              f"K_p={params['K_p_speed']:.2f} v%={params['velocity_percentage']:.2f}"
              f"  → {r['reason']:<10} laps={r['laps']} prog={r['progress_pct']:5.1f}% "
              f"max_v={r['max_v']:4.1f} score={r['score']:+7.1f} ({secs:.1f}s)")
        return r["score"]
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--track", default="YasMarina")
    parser.add_argument("--mu", type=float, default=1.0489)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/lqr_optuna"))
    parser.add_argument("--dt-s", type=float, default=None,
                        help="Override simulation.dt_s (default: from YAML, "
                             "currently 0.04). LQR's K matrix is recomputed "
                             "with this discretisation step every call, so "
                             "sweeping dt is a useful experiment.")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Print the current best trial after every N "
                             "completed trials (set 0 to disable).")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    dt_tag = f"_dt{args.dt_s:g}" if args.dt_s is not None else ""
    study_name = args.study_name or f"lqr_{args.track}{dt_tag}_n{args.n}"
    storage_url = f"sqlite:///{args.results_dir / (study_name + '.db')}"
    print(f"\nstudy: {study_name}\nstorage: {storage_url}")
    print(f"track={args.track} mu={args.mu} max_steps={args.max_steps} "
          f"dt_s={args.dt_s} n_trials={args.n} sampler=TPE seed={args.seed}\n")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=optuna.samplers.TPESampler(seed=args.seed,
                                           n_startup_trials=10),
        load_if_exists=args.resume,
        direction="maximize",
    )
    study.optimize(make_objective(args.track, args.mu, args.max_steps,
                                   dt_s=args.dt_s),
                   n_trials=args.n, show_progress_bar=False,
                   callbacks=[make_progress_callback(args.report_every)])

    print("\n" + "=" * 100)
    best = study.best_trial
    print(f"BEST TRIAL  #{best.number}  score = {best.value:+.2f}")
    print(f"  params: {best.params}")
    print(f"  reason: {best.user_attrs.get('reason')}  "
          f"laps: {best.user_attrs.get('laps')}  "
          f"progress: {best.user_attrs.get('progress_pct'):.1f}%  "
          f"max_v: {best.user_attrs.get('max_v'):.2f} m/s")

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
    print("\n" + "=" * 100)
    print("TOP 10 BY SCORE")
    print("=" * 100)
    cols = [c for c in [
        "number", "value", "params_matrix_q_1", "params_matrix_q_2",
        "params_matrix_q_3", "params_matrix_q_4", "params_matrix_r",
        "params_K_p_speed", "params_velocity_percentage",
        "user_attrs_reason", "user_attrs_laps", "user_attrs_progress_pct",
        "user_attrs_max_v",
    ] if c in df.columns]
    print(df[cols].head(10).to_string(index=False))

    csv_path = args.results_dir / f"{study_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nfull results CSV → {csv_path}")


if __name__ == "__main__":
    main()
