"""
Optuna-based hyperparameter tuning for the stanley_method controller.

Uses TPE (Tree-structured Parzen Estimator), Optuna's default Bayesian-style
sampler. After each trial Optuna decides what to try next based on the
posterior over the score function. Typically converges faster than random
search and exposes promising regions earlier.

Sweeps `K_E`, `K_H`, `K_p_speed`, `velocity_percentage` ∈ [0.3, 1.0],
and the dynamic-lookahead params `lookahead_min_m`, `lookahead_v_gain_s`
(L_d = la_min + la_gain · v applied to the heading + speed reference).
`steering_limit` is fixed at 25°.

Usage:
    python run_exps/run_stanley_optuna.py                   # 50 trials
    python run_exps/run_stanley_optuna.py --n 200 --max-steps 2000
    python run_exps/run_stanley_optuna.py --track Sepang
    python run_exps/run_stanley_optuna.py --resume          # extend an existing study
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
from pathlib import Path

import numpy as np
import optuna
import yaml

from src.api import F1TenthAPI
from src.track import Track

# ── Quiet the noisy controller / API loggers; keep optuna chatter ────────
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
for noisy in (
    "src.controller.stanley_method.controller",
    "src.controller", "src.api", "src.track",
    "src.vehicle_interface.gym_interface",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)
# Suppress Optuna's per-trial INFO noise (it can be deafening); show summary.
optuna.logging.set_verbosity(optuna.logging.WARNING)


BASE_CONFIG = Path("params/config_stanley_method.yaml")
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
        prefix="stanley_optuna_", dir=str(BASE_CONFIG.parent),
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
        prefix=f"map_stanley_optuna_{track}_", dir=str(map_config.parent),
    )
    yaml.dump(cfg, tmp); tmp.close()
    return Path(tmp.name)


def run_trial_episode(params: dict, track: str, mu: float, max_steps: int,
                      dt_s: float | None = None) -> dict:
    """Run one episode with the given params; return scoring + telemetry."""
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
        collided = (reason == "collision")
        # Score = progress_pct + 100·laps + max_v
        #
        # Three terms with deliberately separated scales:
        #   • progress_pct  — fine-grained signal at all times.
        #   • 100·laps      — large jump per completed lap. Dominates so
        #                     completing more laps always beats completing
        #                     fewer laps (even very fast ones).
        #   • max_v         — tiebreaker. Among configs at the same lap
        #                     count + similar progress, the one that hits
        #                     a higher peak speed wins.
        # No collision penalty: the lap-completion jump already makes any
        # crash that misses a lap (progress=99%, laps=0, max_v=8 → ≈107)
        # score far below any config that completes one
        # (progress=199.9%, laps=1, max_v=11 → ≈310.9).
        score = progress_pct + 100.0 * len(lap_times) + max_speed
        return {
            "score": score,
            "progress_pct": progress_pct,
            "laps": len(lap_times),
            "max_v": max_speed,
            "mean_v": float(np.mean(speeds)) if speeds else 0.0,
            "reason": reason,
            "collision": collided,
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
    """Build the Optuna objective callable bound to track / mu / max_steps."""
    def objective(trial: optuna.Trial) -> float:
        # All tuned params are searched on a 2-decimal grid (step=0.01) so the
        # best trial reproduces exactly when written back to the YAML config.
        # The previous run-to-run variance was caused by the controller being
        # near a stability edge — full-precision Optuna params worked, but
        # rounding to 2 decimals crashed the trial. Restricting the search
        # itself to 2 decimals avoids that mismatch entirely. Dropping
        # `log=True` (incompatible with `step=...` in Optuna's suggest_float)
        # is the only tradeoff — fine here since the prior best lived in
        # the mid-range of each interval, not at the extremes.
        params = {
            "K_E":                 trial.suggest_float("K_E", 0.1, 6.0, step=0.01),
            "K_H":                 trial.suggest_float("K_H", 0.05, 5.0, step=0.01),
            "K_p_speed":           trial.suggest_float("K_p_speed", 0.1, 3.0, step=0.01),
            "velocity_percentage": trial.suggest_float("velocity_percentage", 0.3, 1.0, step=0.01),
            "steering_limit":      24.0,   # locked — slightly under the gym's 25° hard limit
            # Dynamic lookahead: heading + speed reference shifts forward by
            # L_d = lookahead_min_m + lookahead_v_gain_s · v along arc-length.
            # Search range covers "no preview" (0) up to ~5 m base + 0.6 s
            # of velocity-scaled preview (e.g. at v=10 m/s that's a 6 m bonus
            # → ~11 m total). Range upper bound reflects the corner-spacing
            # on YasMarina; on tighter tracks Optuna will stay near the low end.
            "lookahead_min_m":     trial.suggest_float("lookahead_min_m", 0.0, 5.0, step=0.01),
            "lookahead_v_gain_s":  trial.suggest_float("lookahead_v_gain_s", 0.0, 0.6, step=0.01),
        }
        t0 = time.time()
        try:
            r = run_trial_episode(params, track, mu, max_steps, dt_s=dt_s)
        except Exception as e:
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            return -1e9
        secs = time.time() - t0
        # Stash telemetry on the trial so we can pull it back at the end.
        for k in ("progress_pct", "laps", "max_v", "mean_v",
                  "reason", "collision", "steps", "lap_times_s"):
            trial.set_user_attr(k, r[k])
        trial.set_user_attr("wall_secs", round(secs, 2))
        # Per-trial console line — easier to watch progress than Optuna's default.
        print(f"  trial {trial.number:>3d}  K_E={params['K_E']:.2f} K_H={params['K_H']:.3f} "
              f"K_p={params['K_p_speed']:.2f} v%={params['velocity_percentage']:.2f} "
              f"la={params['lookahead_min_m']:.2f}+{params['lookahead_v_gain_s']:.2f}·v"
              f"  → {r['reason']:<10} laps={r['laps']} prog={r['progress_pct']:5.1f}% "
              f"max_v={r['max_v']:4.1f} score={r['score']:+7.1f} ({secs:.1f}s)")
        return r["score"]
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--track", default="YasMarina")
    parser.add_argument("--mu", type=float, default=1.0489)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", default=None,
                        help="Custom study name (default = stanley_<track>_<n>)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study (study_name must match).")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/stanley_optuna"))
    parser.add_argument("--dt-s", type=float, default=None,
                        help="Override simulation.dt_s (default: from YAML, "
                             "currently 0.04). The Stanley first-tick "
                             "lookahead and rate-limit margin both scale "
                             "with this — sweeping is a useful experiment.")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Print the current best trial after every N "
                             "completed trials (set 0 to disable).")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    dt_tag = f"_dt{args.dt_s:g}" if args.dt_s is not None else ""
    study_name = args.study_name or f"stanley_{args.track}{dt_tag}_n{args.n}"
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

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    best = study.best_trial
    print(f"BEST TRIAL  #{best.number}  score = {best.value:+.2f}")
    print(f"  params: {best.params}")
    print(f"  reason: {best.user_attrs.get('reason')}  "
          f"laps: {best.user_attrs.get('laps')}  "
          f"progress: {best.user_attrs.get('progress_pct'):.1f}%  "
          f"max_v: {best.user_attrs.get('max_v'):.2f} m/s")
    print(f"  steps: {best.user_attrs.get('steps')}  "
          f"lap_times: {best.user_attrs.get('lap_times_s')}")

    # Top 10 by score
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
    print("\n" + "=" * 100)
    print(f"TOP 10 BY SCORE")
    print("=" * 100)
    cols_pref = [c for c in [
        "number", "value", "params_K_E", "params_K_H", "params_K_p_speed",
        "params_velocity_percentage",
        "params_lookahead_min_m", "params_lookahead_v_gain_s",
        "user_attrs_reason", "user_attrs_laps", "user_attrs_progress_pct",
        "user_attrs_max_v",
    ] if c in df.columns]
    print(df[cols_pref].head(10).to_string(index=False))

    csv_path = args.results_dir / f"{study_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nfull results CSV → {csv_path}")
    print(f"resume / inspect study via:")
    print(f"  python -c \"import optuna; s = optuna.load_study(study_name='{study_name}', "
          f"storage='{storage_url}'); print(s.best_params)\"")


if __name__ == "__main__":
    main()
