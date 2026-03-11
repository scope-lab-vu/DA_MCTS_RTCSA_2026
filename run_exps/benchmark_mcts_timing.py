"""
MCTS Solver Iteration Benchmark.

For each computational budget (mcts_time_budget), runs the MCTS simulation loop
and records the number of MCTS iterations (simulations) completed per step via
learn_timed(). Reports mean iterations vs budget table.

Unlike the MPC benchmark which replays exact problem instances, MCTS requires
full simulation runs per budget since the tree state depends on the trajectory.

Usage:
    python run_exps/benchmark_mcts_timing.py --track YasMarina --mu 1.0 --max-steps 500
    python run_exps/benchmark_mcts_timing.py --track Sepang --mu 0.8 --budgets 1 3 5 10 20 30
    python run_exps/benchmark_mcts_timing.py --track YasMarina --mu 1.0 --max-steps 1000 --warmup-steps 30
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
log = logging.getLogger("benchmark_mcts_timing")

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_BUDGETS_MS = [2, 5, 10, 15, 20, 30, 40]
UNLIMITED_BUDGET_MS = 40  # 1 second = effectively unlimited

BASE_CONFIG = Path("params/config_mcts_budget_aware.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results/benchmarks")


def mu_to_id(mu: float) -> str:
    return str(mu).replace(".", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_map_config(track: str) -> Path:
    """Return a temp map config YAML with file paths hardcoded to mu=1.0489."""
    map_config_path = Path(f"maps/config_{track}_map.yaml")
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
        prefix=f"map_benchmark_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def make_config_with_time_budget(time_budget_s: float) -> Path:
    """Return a temp algo config YAML with the given mcts_time_budget."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["mcts_time_budget"] = time_budget_s
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"config_benchmark_tb{int(time_budget_s*1000)}ms_",
        dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


# ── Iteration capture via wrapper + monkey-patch ─────────────────────────
# The C++ pybind11 tree object doesn't allow attribute assignment, so we
# wrap it in a Python object that intercepts learn_timed() to record
# iteration counts, delegating all other methods via __getattr__.
# No existing source files are modified.

_iteration_log = []  # Global list, reset per experiment


class _TreeWrapper:
    """Thin wrapper around the C++ MCTS tree to capture learn_timed() results."""

    __slots__ = ('_tree',)

    def __init__(self, tree):
        object.__setattr__(self, '_tree', tree)

    def learn_timed(self, budget):
        n = self._tree.learn_timed(budget)
        _iteration_log.append(n)
        return n

    def __getattr__(self, name):
        return getattr(self._tree, name)

    def __setattr__(self, name, value):
        if name in ('_tree',):
            object.__setattr__(self, name, value)
        else:
            setattr(self._tree, name, value)


def _install_iteration_capture(controller):
    """Monkey-patch controller._make_tree to wrap trees in _TreeWrapper."""
    global _iteration_log
    original_make_tree = controller._make_tree  # bound method

    def patched_make_tree(state_np, v_ref_window):
        tree = original_make_tree(state_np, v_ref_window)
        return _TreeWrapper(tree)

    controller._make_tree = patched_make_tree


# ── Single benchmark run ─────────────────────────────────────────────────

def run_benchmark(track, mu, seed, max_steps, time_budget_s, warmup_steps):
    """Run a single benchmark experiment. Returns per-step data and summary."""
    global _iteration_log
    _iteration_log = []

    set_seed(seed)

    map_config_path = make_map_config(track)
    algo_config_path = make_config_with_time_budget(time_budget_s)
    budget_ms = time_budget_s * 1000

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, algo_config_path, map_config_path,
                         mu_override=mu)
        waypoints = api.load_waypoints()

        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        controller = api._controller
        _install_iteration_capture(controller)

        dt_s = api.algo_config.simulation.dt_s

        step = 0
        reason = "max_steps"
        step_data = []
        tau_values = []
        prev_progress = None
        accumulated_progress = 0.0

        try:
            while step < max_steps:
                t0 = time.perf_counter()
                iter_before = len(_iteration_log)

                state = api.get_state()
                cmd = api.compute_action(state, waypoints)

                elapsed = time.perf_counter() - t0

                # Get iteration count for this step
                iter_after = len(_iteration_log)
                if iter_after > iter_before:
                    n_sims = _iteration_log[-1]
                else:
                    n_sims = 0  # startup phase (no learn_timed called)

                tau = (controller.get_last_tau()
                       if hasattr(controller, "get_last_tau") else 0.0)
                tau_values.append(tau)

                done = api.apply_action(cmd)

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

                step_data.append({
                    'step': step,
                    'n_sims': n_sims,
                    'tau': round(tau, 6),
                    'velocity': round(state.v, 4),
                    'wall_time_ms': round(elapsed * 1000, 4),
                })

                step += 1

                if step % 500 == 0:
                    pct = (accumulated_progress / track_obj.total_length) * 100
                    recent = [d['n_sims'] for d in step_data[-100:]
                              if d['n_sims'] > 0]
                    avg_sims = np.mean(recent) if recent else 0
                    log.info(
                        "[%s mu=%.4f %dms] step=%d  v=%.2f  progress=%.1f%%  "
                        "avg_sims=%.0f  tau=%.3f  dt=%.4fs",
                        track, mu, int(budget_ms), step,
                        state.v, pct, avg_sims, tau, elapsed,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info(
                        "[%s mu=%.4f %dms] Finished at step %d (%s)",
                        track, mu, int(budget_ms), step, reason,
                    )
                    break

        except KeyboardInterrupt:
            reason = "interrupted"

        progress_pct = (accumulated_progress / track_obj.total_length) * 100

        return {
            'step_data': step_data,
            'total_steps': step,
            'reason': reason,
            'collision': reason == "collision",
            'progress_pct': round(progress_pct, 2),
            'avg_tau': round(float(np.mean(tau_values)) if tau_values
                             else 0.0, 6),
        }

    finally:
        try:
            os.unlink(map_config_path)
        except OSError:
            pass
        try:
            os.unlink(algo_config_path)
        except OSError:
            pass


def compute_stats(step_data, warmup_steps):
    """Compute statistics from step data, excluding warmup and startup steps."""
    sims = [d['n_sims'] for d in step_data[warmup_steps:] if d['n_sims'] > 0]
    walls = [d['wall_time_ms'] for d in step_data[warmup_steps:]
             if d['n_sims'] > 0]
    taus = [d['tau'] for d in step_data[warmup_steps:] if d['n_sims'] > 0]

    if not sims:
        return {
            'mean_iters': 0, 'std_iters': 0, 'min_iters': 0,
            'p50_iters': 0, 'p95_iters': 0, 'max_iters': 0,
            'mean_wall_ms': 0, 'mean_tau': 0,
        }

    arr = np.array(sims)
    wall_arr = np.array(walls)
    tau_arr = np.array(taus)

    return {
        'mean_iters': round(float(arr.mean()), 2),
        'std_iters': round(float(arr.std()), 2),
        'min_iters': int(arr.min()),
        'p50_iters': int(np.median(arr)),
        'p95_iters': int(np.percentile(arr, 95)),
        'max_iters': int(arr.max()),
        'mean_wall_ms': round(float(wall_arr.mean()), 2),
        'mean_tau': round(float(tau_arr.mean()), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MCTS solver iteration benchmark"
    )
    parser.add_argument("--track", type=str, default="YasMarina",
                        help="Track name (YasMarina, Sepang, Shanghai)")
    parser.add_argument("--mu", type=float, default=1.0489,
                        help="Friction coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=30,
                        help="Initial steps to exclude from stats")
    parser.add_argument("--budgets", type=float, nargs="+", default=None,
                        help="Computational budgets in ms to test")
    parser.add_argument("--unlimited-budget", type=float,
                        default=UNLIMITED_BUDGET_MS,
                        help="Budget in ms for 'unlimited' baseline "
                             "(default: 1000)")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    budgets_ms = args.budgets if args.budgets else DEFAULT_BUDGETS_MS
    unlimited_ms = args.unlimited_budget
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mu_id = mu_to_id(args.mu)
    all_results = {}  # budget_ms -> {step_data, stats, ...}

    # ── Phase 1: Unlimited budget baseline ───────────────────────────────
    print(f"\n{'='*80}")
    print("MCTS SOLVER ITERATION BENCHMARK")
    print(f"  Track: {args.track},  mu: {args.mu},  seed: {args.seed}")
    print(f"  max_steps: {args.max_steps},  warmup: {args.warmup_steps}")
    print(f"  Budgets (ms): {budgets_ms}")
    print(f"  Unlimited budget: {unlimited_ms}ms")
    print(f"{'='*80}")

    print(f"\n>>> Phase 1: Unlimited baseline ({unlimited_ms}ms budget)")
    t0 = time.monotonic()
    result = run_benchmark(
        args.track, args.mu, args.seed, args.max_steps,
        unlimited_ms / 1000.0, args.warmup_steps,
    )
    wall_s = time.monotonic() - t0
    stats = compute_stats(result['step_data'], args.warmup_steps)
    all_results['unlimited'] = {**result, 'stats': stats, 'wall_s': wall_s}

    print(f"  Completed: {result['total_steps']} steps, "
          f"mean_iters={stats['mean_iters']:.0f}, "
          f"progress={result['progress_pct']:.1f}%, "
          f"wall={wall_s:.1f}s")

    # ── Phase 2: Budget sweep ────────────────────────────────────────────
    print(f"\n>>> Phase 2: Budget sweep ({len(budgets_ms)} budgets)")

    for budget_ms in budgets_ms:
        print(f"\n  Budget={budget_ms}ms ... ", end='', flush=True)
        t0 = time.monotonic()
        result = run_benchmark(
            args.track, args.mu, args.seed, args.max_steps,
            budget_ms / 1000.0, args.warmup_steps,
        )
        wall_s = time.monotonic() - t0
        stats = compute_stats(result['step_data'], args.warmup_steps)
        all_results[budget_ms] = {**result, 'stats': stats, 'wall_s': wall_s}

        print(f"steps={result['total_steps']}, "
              f"mean_iters={stats['mean_iters']:.0f}, "
              f"mean_tau={stats['mean_tau']:.3f}, "
              f"progress={result['progress_pct']:.1f}%, "
              f"wall={wall_s:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════════
    w = 140
    print(f"\n{'='*w}")
    print("MCTS ITERATIONS vs COMPUTATIONAL BUDGET")
    print(f"  Track: {args.track},  mu: {args.mu},  seed: {args.seed},  "
          f"Warmup: {args.warmup_steps}")
    print(f"{'='*w}")
    print(f"  {'Budget (ms)':>12}  {'Mean Iters':>11}  {'Std':>7}  "
          f"{'Min':>5}  {'P50':>5}  {'P95':>5}  {'Max':>5}  "
          f"{'Mean Step (ms)':>15}  {'Mean Tau':>9}  "
          f"{'Steps':>7}  {'Progress':>9}  {'Reason':<12}")
    print(f"{'-'*w}")

    # Unlimited row
    s = all_results['unlimited']['stats']
    r = all_results['unlimited']
    print(f"  {'Unlimited':>12}  {s['mean_iters']:>11.1f}  {s['std_iters']:>7.1f}  "
          f"{s['min_iters']:>5d}  {s['p50_iters']:>5d}  "
          f"{s['p95_iters']:>5d}  {s['max_iters']:>5d}  "
          f"{s['mean_wall_ms']:>15.2f}  {s['mean_tau']:>9.4f}  "
          f"{r['total_steps']:>7d}  {r['progress_pct']:>8.1f}%  "
          f"{r['reason']:<12}")

    # Budget rows
    for budget_ms in budgets_ms:
        s = all_results[budget_ms]['stats']
        r = all_results[budget_ms]
        print(f"  {budget_ms:>10.0f} ms  {s['mean_iters']:>11.1f}  {s['std_iters']:>7.1f}  "
              f"{s['min_iters']:>5d}  {s['p50_iters']:>5d}  "
              f"{s['p95_iters']:>5d}  {s['max_iters']:>5d}  "
              f"{s['mean_wall_ms']:>15.2f}  {s['mean_tau']:>9.4f}  "
              f"{r['total_steps']:>7d}  {r['progress_pct']:>8.1f}%  "
              f"{r['reason']:<12}")

    print(f"{'='*w}")

    # ── Per-budget CSVs ──────────────────────────────────────────────────
    for key, res in all_results.items():
        if key == 'unlimited':
            budget_tag = f"unlimited_{int(unlimited_ms)}ms"
        else:
            budget_tag = f"{int(key)}ms"

        csv_path = (output_dir /
                    f"benchmark_mcts_{args.track}_mu{mu_id}"
                    f"_tb{budget_tag}_seed{args.seed}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "n_sims", "tau", "velocity",
                            "wall_time_ms"],
            )
            writer.writeheader()
            writer.writerows(res['step_data'])

    # ── JSON summary ─────────────────────────────────────────────────────
    summary = {
        'track': args.track,
        'mu': args.mu,
        'seed': args.seed,
        'max_steps': args.max_steps,
        'warmup_steps': args.warmup_steps,
        'unlimited': {
            'budget_ms': unlimited_ms,
            **all_results['unlimited']['stats'],
            'total_steps': all_results['unlimited']['total_steps'],
            'progress_pct': all_results['unlimited']['progress_pct'],
            'reason': all_results['unlimited']['reason'],
        },
        'budgets': {},
    }

    ul_mean = all_results['unlimited']['stats']['mean_iters']

    for budget_ms in budgets_ms:
        res = all_results[budget_ms]
        norm_iters = round(res['stats']['mean_iters'] / ul_mean * 100, 2) if ul_mean > 0 else 0.0
        summary['budgets'][f'{int(budget_ms)}ms'] = {
            **res['stats'],
            'normalized_iters_pct': norm_iters,
            'total_steps': res['total_steps'],
            'progress_pct': res['progress_pct'],
            'reason': res['reason'],
            'avg_tau': res['avg_tau'],
        }

    json_path = (output_dir /
                 f"benchmark_mcts_{args.track}_mu{mu_id}"
                 f"_seed{args.seed}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-budget CSVs saved to: {output_dir}/")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
