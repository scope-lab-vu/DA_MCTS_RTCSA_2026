"""
Re-run only the failed (collision) experiments from the localize-budget sweep.

Imports the experiment machinery from run_localize_budget_experiments
and replays each (track, mu, trial, seed) that collided.

Usage:
    python run_exps/rerun_failed_localize_budget_experiments.py
    python run_exps/rerun_failed_localize_budget_experiments.py --max-steps 10000
    python run_exps/rerun_failed_localize_budget_experiments.py --render
"""

# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging

from run_exps.run_localize_budget_experiments import (
    run_experiment_in_subprocess,
    print_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("rerun_failed_localize_budget")

# (track, mu, trial, seed) — extracted from collision results
FAILED_EXPERIMENTS = [
    ("Sepang", 0.5, 1, 42),

    # ("YasMarina", 0.1, 1, 42),
    # ("YasMarina", 0.1, 2, 43),
    # ("YasMarina", 0.1, 3, 44),
    # ("YasMarina", 0.1, 4, 45),
    # ("YasMarina", 0.1, 5, 46),
    # ("YasMarina", 0.1, 6, 47),
    # ("YasMarina", 0.1, 7, 48),
    # ("YasMarina", 0.1, 8, 49),
    # ("YasMarina", 0.1, 9, 50),
    # ("YasMarina", 0.1, 10, 51),

    ("Sepang", 0.6, 6, 47),
    ("Sepang", 0.5, 6, 47),
    ("Sepang", 0.6, 7, 48),
    ("Sepang", 0.5, 7, 48),
    ("YasMarina", 1.0, 8, 49),
    ("YasMarina", 0.9, 8, 49),
    ("Sepang", 1.0, 9, 50),
    ("Sepang", 0.5, 10, 51),
]


def main():
    parser = argparse.ArgumentParser(
        description="Re-run failed (collision) localize-budget MCTS experiments",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overhead", type=float, default=0.010,
                        help="MCTS overhead in seconds")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    total = len(FAILED_EXPERIMENTS)
    log.info("Re-running %d failed experiments", total)

    results = []
    for idx, (track, mu, trial, seed) in enumerate(FAILED_EXPERIMENTS, 1):
        log.info(
            "\n>>> Rerun %d/%d: %s  mu=%.4f  localize_budget  trial=%d  seed=%d",
            idx, total, track, mu, trial, seed,
        )
        result = run_experiment_in_subprocess(
            track, mu, seed, trial, args.max_steps, args.render,
            overhead_s=args.overhead,
        )
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

    print_summary_table(results)


if __name__ == "__main__":
    main()
