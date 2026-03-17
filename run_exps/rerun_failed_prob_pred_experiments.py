"""
Re-run only the failed (collision) experiments from the prob-pred sweep.

Imports the experiment machinery from run_budget_aware_prob_pred_experiments
and replays each (track, mu, budget, trial, seed) that collided.

Usage:
    python run_exps/rerun_failed_prob_pred_experiments.py
    python run_exps/rerun_failed_prob_pred_experiments.py --max-steps 10000
    python run_exps/rerun_failed_prob_pred_experiments.py --render
"""

# ── Repo-root setup (run from any directory) ─────────────────────────────
import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702
# ─────────────────────────────────────────────────────────────────────────

import argparse
import logging

from run_exps.run_budget_aware_prob_pred_experiments import (
    run_experiment_in_subprocess,
    print_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("rerun_failed_prob_pred")

# (track, mu, time_budget_s, trial, seed) — extracted from collision results
# Excludes mu=0.1 and 1ms budget experiments
FAILED_EXPERIMENTS = [
    # ("Sepang", 0.6, 0.02, 2, 43),
    # ("Sepang", 0.5, 0.015, 2, 43),
    # ("Sepang", 0.5, 0.03, 2, 43),
    # ("Sepang", 0.6, 0.025, 3, 44),
    # ("Sepang", 0.5, 0.03, 3, 44),
    # ("Sepang", 0.6, 0.02, 5, 46),
    # ("Sepang", 0.5, 0.015, 5, 46),
    # ("Sepang", 0.5, 0.02, 6, 47),
    # ("Sepang", 0.5, 0.015, 7, 48),
    # ("Sepang", 0.6, 0.03, 8, 49),
    ("Sepang", 0.5, 0.02, 8, 49),
    # ("Sepang", 0.5, 0.01, 9, 50),
    # ("YasMarina", 1.0, 0.015, 9, 50),
]


def main():
    parser = argparse.ArgumentParser(
        description="Re-run failed (collision) prob-pred MCTS experiments",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    total = len(FAILED_EXPERIMENTS)
    log.info("Re-running %d failed experiments", total)

    results = []
    for idx, (track, mu, tb, trial, seed) in enumerate(FAILED_EXPERIMENTS, 1):
        log.info(
            "\n>>> Rerun %d/%d: %s  mu=%.4f  budget=%dms  trial=%d  seed=%d",
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


if __name__ == "__main__":
    main()
