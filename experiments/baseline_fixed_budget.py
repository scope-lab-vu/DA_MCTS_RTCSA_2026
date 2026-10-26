"""Run ONE episode of the fixed-work baseline (exactly N MCTS iterations per step, no clock).

    python experiments/baseline_fixed_budget.py --load heavy --seed 42 [--n-iters 900]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--load", default="none")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n-iters", type=int, default=None,
                help="default: from study_config.yaml or results/calibration.json")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
cfg = common.load_config()
sys.exit(common.run_episode(cfg, "fixed", a.load, a.seed,
                            a.n_iters or common.resolve_n_iters(cfg), a.dry_run))
