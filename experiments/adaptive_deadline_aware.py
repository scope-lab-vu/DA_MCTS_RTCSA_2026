"""Run ONE episode of a time-bounded (anytime) planner.

    python experiments/adaptive_deadline_aware.py --load heavy --seed 42 --variant da

  da       full DA-MCTS: deadline-aware sampling weights (the method under study)
  anytime  same time-bounded search, but the weights stay 1/3 each
           (isolates the effect of the deadline-aware sampling itself)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--load", default="none")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--variant", choices=["da", "anytime"], default="da")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
cfg = common.load_config()
sys.exit(common.run_episode(cfg, a.variant, a.load, a.seed, None, a.dry_run))
