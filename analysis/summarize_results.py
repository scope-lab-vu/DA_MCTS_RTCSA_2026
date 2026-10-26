"""Summarise the contention study.

    python analysis/summarize_results.py

Reads   results/contention_summary.csv
Writes  results/analysis/condition_table.csv      (one row per arm x load)
        results/analysis/paired_differences.csv   (seed-paired differences with 95% CI)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--timing-only", action="store_true",
                help="analyse the timing-only variant (late-command model off)")
args = ap.parse_args()
OUTD = su.out_dir(args.timing_only)
OUTD.mkdir(parents=True, exist_ok=True)
if args.timing_only:
    print("*** TIMING-ONLY VARIANT: late-command model off. Use for deadline/runtime statistics; "
          "collision and completion here do NOT include the effect of late commands. ***\n")
df = su.load_summary(timing_only=args.timing_only)
table = su.build_condition_table(df)
table.to_csv(OUTD / "condition_table.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
fmt = lambda x: f"{x:.3f}"  # noqa: E731

print("=" * 100)
print("PER-CONDITION SUMMARY  (episodes = seeds; CI = 95%)")
print("=" * 100)
cols = ["arm", "load", "episodes", "completion_rate", "collision_rate", "progress_mean",
        "hit_mean", "plan_median_ms", "plan_p99_ms", "plan_max_ms_worst", "severe_miss_rate",
        "mean_iters", "mean_lap_s", "steps_before_crash", "steer_rate_change"]
print(table[cols].to_string(index=False, float_format=fmt))
print("\nWilson 95% intervals for collision and completion rates are in condition_table.csv.")
print("Lap times use completed episodes only; read them next to the completion rate.")

metrics = ["deadline_hit_rate", "progress_pct", "completed", "collision", "plan_p99_ms",
           "steer_rate_change"]
d = df.copy()
d["completed"] = d["completed"].astype(float)
d["collision"] = d["collision"].astype(float)
pairs = [("da", "fixed"), ("da", "anytime"), ("anytime", "fixed")]
diffs = pd.concat([su.paired_differences(d, a, b, metrics) for a, b in pairs], ignore_index=True)
diffs.to_csv(OUTD / "paired_differences.csv", index=False)


def show(title, comparison, metric_list):
    sub = diffs[(diffs["comparison"] == comparison) & (diffs["metric"].isin(metric_list))]
    print("\n" + title)
    if sub.empty:
        print("   (no seed-paired data yet)")
        return
    print(sub[["load", "metric", "pairs", "mean_diff", "ci_lo", "ci_hi", "ci_excludes_0"]]
          .to_string(index=False, float_format=fmt))


print("\n" + "=" * 100)
print("HYPOTHESIS CHECKS  (differences are DA-MCTS minus fixed-work, paired by seed)")
print("=" * 100)
show("H1  deadline hit rate: positive difference = DA-MCTS meets more deadlines",
     "da - fixed", ["deadline_hit_rate"])
show("H2  progress / completion / collisions under contention",
     "da - fixed", ["progress_pct", "completed", "collision"])

h = diffs[(diffs["comparison"] == "da - fixed") & (diffs["metric"] == "deadline_hit_rate")]
h = h.set_index("load").reindex([x for x in su.LOAD_ORDER if x in set(h["load"])])
if len(h) >= 3:
    gaps = h["mean_diff"].values
    mono = bool(np.all(np.diff(gaps[:3]) >= 0))
    print("\nH3  hit-rate gap across none -> light -> heavy: "
          + "  ".join(f"{l}={g:+.3f}" for l, g in zip(h.index, gaps))
          + f"   non-decreasing over the first three levels: {mono}")
    print("    (a widening gap is suggestive only; check the intervals above)")

show("Separating the effects: DA-MCTS vs anytime with uniform weights (sampling effect only)",
     "da - anytime", ["deadline_hit_rate", "progress_pct", "completed", "collision"])
show("Anytime vs fixed (effect of stopping on time)",
     "anytime - fixed", ["deadline_hit_rate", "progress_pct", "completed", "collision"])

print(f"\nWrote {OUTD/'condition_table.csv'} and {OUTD/'paired_differences.csv'}")
print("Rule of thumb: a difference is only worth reporting when ci_excludes_0 is True "
      "AND pairs is large enough to trust (aim for 10).")
