"""Where did the sweep's time go?   python analysis/episode_costs.py

Uses results/machine_log.csv (before/after timestamps of every episode) and reports the
measured duration per cell, plus what a 10-seed run would cost at the same speed.
Cooldown pauses are NOT included in the durations (they happen after the 'after' row).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

ml = su.RESULTS / "machine_log.csv"
if not ml.exists():
    sys.exit(f"{ml} not found (it is written by experiments/cpu_contention_experiments.py run)")

m = pd.read_csv(ml)
key = ["arm", "load", "seed", "variant"]
b = m[m["event"] == "before"].drop_duplicates(key, keep="last").set_index(key)[["time"]]
a = m[m["event"] == "after"].drop_duplicates(key, keep="last").set_index(key)[["time"]]
j = b.join(a, lsuffix="_b", rsuffix="_a", how="inner")
j["minutes"] = (j["time_a"] - j["time_b"]) / 60.0
j = j.reset_index()

pd.set_option("display.width", 160)
g = (j.groupby(["variant", "arm", "load"])["minutes"]
      .agg(episodes="count", median_min="median", total_h=lambda s: s.sum() / 60.0))
print("Measured time per cell (minutes per episode, total hours per cell):")
print(g.round(2).to_string())

tot = j["minutes"].sum() / 60.0
print(f"\nTotal measured episode time: {tot:.1f} h for {len(j)} episodes "
      f"({j['minutes'].mean():.1f} min on average)")
print("\nShare of total time by variant, arm and load:")
for col in ["variant", "arm", "load"]:
    s = (j.groupby(col)["minutes"].sum() / j["minutes"].sum() * 100).round(0)
    print("  " + col + ": " + ", ".join(f"{k} {int(v)}%" for k, v in s.items()))

per_cell = j.groupby(["variant", "arm", "load"]).size()
seeds_done = int(per_cell.median())
if seeds_done > 0:
    print(f"\nAt this speed, 10 seeds per cell would take about {tot * 10 / seeds_done:.1f} h "
          f"(you have {seeds_done} seeds per cell), plus cooldowns.")
print("\nMost expensive cells (hours):")
print(g.sort_values("total_h", ascending=False).head(6)["total_h"].round(2).to_string())
