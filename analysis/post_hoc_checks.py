"""Two checks that need no rerun.   python analysis/post_hoc_checks.py

1. Decision-weighted vs time-weighted deadline hit rate (timing-only variant).
   A slower planner makes fewer decisions while the machine is overloaded, so counting
   decisions flatters it relative to counting wall-clock time (matters for 'bursty').
2. Stall spikes in IDLE cells (no competing load): any planning call over 100 ms there
   is not caused by the contention generator. Joined with the machine log (memory stall,
   swap) to see whether memory pressure coincides with the spikes.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

PAT = re.compile(r"timing_(?P<arm>fixed|anytime|da)_(?P<tag>.+?)_(?P<track>[A-Za-z]+)"
                 r"_mu(?P<mu>[\d.]+)_tb(?P<tb>\d+)ms_t(?P<trial>\d+)_s(?P<seed>\d+)\.csv$")


def episodes():
    for f in sorted(glob.glob(str(su.TIMING_DIR / "timing_*.csv"))):
        m = PAT.match(Path(f).name)
        if not m:
            continue
        t = pd.read_csv(f)
        t = t[(t["in_warmup"] == 0) & (t["n_iters"] >= 0)]
        if len(t) < 5:
            continue
        tag = m["tag"]
        timing_only = tag.endswith(su.TIMING_SUFFIX)
        yield {"arm": m["arm"], "seed": int(m["seed"]),
               "load": tag[: -len(su.TIMING_SUFFIX)] if timing_only else tag,
               "variant": "timing-only" if timing_only else "closed-loop", "t": t}


eps = list(episodes())
if not eps:
    sys.exit(f"No timing files in {su.TIMING_DIR}")

# ---- 1. decision-weighted vs time-weighted hit rate
rows = []
for e in eps:
    if e["variant"] != "timing-only":
        continue
    pt = e["t"]["plan_time_s"].values
    hit = e["t"]["met_deadline"].values == 1
    rows.append({"arm": e["arm"], "load": e["load"], "seed": e["seed"],
                 "decision_weighted": hit.mean(),
                 "time_weighted": pt[hit].sum() / pt.sum()})
if rows:
    d = pd.DataFrame(rows).groupby(["arm", "load"])[["decision_weighted", "time_weighted"]].mean()
    d = d.unstack("load")
    print("=" * 90)
    print("1. DEADLINE HIT RATE, counted per decision vs per unit of wall-clock time (timing-only)")
    print("=" * 90)
    for kind in ["decision_weighted", "time_weighted"]:
        tab = d[kind]
        tab = tab.reindex([a for a in su.ARM_ORDER if a in tab.index])
        tab = tab[[c for c in su.LOAD_ORDER if c in tab.columns]]
        print(f"\n{kind}:")
        print(tab.round(3).to_string())
    print("\nIf 'bursty' differs between arms by decision count but not by time, the gap is a "
          "counting artifact, not a real difference in reliability.")

# ---- 2. spikes in idle cells
spikes = []
for e in eps:
    if e["load"] != "none":
        continue
    t = e["t"]
    big = t[t["plan_time_s"] > 0.1]
    spikes.append({"variant": e["variant"], "arm": e["arm"], "seed": e["seed"],
                   "decisions": len(t), "spikes_over_100ms": len(big),
                   "max_ms": t["plan_time_s"].max() * 1000.0})
sp = pd.DataFrame(spikes)
print("\n" + "=" * 90)
print("2. STALL SPIKES IN IDLE CELLS (no competing load; a call over 100 ms is not contention)")
print("=" * 90)
print(f"Idle episodes: {len(sp)};  with at least one call over 100 ms: "
      f"{int((sp['spikes_over_100ms'] > 0).sum())}")

ml = su.RESULTS / "machine_log.csv"
if ml.exists():
    m = pd.read_csv(ml)
    key = ["arm", "load", "seed", "variant"]
    b = m[m["event"] == "before"].drop_duplicates(key, keep="last").set_index(key)
    a = m[m["event"] == "after"].drop_duplicates(key, keep="last").set_index(key)
    j = b.join(a, lsuffix="_b", rsuffix="_a", how="inner").reset_index()
    j["dur_s"] = (j["time_a"] - j["time_b"]).clip(lower=1.0)
    j["stall_frac"] = (j["psi_some_us_a"] - j["psi_some_us_b"]) / 1e6 / j["dur_s"]
    j = j[j["load"] == "none"][["arm", "seed", "variant", "stall_frac", "swap_used_mb_a"]]
    sp = sp.merge(j, on=["arm", "seed", "variant"], how="left")
    sp = sp.rename(columns={"swap_used_mb_a": "swap_mb"})
if len(sp[sp["spikes_over_100ms"] > 0]):
    print("\nIdle episodes WITH spikes:")
    print(sp[sp["spikes_over_100ms"] > 0].round(3).to_string(index=False))
if "stall_frac" in sp:
    w = sp[sp["spikes_over_100ms"] > 0]["stall_frac"].median()
    wo = sp[sp["spikes_over_100ms"] == 0]["stall_frac"].median()
    print(f"\nMedian memory-stall fraction of episode time, idle cells: with spikes {w:.3f}, "
          f"without spikes {wo:.3f}")
    print("Memory pressure is a plausible cause only if the 'with spikes' value is clearly higher.")
else:
    print("(no machine_log.csv with memory columns: cannot compare with memory stall)")
