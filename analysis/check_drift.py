"""Did the machine's speed change during the sweep?

    python analysis/check_drift.py
    STUDY_RESULTS=~/results_sweep1 python analysis/check_drift.py     # an archived sweep

Reads results/timing/*.csv (post-warm-up decisions only) and, if present,
results/calibration.json and results/machine_log.csv.

Reference cells are the ones whose speed reflects the machine, not the planner's clock:
  * fixed-work arm: plan time for N iterations (should be constant when the machine is constant)
  * anytime/da arms at 'none': iterations achieved within the time budget
Writes results/analysis/drift.png.
"""
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

PAT = re.compile(r"timing_(?P<arm>fixed|anytime|da)_(?P<tag>.+?)_(?P<track>[A-Za-z]+)"
                 r"_mu(?P<mu>[\d.]+)_tb(?P<tb>\d+)ms_t(?P<trial>\d+)_s(?P<seed>\d+)\.csv$")

rows = []
for f in glob.glob(str(su.TIMING_DIR / "timing_*.csv")):
    m = PAT.match(Path(f).name)
    if not m:
        continue
    t = pd.read_csv(f)
    t = t[(t["in_warmup"] == 0) & (t["n_iters"] >= 0)]
    if len(t) < 30:
        continue
    tag = m["tag"]
    timing_only = tag.endswith(su.TIMING_SUFFIX)
    rows.append({
        "t0": float(t["wall_time"].iloc[0]),
        "variant": "timing-only" if timing_only else "closed-loop",
        "arm": m["arm"],
        "load": tag[: -len(su.TIMING_SUFFIX)] if timing_only else tag,
        "seed": int(m["seed"]),
        "decisions": len(t),
        "plan_med_ms": float(t["plan_time_s"].median() * 1000.0),
        "iters_med": float(t["n_iters"].median()),
    })
if not rows:
    sys.exit(f"No usable timing files found in {su.TIMING_DIR}")

df = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
df["hours"] = (df["t0"] - df["t0"].min()) / 3600.0
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print("=" * 90)
print(f"Sweep span: {df['hours'].max():.1f} h between first and last episode start "
      f"({len(df)} episodes with >= 30 decisions)")
cal = su.RESULTS / "calibration.json"
if cal.exists():
    c = json.loads(cal.read_text())
    print(f"Calibration reference (idle, da): plan p50 = {c['plan_ms_p50_p95_p99'][0]:.1f} ms, "
          f"iterations p50 = {c['iters_p10_p50_p90'][1]:.0f}, N = {c['suggested_n_iters']}")
print("=" * 90)


def rank_corr(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank())) if len(a) > 2 else float("nan")


for variant in ["closed-loop", "timing-only"]:
    v = df[df["variant"] == variant]
    print(f"\n--- {variant} ---")
    fx = v[(v["arm"] == "fixed") & (v["load"] == "none")].sort_values("hours")
    if len(fx) >= 3:
        half = len(fx) // 2
        print("IDLE fixed-work cells (same N each time): plan-time median in ms, in time order")
        print("  " + "  ".join(f"s{r.seed}:{r.plan_med_ms:.1f}" for r in fx.itertuples()))
        print(f"  first half {fx['plan_med_ms'].iloc[:half].mean():.1f} ms -> "
              f"second half {fx['plan_med_ms'].iloc[half:].mean():.1f} ms;  "
              f"rank correlation with elapsed time: {rank_corr(fx['hours'], fx['plan_med_ms']):+.2f}")
    for arm in ["anytime", "da"]:
        a = v[(v["arm"] == arm) & (v["load"] == "none")].sort_values("hours")
        if len(a) >= 3:
            half = len(a) // 2
            print(f"IDLE {arm} cells: iterations achieved within the budget (median), in time order")
            print("  " + "  ".join(f"s{r.seed}:{r.iters_med:.0f}" for r in a.itertuples()))
            print(f"  first half {a['iters_med'].iloc[:half].mean():.0f} -> second half "
                  f"{a['iters_med'].iloc[half:].mean():.0f};  rank correlation with elapsed time: "
                  f"{rank_corr(a['hours'], a['iters_med']):+.2f}")

# position effect: fixed-arm plan time per load, per seed (timing-only: full-length episodes)
v = df[(df["variant"] == "timing-only") & (df["arm"] == "fixed")]
if not v.empty:
    pv = v.pivot_table(index="seed", columns="load", values="plan_med_ms")
    pv = pv[[c for c in su.LOAD_ORDER if c in pv.columns]]
    print("\nFixed arm, timing-only: median plan time (ms) per seed and load")
    print("  Expect: none < bursty < light(about 2x none) < heavy(about 4x none).")
    print(pv.round(1).to_string())
    if {"none", "bursty"} <= set(pv.columns):
        n_bad = int((pv["none"] > pv["bursty"]).sum())
        print(f"  Seeds where the IDLE cell is slower than the partly loaded 'bursty' cell: "
              f"{n_bad} of {len(pv)}  (should be about 0 if the machine were steady)")

# machine log (written by the orchestrator in newer versions)
ml = su.RESULTS / "machine_log.csv"
if ml.exists():
    m = pd.read_csv(ml)
    print(f"\nmachine_log.csv: {len(m)} snapshots; core freq MHz min/median/max = "
          f"{m['freq_mhz'].min():.0f}/{m['freq_mhz'].median():.0f}/{m['freq_mhz'].max():.0f}; "
          f"temp C min/median/max = {m['temp_c'].min():.0f}/{m['temp_c'].median():.0f}/{m['temp_c'].max():.0f}")
    for col in ["governor", "ac_online", "no_turbo"]:
        if col in m:
            print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in m[col].fillna("?").value_counts().items()))
    if "psi_some_us" in m:
        key = ["arm", "load", "seed", "variant"]
        mb = m[m["event"] == "before"].drop_duplicates(key, keep="last").set_index(key)
        ma = m[m["event"] == "after"].drop_duplicates(key, keep="last").set_index(key)
        j = mb.join(ma, lsuffix="_b", rsuffix="_a", how="inner")
        j["dur_s"] = (j["time_a"] - j["time_b"]).clip(lower=1.0)
        j["stall_frac"] = (j["psi_some_us_a"] - j["psi_some_us_b"]) / 1e6 / j["dur_s"]
        print("  Memory stall as a fraction of episode time (system-wide 'some' stall; the window")
        print("  includes the one-off map build at the start of each episode):")
        print(j.reset_index().groupby("load")["stall_frac"].agg(["median", "max"]).round(3)
              .to_string())
        bad = j[j["stall_frac"] > 0.05]
        print(f"  Episodes spending more than 5% of their time in memory stall: {len(bad)} of {len(j)}")
        if len(bad):
            print(bad.sort_values("stall_frac", ascending=False)[["dur_s", "stall_frac"]].head(5)
                  .round(2).to_string())
        print(f"  Swap in use (MB) min/median/max: {m['swap_used_mb'].min():.0f}/"
              f"{m['swap_used_mb'].median():.0f}/{m['swap_used_mb'].max():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for variant, mk in [("closed-loop", "o"), ("timing-only", "s")]:
    v = df[(df["variant"] == variant) & (df["load"] == "none")]
    f = v[v["arm"] == "fixed"]
    axes[0].scatter(f["hours"], f["plan_med_ms"], marker=mk, label=variant, alpha=0.8)
    for arm, col in [("anytime", "tab:orange"), ("da", "tab:blue")]:
        a = v[v["arm"] == arm]
        axes[1].scatter(a["hours"], a["iters_med"], marker=mk, c=col, alpha=0.8,
                        label=f"{arm} ({variant})")
axes[0].set_title("Idle, fixed-work arm: plan time for the same N")
axes[0].set_xlabel("hours since sweep start"); axes[0].set_ylabel("median plan time (ms)")
axes[1].set_title("Idle, time-bounded arms: iterations within the budget")
axes[1].set_xlabel("hours since sweep start"); axes[1].set_ylabel("median iterations")
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
su.OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(su.OUT / "drift.png", dpi=150)
print(f"\nwrote {su.OUT / 'drift.png'}")
