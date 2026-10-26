"""Generate the study's figures.

    python analysis/generate_plots.py

Reads   results/contention_summary.csv and results/timing/*.csv
Writes  results/analysis/deadline_hit_rate.png
        results/analysis/collision_rate.png
        results/analysis/lap_completion.png
        results/analysis/progress.png
        results/analysis/runtime_distribution.png
        results/analysis/tradeoff.png
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--timing-only", action="store_true",
                help="plot the timing-only variant (late-command model off)")
args = ap.parse_args()
OUTD = su.out_dir(args.timing_only)
OUTD.mkdir(parents=True, exist_ok=True)
SUF = su.TIMING_SUFFIX if args.timing_only else ""
df = su.load_summary(timing_only=args.timing_only)
table = su.build_condition_table(df)
loads = [l for l in su.LOAD_ORDER if l in set(table["load"])]
arms = [a for a in su.ARM_ORDER if a in set(table["arm"])]


def grouped_bars(value, lo, hi, ylabel, title, fname, ylim=None, pct=False):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        xs, vs, ls, hs = [], [], [], []
        for j, load in enumerate(loads):
            r = table[(table["arm"] == arm) & (table["load"] == load)]
            if r.empty:
                continue
            r = r.iloc[0]
            xs.append(j + (i - (len(arms) - 1) / 2) * w)
            vs.append(r[value]); ls.append(r[lo]); hs.append(r[hi])
        vs, ls, hs = np.array(vs, float), np.array(ls, float), np.array(hs, float)
        err = np.array([np.clip(vs - ls, 0, None), np.clip(hs - vs, 0, None)])
        scale = 100.0 if pct else 1.0
        ax.bar(xs, vs * scale, width=w * 0.92, yerr=err * scale, capsize=3,
               color=su.ARM_COLOR[arm], label=su.ARM_LABEL[arm])
    ax.set_xticks(range(len(loads)))
    ax.set_xticklabels(loads)
    ax.set_xlabel("CPU contention condition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(OUTD / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTD / fname)


grouped_bars("hit_mean", "hit_lo", "hit_hi", "Deadline hit rate (%)",
             "Planning calls finishing within the 40 ms control period (95% bootstrap CI)",
             "deadline_hit_rate.png", ylim=(0, 105), pct=True)
grouped_bars("collision_rate", "collision_lo", "collision_hi", "Collision rate (%)",
             "Episodes ending in a collision (95% Wilson CI)",
             "collision_rate.png", ylim=(0, 105), pct=True)
grouped_bars("completion_rate", "completion_lo", "completion_hi", "Lap completion rate (%)",
             "Episodes completing all laps without a crash (95% Wilson CI)",
             "lap_completion.png", ylim=(0, 105), pct=True)

# progress: mean +/- sd
fig, ax = plt.subplots(figsize=(8, 4.8))
w = 0.8 / max(len(arms), 1)
for i, arm in enumerate(arms):
    xs, vs, sd = [], [], []
    for j, load in enumerate(loads):
        r = table[(table["arm"] == arm) & (table["load"] == load)]
        if r.empty:
            continue
        xs.append(j + (i - (len(arms) - 1) / 2) * w)
        vs.append(r.iloc[0]["progress_mean"]); sd.append(r.iloc[0]["progress_sd"])
    ax.bar(xs, vs, width=w * 0.92, yerr=np.nan_to_num(sd), capsize=3,
           color=su.ARM_COLOR[arm], label=su.ARM_LABEL[arm])
ax.set_xticks(range(len(loads))); ax.set_xticklabels(loads)
ax.set_xlabel("CPU contention condition"); ax.set_ylabel("Track progress (%)")
ax.set_title("Mean progress completed (error bars: 1 standard deviation)")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
fig.tight_layout(); fig.savefig(OUTD / "progress.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", OUTD / "progress.png")

# runtime distribution from the raw timing logs (post warm-up)
fig, axes = plt.subplots(1, len(loads), figsize=(3.2 * len(loads), 4.6), sharey=True, squeeze=False)
period_ms = 40.0
for ax, load in zip(axes[0], loads):
    data, labels, colors = [], [], []
    for arm in arms:
        vals = []
        for f in glob.glob(str(su.TIMING_DIR / f"timing_{arm}_{load}{SUF}_*.csv")):
            t = pd.read_csv(f)
            t = t[(t["in_warmup"] == 0) & (t["n_iters"] >= 0)]
            vals.append(t["plan_time_s"].values * 1000.0)
        if vals:
            data.append(np.concatenate(vals)); labels.append(arm); colors.append(su.ARM_COLOR[arm])
    if data:
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(period_ms, color="k", ls="--", lw=1)
    ax.set_title(load); ax.grid(axis="y", alpha=0.3)
axes[0][0].set_ylabel("Planning time per step (ms)")
fig.suptitle("Planning-time distribution (dashed line = 40 ms deadline; outliers hidden)")
fig.tight_layout(); fig.savefig(OUTD / "runtime_distribution.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "runtime_distribution.png")

# trade-off: miss rate vs mean lap time (completed episodes only)
fig, ax = plt.subplots(figsize=(7, 5))
markers = {"none": "o", "light": "s", "heavy": "^", "bursty": "D"}
for _, r in table.iterrows():
    if not (np.isfinite(r["mean_lap_s"]) and np.isfinite(r["hit_mean"])):
        continue
    ax.scatter((1 - r["hit_mean"]) * 100, r["mean_lap_s"], c=su.ARM_COLOR[r["arm"]],
               marker=markers.get(r["load"], "x"), s=80,
               label=f"{r['arm']} / {r['load']} ({r['completion_rate']*100:.0f}% done)")
ax.set_xlabel("Deadline miss rate (%)")
ax.set_ylabel("Mean lap time (s), completed episodes only")
ax.set_title("Reliability vs. quality trade-off (lower-left is better)")
ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUTD / "tradeoff.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "tradeoff.png")
