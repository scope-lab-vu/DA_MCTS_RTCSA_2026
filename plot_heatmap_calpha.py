"""Generate heatmaps from MCTS + Krasser SVGP (C_alpha) experiment results."""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ── Parse nohup log ─────────────────────────────────────────────────────────

LOG_PATH = "nohup_yas_sepang_calpha.txt"
DT = 0.04

lines = open(LOG_PATH).readlines()

raw_results = []
current_lw = None
pending = False  # True if we saw a Dual boundaries line but no Finished yet

for line in lines:
    m = re.search(
        r'Dual boundaries: imaginary=[\d.]+ \(soft\), real=([\d.]+) \(hard/done\)', line)
    if m:
        if pending:
            # Previous experiment had no Finished → segfault
            raw_results.append((None, None, current_lw, 0, 'segfault', 0.0))
        current_lw = float(m.group(1))
        pending = True
        continue

    m = re.search(
        r'\[(\w+)\s+mu=([\d.]+)\]\s+Finished at step (\d+)\s+\((\w+)\)', line)
    if m:
        track, mu = m.group(1), float(m.group(2))
        steps, outcome = int(m.group(3)), m.group(4)
        raw_results.append((track, mu, current_lw, steps, outcome, steps * DT / 2))
        pending = False
        continue

# ── Build structured data: data[track][(mu, lw)] = {status, lap_time} ────────

tracks_order = ["YasMarina", "Sepang"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
lws_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

results = {t: {} for t in tracks_order}

# Walk through raw results and assign segfaults to their (track, mu) context.
# Segfaults have track=None; infer from the surrounding results.
# Strategy: keep track of the current (track, mu) being tested.
current_track = None
current_mu = None

for track, mu, lw, steps, outcome, lap_time in raw_results:
    if track is not None:
        current_track = track
        current_mu = mu

    t = track if track is not None else current_track
    m = mu if mu is not None else current_mu

    if t is None or t not in results:
        continue

    if outcome in ('done',):
        results[t][(m, lw)] = {"status": "done", "lap_time": lap_time}
    elif outcome == 'collision':
        results[t][(m, lw)] = {"status": "collision", "lap_time": None}
    elif outcome == 'segfault':
        results[t][(m, lw)] = {"status": "segfault", "lap_time": None}

# ── Plot ─────────────────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

for ax, track in zip(axes, tracks_order):
    n_mu = len(mus_order)
    n_lw = len(lws_order)

    data = np.full((n_mu, n_lw), np.nan)
    status = np.full((n_mu, n_lw), "", dtype=object)

    for i, mu in enumerate(mus_order):
        for j, lw in enumerate(lws_order):
            r = results[track].get((mu, lw))
            if r is None:
                status[i, j] = "missing"
            else:
                status[i, j] = r["status"]
                if r["lap_time"] is not None:
                    data[i, j] = r["lap_time"]

    for i in range(n_mu):
        for j in range(n_lw):
            s = status[i, j]
            if s == "segfault":
                color = COLOR_BLUE
                label = "SEG\nFAULT"
                text_color = "white"
            elif s == "collision":
                color = COLOR_RED
                label = "COL"
                text_color = "white"
            elif s == "done":
                color = COLOR_GREEN
                label = f"{data[i, j]:.1f}s" if not np.isnan(data[i, j]) else ""
                text_color = "black"
            else:
                color = COLOR_GRAY
                label = ""
                text_color = "black"

            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                         fill=True, facecolor=color, alpha=0.85,
                         edgecolor="white", lw=1.5))
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n_lw - 0.5)
    ax.set_ylim(n_mu - 0.5, -0.5)
    ax.set_xticks(range(n_lw))
    ax.set_xticklabels([f"{lw:.1f}" for lw in lws_order], fontsize=8)
    ax.set_yticks(range(n_mu))
    ax.set_yticklabels([f"{mu:.1f}" for mu in mus_order], fontsize=9)
    ax.set_xlabel("Real Track Width", fontsize=11)
    ax.set_ylabel("Friction (mu)", fontsize=11)
    ax.set_title(track, fontsize=14, fontweight="bold")

    # Red bounding box around mu=0.4 and mu=0.3 rows (OOD region)
    ood_row_start = mus_order.index(0.4)  # first OOD row
    ax.add_patch(plt.Rectangle(
        (-0.5, ood_row_start - 0.5), n_lw, 2,
        fill=False, edgecolor="red", linewidth=2.5, linestyle="-", zorder=10))

# Legend
legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_BLUE, edgecolor="white", label="Seg Fault"),
    Patch(facecolor=COLOR_GRAY, edgecolor="white", label="Missing / In Progress"),
    Patch(facecolor="none", edgecolor="red", linewidth=2.5, label="OOD (out of distribution)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "MCTS + Krasser SVGP (C_alpha, ARD kernel): Avg Lap Time vs Real Track Width & Friction\n"
    "(lane_width=0.5, heading backward check enabled, n_simulations=2000, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

plt.savefig("heatmap_calpha_results.png", dpi=150, bbox_inches="tight")
plt.savefig("heatmap_calpha_results.pdf", bbox_inches="tight")
print("Saved heatmap_calpha_results.png and heatmap_calpha_results.pdf")
