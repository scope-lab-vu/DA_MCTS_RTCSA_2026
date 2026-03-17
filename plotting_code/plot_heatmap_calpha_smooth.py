"""Generate heatmaps from MCTS + Krasser SVGP C_alpha_smooth experiment results.

Data sources:
  1. nohup_all_maps_10_mu_11_lw_smooth_model.txt        -> YasMarina (n_sim=2000)
  2. nohup_all_maps_10_mu_11_lw_smooth_model_ns_3000.txt -> Sepang    (n_sim=3000)
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent

DT_S = 0.04
N_LAPS = 2

tracks_order = ["YasMarina", "Sepang"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
lws_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# ── Parse log files ─────────────────────────────────────────────────────────

def parse_nohup(path):
    """Return list of (track, mu, lw, steps, status) from a nohup log."""
    results = []
    current = None
    with open(path) as f:
        for line in f:
            m = re.search(
                r"EXPERIMENT:\s+(\w+)\s+mu=([\d.]+)\s+trial=\d+\s+seed=\d+\s+lane_width=([\d.]+)",
                line,
            )
            if m:
                current = (m.group(1), float(m.group(2)), float(m.group(3)))
            m2 = re.search(r"Finished at step (\d+) \((\w+)\)", line)
            if m2 and current:
                results.append((*current, int(m2.group(1)), m2.group(2)))
                current = None
    return results


raw1 = parse_nohup(_ROOT / "nohup_all_maps_10_mu_11_lw_smooth_model.txt")
raw2 = parse_nohup(_ROOT / "nohup_all_maps_10_mu_11_lw_smooth_model_ns_3000.txt")

# result[track][(mu, lw)] = {"status": ..., "avg_lap": ...}
results = {t: {} for t in tracks_order}

for track, mu, lw, steps, status in raw1:
    if track != "YasMarina":
        continue
    if status == "done":
        avg_lap = steps * DT_S / N_LAPS
        results["YasMarina"][(mu, lw)] = {"status": "done", "avg_lap": avg_lap}
    elif status == "collision":
        results["YasMarina"][(mu, lw)] = {"status": "collision", "avg_lap": None}
    else:
        results["YasMarina"][(mu, lw)] = {"status": "crashed", "avg_lap": None}

for track, mu, lw, steps, status in raw2:
    if track != "Sepang":
        continue
    if status == "done":
        avg_lap = steps * DT_S / N_LAPS
        results["Sepang"][(mu, lw)] = {"status": "done", "avg_lap": avg_lap}
    elif status == "collision":
        results["Sepang"][(mu, lw)] = {"status": "collision", "avg_lap": None}
    else:
        results["Sepang"][(mu, lw)] = {"status": "crashed", "avg_lap": None}


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
                if r["avg_lap"] is not None:
                    data[i, j] = r["avg_lap"]

    for i in range(n_mu):
        for j in range(n_lw):
            s = status[i, j]
            if s == "crashed":
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
                    fontsize=8, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n_lw - 0.5)
    ax.set_ylim(n_mu - 0.5, -0.5)
    ax.set_xticks(range(n_lw))
    ax.set_xticklabels([f"{lw:.1f}" for lw in lws_order], fontsize=8)
    ax.set_yticks(range(n_mu))
    ax.set_yticklabels([f"{mu:.1f}" for mu in mus_order], fontsize=9)
    ax.set_xlabel("Real Track Width", fontsize=11)
    ax.set_ylabel("Friction (mu)", fontsize=11)
    ax.set_title(track, fontsize=14, fontweight="bold")

    # Red bounding box around OOD region (mu=0.4 and below)
    ood_row_start = mus_order.index(0.4)
    ood_rows = n_mu - ood_row_start
    ax.add_patch(plt.Rectangle(
        (-0.5, ood_row_start - 0.5), n_lw, ood_rows,
        fill=False, edgecolor="red", linewidth=2.5, linestyle="-", zorder=10))

# Legend
legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_BLUE, edgecolor="white", label="Seg Fault"),
    Patch(facecolor="none", edgecolor="red", linewidth=2.5, label="OOD (out of distribution)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "MCTS + Krasser SVGP (C_alpha_smooth, ARD kernel): Avg Lap Time vs Real Track Width & Friction\n"
    "(lane_width=0.5, heading backward check enabled, YasMarina n_sim=2000, Sepang n_sim=3000, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

_OUT = _DIR / "exp_heatmaps"
_OUT.mkdir(exist_ok=True)

plt.savefig(_OUT / "heatmap_calpha_smooth_results.png", dpi=150, bbox_inches="tight")
plt.savefig(_OUT / "heatmap_calpha_smooth_results.pdf", bbox_inches="tight")
print(f"Saved {_OUT / 'heatmap_calpha_smooth_results.png'}")
print(f"  and {_OUT / 'heatmap_calpha_smooth_results.pdf'}")
