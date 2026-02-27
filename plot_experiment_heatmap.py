#!/usr/bin/env python3
"""
Parse experiment log and generate a heatmap of crash/done results.

Usage:
    python plot_experiment_heatmap.py nohup_prior_gp_exp.txt
"""

import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "nohup_prior_gp_exp.txt"

# Parse all result lines from the log
# Pattern: results/mcts_pi_svgp_cs_{Track}_{mu_id}_lw{lw}/traj_...csv  (reason=collision|done)
pattern = re.compile(
    r"results/mcts_pi_svgp_cs_(\w+?)_(\d+)_lw([\d.]+)/.*\.csv\s+\(reason=(\w+)\)"
)

results = []
with open(LOG_FILE) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            track = m.group(1)
            mu_id = m.group(2)
            lw = float(m.group(3))
            reason = m.group(4)
            # Convert mu_id to float: "10" -> 1.0, "05" -> 0.5, "03" -> 0.3
            mu = int(mu_id) / 10.0
            results.append({
                "track": track,
                "mu": mu,
                "lw": lw,
                "reason": reason,
                "crashed": reason == "collision",
            })

print(f"Parsed {len(results)} experiment results from {LOG_FILE}")

# Get unique sorted axes
tracks = sorted(set(r["track"] for r in results))
mus = sorted(set(r["mu"] for r in results), reverse=True)
lws = sorted(set(r["lw"] for r in results))

# Row labels: "Track mu=X"
row_labels = []
for track in tracks:
    for mu in mus:
        row_labels.append(f"{track} mu={mu}")

# Build matrix: 0 = done, 1 = collision, -1 = missing
n_rows = len(row_labels)
n_cols = len(lws)
data = np.full((n_rows, n_cols), np.nan)

for r in results:
    row_idx = tracks.index(r["track"]) * len(mus) + mus.index(r["mu"])
    col_idx = lws.index(r["lw"])
    data[row_idx, col_idx] = 1.0 if r["crashed"] else 0.0

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Custom colormap: green=done, red=collision, gray=missing
cmap = mcolors.ListedColormap(["#4CAF50", "#F44336"])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot with nan shown as light gray
ax.set_facecolor("#E0E0E0")
im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

# Tick labels
ax.set_xticks(range(n_cols))
ax.set_xticklabels([f"{lw:.1f}" for lw in lws], fontsize=9)
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)

ax.set_xlabel("Lane Width", fontsize=12)
ax.set_title("MCTS + SVGP Action Prior: Crash Heatmap", fontsize=14, fontweight="bold")

# Add text annotations
for i in range(n_rows):
    for j in range(n_cols):
        if np.isnan(data[i, j]):
            ax.text(j, i, "-", ha="center", va="center", color="#999", fontsize=10)
        elif data[i, j] == 0:
            ax.text(j, i, "OK", ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")
        else:
            ax.text(j, i, "X", ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Horizontal separators between tracks
for i in range(1, len(tracks)):
    y = i * len(mus) - 0.5
    ax.axhline(y=y, color="black", linewidth=2.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="Completed (OK)"),
    Patch(facecolor="#F44336", label="Collision (X)"),
    Patch(facecolor="#E0E0E0", label="Not run (-)"),
]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1),
          fontsize=10)

# Summary stats
n_total = sum(1 for r in results)
n_crash = sum(1 for r in results if r["crashed"])
n_ok = n_total - n_crash
fig.text(0.5, 0.01,
         f"Total: {n_total} runs  |  OK: {n_ok} ({n_ok/n_total*100:.0f}%)  |  "
         f"Crashed: {n_crash} ({n_crash/n_total*100:.0f}%)",
         ha="center", fontsize=11, style="italic")

plt.tight_layout(rect=[0, 0.03, 0.88, 1.0])
out_path = "experiment_heatmap.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved heatmap to {out_path}")
plt.show()
