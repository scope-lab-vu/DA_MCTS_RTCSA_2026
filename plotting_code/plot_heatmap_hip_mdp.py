"""Generate heatmaps from MCTS + Ekin + HiP-MDP SVGP experiment results."""

from pathlib import Path as _Path
_DIR = _Path(__file__).resolve().parent
_REPO = _DIR.parent

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Parse the results table from the nohup log ──────────────────────────────

LOG_PATH = _REPO / "run_Nohup_results" / "hip_mdp_svgp" / "nohup_run_exp_svgp_ekin_3_mu_2_map_hip_mdp_svgp.txt"

lines = open(LOG_PATH).readlines()

# Find the results table (starts after the header line)
in_table = False
raw_results = []

for line in lines:
    line = line.rstrip()
    if line.startswith("Track") and "mu" in line and "LW" in line:
        in_table = True
        continue
    if in_table and line.startswith("==="):
        break
    if in_table and line.startswith("---"):
        continue
    if not in_table:
        continue

    # Parse result lines
    if "CRASHED" in line:
        # e.g. "YasMarina     1.0489   1.2     0      1     42  --- CRASHED (segfault) ---"
        parts = line.split()
        track, mu, lw = parts[0], float(parts[1]), float(parts[2])
        raw_results.append((track, mu, lw, "crashed", None))
    elif "YES" in line:
        # Collision: has "YES" in Collision column
        # e.g. "Sepang        1.0489   0.5     0      1     42     786 YES ..."
        parts = line.split()
        track, mu, lw = parts[0], float(parts[1]), float(parts[2])
        raw_results.append((track, mu, lw, "collision", None))
    elif "done" in line:
        # Completed: extract lap times from the end [49.6s  46.7s]
        parts = line.split()
        track, mu, lw = parts[0], float(parts[1]), float(parts[2])
        # Extract lap times from bracket notation
        bracket_match = re.search(r'\[(.+?)\]', line)
        if bracket_match:
            times_str = bracket_match.group(1)
            lap_times = [float(t.replace('s', '')) for t in times_str.split()]
            avg_lap = np.mean(lap_times)
        else:
            avg_lap = None
        raw_results.append((track, mu, lw, "done", avg_lap))

# ── Build structured data ────────────────────────────────────────────────────

tracks_order = ["YasMarina", "Sepang"]
mus_order = [1.0489, 0.8642, 0.4947]
lws_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

results = {t: {} for t in tracks_order}

for track, mu, lw, outcome, avg_lap in raw_results:
    if track not in results:
        continue
    results[track][(mu, lw)] = {"status": outcome, "avg_lap": avg_lap}

# ── Plot ─────────────────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

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
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n_lw - 0.5)
    ax.set_ylim(n_mu - 0.5, -0.5)
    ax.set_xticks(range(n_lw))
    ax.set_xticklabels([f"{lw:.1f}" for lw in lws_order], fontsize=8)
    ax.set_yticks(range(n_mu))
    ax.set_yticklabels([f"{mu:.4f}" for mu in mus_order], fontsize=9)
    ax.set_xlabel("Real Track Width", fontsize=11)
    ax.set_ylabel("Friction (mu)", fontsize=11)
    ax.set_title(track, fontsize=14, fontweight="bold")

# Legend
legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_BLUE, edgecolor="white", label="Seg Fault / Crash"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "MCTS + Ekin + HiP-MDP SVGP: Avg Lap Time vs Real Track Width & Friction\n"
    "(lane_width=0.5, heading backward check enabled, n_simulations=1000, 2 laps)",
    fontsize=13, fontweight="bold", y=1.10)

plt.savefig(_DIR / "heatmap_hip_mdp_results.png", dpi=150, bbox_inches="tight")
plt.savefig(_DIR / "heatmap_hip_mdp_results.pdf", bbox_inches="tight")
print(f"Saved {_DIR / 'heatmap_hip_mdp_results.png'} and {_DIR / 'heatmap_hip_mdp_results.pdf'}")
