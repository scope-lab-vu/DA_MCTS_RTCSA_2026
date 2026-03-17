"""Generate heatmaps from Budget-Aware MCTS experiment results.

X-axis: MCTS time budget (ms)
Y-axis: Friction (mu)
Generates: 1 average heatmap + 3 per-trial heatmaps, per track.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_DIR = Path(__file__).resolve().parent

DATA_FILE = _DIR / "budget_mcts_1st_res.txt"

tracks_order = ["YasMarina", "Sepang"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
budgets_order = ["1ms", "3ms", "9ms", "12ms", "15ms", "18ms", "21ms", "24ms", "27ms", "30ms"]
trials_list = [1, 2, 3]


# ── Parse ────────────────────────────────────────────────────────────────────

# result[track][(mu, budget, trial)] = {"status": ..., "avg_lap": ...}
results = {t: {} for t in tracks_order}

with open(DATA_FILE) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 10 or parts[0] not in ("YasMarina", "Sepang"):
            continue
        track = parts[0]
        mu = float(parts[1])
        budget = parts[2]
        trial = int(parts[3])
        reason = parts[9]

        if reason == "collision":
            results[track][(mu, budget, trial)] = {"status": "collision", "avg_lap": None, "lap_times": []}
        elif reason == "done":
            lap_times = [float(t) for t in re.findall(r"([\d.]+)s", " ".join(parts[12:]))]
            avg_lap = np.mean(lap_times) if lap_times else None
            results[track][(mu, budget, trial)] = {"status": "done", "avg_lap": avg_lap, "lap_times": lap_times}
        else:
            results[track][(mu, budget, trial)] = {"status": "crashed", "avg_lap": None, "lap_times": []}


# ── Plotting helpers ─────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_GRAY, edgecolor="white", label="Missing"),
    Patch(facecolor="none", edgecolor="red", linewidth=2.5, label="OOD (out of distribution)"),
]


def draw_heatmap(ax, track, data_grid, status_grid):
    n_mu = len(mus_order)
    n_b = len(budgets_order)

    for i in range(n_mu):
        for j in range(n_b):
            s = status_grid[i, j]
            if s == "crashed":
                color, label, tc = COLOR_BLUE, "SEG\nFAULT", "white"
            elif s == "collision":
                color, label, tc = COLOR_RED, "COL", "white"
            elif s == "done":
                v = data_grid[i, j]
                color = COLOR_GREEN
                label = f"{v:.1f}s" if not np.isnan(v) else ""
                tc = "black"
            else:
                color, label, tc = COLOR_GRAY, "", "black"

            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                         fill=True, facecolor=color, alpha=0.85,
                         edgecolor="white", lw=1.5))
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=tc)

    ax.set_xlim(-0.5, n_b - 0.5)
    ax.set_ylim(n_mu - 0.5, -0.5)
    ax.set_xticks(range(n_b))
    ax.set_xticklabels(budgets_order, fontsize=8)
    ax.set_yticks(range(n_mu))
    ax.set_yticklabels([f"{mu:.1f}" for mu in mus_order], fontsize=9)
    ax.set_xlabel("MCTS Time Budget", fontsize=11)
    ax.set_ylabel("Friction (mu)", fontsize=11)
    ax.set_title(track, fontsize=14, fontweight="bold")

    # OOD box around mu <= 0.4
    ood_row_start = mus_order.index(0.4)
    ood_rows = len(mus_order) - ood_row_start
    ax.add_patch(plt.Rectangle(
        (-0.5, ood_row_start - 0.5), n_b, ood_rows,
        fill=False, edgecolor="red", linewidth=2.5, linestyle="-", zorder=10))


# ── Average heatmap ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

for ax, track in zip(axes, tracks_order):
    n_mu = len(mus_order)
    n_b = len(budgets_order)
    data_grid = np.full((n_mu, n_b), np.nan)
    status_grid = np.full((n_mu, n_b), "", dtype=object)

    for i, mu in enumerate(mus_order):
        for j, budget in enumerate(budgets_order):
            lap_times = []
            any_collision = False
            any_done = False
            for trial in trials_list:
                r = results[track].get((mu, budget, trial))
                if r is None:
                    continue
                if r["status"] == "collision":
                    any_collision = True
                elif r["status"] == "done" and r["avg_lap"] is not None:
                    any_done = True
                    lap_times.append(r["avg_lap"])

            if lap_times:
                data_grid[i, j] = np.mean(lap_times)
                status_grid[i, j] = "done"
            elif any_collision:
                status_grid[i, j] = "collision"
            else:
                status_grid[i, j] = "missing"

    draw_heatmap(ax, track, data_grid, status_grid)

fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    "Budget-Aware MCTS: Avg Lap Time vs MCTS Budget & Friction (avg over 3 trials)\n"
    "(lane_width=0.5, real_track_width=1.5, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

_OUT = _DIR / "exp_heatmaps"
_OUT.mkdir(exist_ok=True)

plt.savefig(_OUT / "heatmap_budget_mcts_avg.png", dpi=150, bbox_inches="tight")
plt.savefig(_OUT / "heatmap_budget_mcts_avg.pdf", bbox_inches="tight")
print(f"Saved {_OUT / 'heatmap_budget_mcts_avg.png'}")
plt.close()


# ── Per-trial heatmaps ──────────────────────────────────────────────────────

for trial in trials_list:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, track in zip(axes, tracks_order):
        n_mu = len(mus_order)
        n_b = len(budgets_order)
        data_grid = np.full((n_mu, n_b), np.nan)
        status_grid = np.full((n_mu, n_b), "", dtype=object)

        for i, mu in enumerate(mus_order):
            for j, budget in enumerate(budgets_order):
                r = results[track].get((mu, budget, trial))
                if r is None:
                    status_grid[i, j] = "missing"
                elif r["status"] == "collision":
                    status_grid[i, j] = "collision"
                elif r["status"] == "done" and r["avg_lap"] is not None:
                    data_grid[i, j] = r["avg_lap"]
                    status_grid[i, j] = "done"
                else:
                    status_grid[i, j] = "missing"

        draw_heatmap(ax, track, data_grid, status_grid)

    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Budget-Aware MCTS: Avg Lap Time vs MCTS Budget & Friction (Trial {trial})\n"
        "(lane_width=0.5, real_track_width=1.5, 2 laps)",
        fontsize=13, fontweight="bold", y=1.06)

    plt.savefig(_OUT / f"heatmap_budget_mcts_trial{trial}.png", dpi=150, bbox_inches="tight")
    plt.savefig(_OUT / f"heatmap_budget_mcts_trial{trial}.pdf", bbox_inches="tight")
    print(f"Saved {_OUT / f'heatmap_budget_mcts_trial{trial}.png'}")
    plt.close()


# ── Fastest lap heatmap ──────────────────────────────────────────────────────

fastest_legend = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (fastest lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_GRAY, edgecolor="white", label="Missing"),
    Patch(facecolor="none", edgecolor="red", linewidth=2.5, label="OOD (out of distribution)"),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

for ax, track in zip(axes, tracks_order):
    n_mu = len(mus_order)
    n_b = len(budgets_order)
    data_grid = np.full((n_mu, n_b), np.nan)
    status_grid = np.full((n_mu, n_b), "", dtype=object)

    for i, mu in enumerate(mus_order):
        for j, budget in enumerate(budgets_order):
            all_laps = []
            any_collision = False
            for trial in trials_list:
                r = results[track].get((mu, budget, trial))
                if r is None:
                    continue
                if r["status"] == "collision":
                    any_collision = True
                elif r["status"] == "done":
                    all_laps.extend(r["lap_times"])

            if all_laps:
                data_grid[i, j] = min(all_laps)
                status_grid[i, j] = "done"
            elif any_collision:
                status_grid[i, j] = "collision"
            else:
                status_grid[i, j] = "missing"

    draw_heatmap(ax, track, data_grid, status_grid)

fig.legend(handles=fastest_legend, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    "Budget-Aware MCTS: Fastest Lap Time vs MCTS Budget & Friction (best across 3 trials)\n"
    "(lane_width=0.5, real_track_width=1.5, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

plt.savefig(_OUT / "heatmap_budget_mcts_fastest.png", dpi=150, bbox_inches="tight")
plt.savefig(_OUT / "heatmap_budget_mcts_fastest.pdf", bbox_inches="tight")
print(f"Saved {_OUT / 'heatmap_budget_mcts_fastest.png'}")
plt.close()
