"""Generate heatmaps from MCTS + Known Dynamics experiment results.

Parses log files to extract lap times and crash info.
Supports merging multiple log files (original + retry) — later files override.
Experiments sweep mu (friction) × lwi (imaginary lane width) with fixed lwr=1.75.
"""

import re
from pathlib import Path as _Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_DIR = _Path(__file__).resolve().parent

# ── Log files to parse (in order — later entries override earlier ones) ─────

LOG_FILES = [
    _DIR.parent / "run_Nohup_results" / "known_dyn_mcts_new_sample" / "nohup_new_sample_known_dyn_mcts.txt",
    _DIR.parent / "run_Nohup_results" / "known_dyn_mcts_new_sample" / "nohup_retry_known_mcts_new_sample.txt",
]

DT = 0.04
N_LAPS = 2

# Regex patterns
RUN_RE = re.compile(
    r">>> Run \d+/\d+:\s+(\S+)\s+mu=([\d.]+)\s+lwi=([\d.]+)\s+lwr=([\d.]+)"
)
FINISH_RE = re.compile(
    r"Finished at step (\d+) \(done\)"
)
CRASH_RE = re.compile(
    r"Subprocess died:\s+(.+)"
)


def parse_log(path, results):
    """Parse a single log file, updating the results dict in-place.

    Later calls override earlier entries (retry results replace crashes).
    """
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return

    current_track = None
    current_mu = None
    current_lwi = None

    with open(path) as f:
        for line in f:
            m = RUN_RE.search(line)
            if m:
                current_track = m.group(1)
                current_mu = float(m.group(2))
                current_lwi = float(m.group(3))
                continue

            m = CRASH_RE.search(line)
            if m and current_track is not None:
                crash_type = m.group(1).strip()
                if current_track not in results:
                    results[current_track] = {}
                results[current_track][(current_mu, current_lwi)] = {
                    "status": "crashed", "avg_lap": None, "crash_type": crash_type,
                }
                current_track = None
                continue

            m = FINISH_RE.search(line)
            if m and current_track is not None:
                steps = int(m.group(1))
                avg_lap = steps * DT / N_LAPS
                if current_track not in results:
                    results[current_track] = {}
                results[current_track][(current_mu, current_lwi)] = {
                    "status": "done", "avg_lap": avg_lap,
                }
                current_track = None
                continue


# result[track][(mu, lwi)] = {"status": ..., "avg_lap": ...}
results = {}
for log_file in LOG_FILES:
    parse_log(log_file, results)
    print(f"Parsed {log_file.name}")

# ── Print summary ───────────────────────────────────────────────────────────

tracks_order = ["Sepang", "YasMarina"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
lwis_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

for track in tracks_order:
    done = sum(1 for v in results.get(track, {}).values() if v["status"] == "done")
    crash = sum(1 for v in results.get(track, {}).values() if v["status"] == "crashed")
    total = len(mus_order) * len(lwis_order)
    missing = total - done - crash
    print(f"{track}: {done} done, {crash} crashed, {missing} missing (of {total})")

# ── Plot ────────────────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)

for ax, track in zip(axes, tracks_order):
    n_mu = len(mus_order)
    n_lw = len(lwis_order)

    data = np.full((n_mu, n_lw), np.nan)
    status = np.full((n_mu, n_lw), "", dtype=object)

    for i, mu in enumerate(mus_order):
        for j, lwi in enumerate(lwis_order):
            r = results.get(track, {}).get((mu, lwi))
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
    ax.set_xticklabels([f"{lw:.1f}" for lw in lwis_order], fontsize=8)
    ax.set_yticks(range(n_mu))
    ax.set_yticklabels([f"{mu:.1f}" for mu in mus_order], fontsize=9)
    ax.set_xlabel("Imaginary Lane Width", fontsize=11)
    ax.set_ylabel("Friction (mu)", fontsize=11)
    ax.set_title(track, fontsize=14, fontweight="bold")

    # Red bounding box around OOD rows (mu <= 0.4)
    ood_row_start = mus_order.index(0.4)
    n_ood_rows = len(mus_order) - ood_row_start
    ax.add_patch(plt.Rectangle(
        (-0.5, ood_row_start - 0.5), n_lw, n_ood_rows,
        fill=False, edgecolor="red", linewidth=2.5, linestyle="-", zorder=10))

# Legend
legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_BLUE, edgecolor="white", label="Crash (segfault / abort)"),
    Patch(facecolor=COLOR_GRAY, edgecolor="white", label="Missing / not run"),
    Patch(facecolor="none", edgecolor="red", linewidth=2.5, label="OOD (out of distribution)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "MCTS + Known Dynamics + SVGP Action Prior: Avg Lap Time vs Imaginary Lane Width & Friction\n"
    "(real_track_width=1.75, heading backward check, n_simulations=3000, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

plt.savefig(_DIR / "heatmap_known_dyn_results.png", dpi=150, bbox_inches="tight")
plt.savefig(_DIR / "heatmap_known_dyn_results.pdf", bbox_inches="tight")
print(f"Saved {_DIR / 'heatmap_known_dyn_results.png'} and {_DIR / 'heatmap_known_dyn_results.pdf'}")
