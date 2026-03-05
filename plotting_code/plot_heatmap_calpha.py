"""Generate heatmaps from MCTS + Krasser SVGP (C_alpha) experiment results.

Merged from 3 result sources:
  1. nohup_yas_sepang_calpha.txt           (original mu=1.0..0.3, YasMarina+Sepang)
  2. nohup_yas_sepang_calpha_retry.txt     (retry of 17 crashed/truncated)
  3. Last retry of Sepang 0.3 lw=1.0       (the one remaining crash)
"""

from pathlib import Path as _Path
_DIR = _Path(__file__).resolve().parent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Parse results ────────────────────────────────────────────────────────────
# Format: Track  mu  lw  done  lap1  lap2   (or CRASHED / COLLISION)
# Later entries override earlier ones, so retries override original crashes.

# Original results from nohup_yas_sepang_calpha.txt (153 done + 6 collision)
# Avg lap time computed as steps * 0.04 / 2
raw_original = """
YasMarina     1.0   0.5  done  55.1  55.1
YasMarina     1.0   0.6  done  54.9  54.9
YasMarina     1.0   0.7  done  56.9  56.9
YasMarina     1.0   0.8  done  55.4  55.4
YasMarina     1.0   0.9  done  55.6  55.6
YasMarina     1.0   1.0  done  55.3  55.3
YasMarina     1.0   1.1  done  55.9  55.9
YasMarina     1.0   1.2  done  56.1  56.1
YasMarina     1.0   1.3  done  56.3  56.3
YasMarina     1.0   1.4  done  55.4  55.4
YasMarina     1.0   1.5  done  54.6  54.6
YasMarina     0.9   0.5  COLLISION
YasMarina     0.9   0.6  done  68.1  68.1
YasMarina     0.9   0.7  done  68.5  68.5
YasMarina     0.9   0.8  done  67.9  67.9
YasMarina     0.9   0.9  COLLISION
YasMarina     0.9   1.0  done  69.9  69.9
YasMarina     0.9   1.1  done  68.2  68.2
YasMarina     0.9   1.2  done  70.3  70.3
YasMarina     0.9   1.3  done  67.4  67.4
YasMarina     0.9   1.4  done  69.2  69.2
YasMarina     0.9   1.5  done  68.3  68.3
YasMarina     0.8   0.5  COLLISION
YasMarina     0.8   0.6  done  77.3  77.3
YasMarina     0.8   0.7  done  77.5  77.5
YasMarina     0.8   0.8  done  77.0  77.0
YasMarina     0.8   0.9  done  77.9  77.9
YasMarina     0.8   1.0  done  77.6  77.6
YasMarina     0.8   1.1  done  78.4  78.4
YasMarina     0.8   1.2  done  78.7  78.7
YasMarina     0.8   1.3  done  77.4  77.4
YasMarina     0.8   1.4  done  77.3  77.3
YasMarina     0.8   1.5  done  77.2  77.2
YasMarina     0.7   0.5  done  73.9  73.9
YasMarina     0.7   0.6  done  74.8  74.8
YasMarina     0.7   0.7  done  75.4  75.4
YasMarina     0.7   0.8  done  74.3  74.3
YasMarina     0.7   0.9  done  76.1  76.1
YasMarina     0.7   1.0  done  75.6  75.6
YasMarina     0.7   1.1  done  75.5  75.5
YasMarina     0.7   1.2  done  75.2  75.2
YasMarina     0.7   1.3  done  74.1  74.1
YasMarina     0.7   1.4  done  74.3  74.3
YasMarina     0.7   1.5  done  73.9  73.9
YasMarina     0.6   0.5  done  67.9  67.9
YasMarina     0.6   0.6  done  67.1  67.1
YasMarina     0.6   0.9  done  67.9  67.9
YasMarina     0.6   1.0  done  69.7  69.7
YasMarina     0.6   1.1  done  68.5  68.5
YasMarina     0.6   1.2  done  68.3  68.3
YasMarina     0.6   1.3  done  68.5  68.5
YasMarina     0.6   1.4  done  67.1  67.1
YasMarina     0.6   1.5  done  66.2  66.2
YasMarina     0.5   0.5  done  74.5  74.5
YasMarina     0.5   0.6  done  77.0  77.0
YasMarina     0.5   0.7  done  76.6  76.6
YasMarina     0.5   0.8  done  77.2  77.2
YasMarina     0.5   0.9  done  77.8  77.8
YasMarina     0.5   1.0  done  77.6  77.6
YasMarina     0.5   1.1  done  76.7  76.7
YasMarina     0.5   1.3  done  77.3  77.3
YasMarina     0.5   1.4  done  74.6  74.6
YasMarina     0.5   1.5  done  74.4  74.4
YasMarina     0.4   0.5  done  117.8  117.8
YasMarina     0.4   0.6  done  127.7  127.7
YasMarina     0.4   0.7  done  125.8  125.8
YasMarina     0.4   0.8  done  130.8  130.8
YasMarina     0.4   0.9  done  130.2  130.2
YasMarina     0.4   1.0  done  126.0  126.0
YasMarina     0.4   1.1  done  128.0  128.0
YasMarina     0.4   1.2  done  121.1  121.1
YasMarina     0.4   1.3  done  117.4  117.4
YasMarina     0.4   1.4  done  117.4  117.4
YasMarina     0.4   1.5  done  115.9  115.9
YasMarina     0.3   0.5  done  123.8  123.8
YasMarina     0.3   0.6  done  124.4  124.4
YasMarina     0.3   0.7  done  125.5  125.5
YasMarina     0.3   0.8  done  125.8  125.8
YasMarina     0.3   0.9  done  125.9  125.9
YasMarina     0.3   1.1  done  122.3  122.3
YasMarina     0.3   1.2  done  120.5  120.5
YasMarina     0.3   1.3  done  119.3  119.3
YasMarina     0.3   1.4  done  115.9  115.9
YasMarina     0.3   1.5  done  114.1  114.1
Sepang        1.0   0.5  COLLISION
Sepang        1.0   0.6  COLLISION
Sepang        1.0   0.7  done  51.4  51.4
Sepang        1.0   0.8  done  50.2  50.2
Sepang        1.0   0.9  done  49.3  49.3
Sepang        1.0   1.0  done  49.9  49.9
Sepang        1.0   1.1  done  51.5  51.5
Sepang        1.0   1.2  COLLISION
Sepang        1.0   1.3  done  49.8  49.8
Sepang        1.0   1.4  done  49.7  49.7
Sepang        1.0   1.5  done  49.4  49.4
Sepang        0.9   0.6  done  58.2  58.2
Sepang        0.9   0.7  done  58.5  58.5
Sepang        0.9   0.8  done  60.7  60.7
Sepang        0.9   0.9  done  59.1  59.1
Sepang        0.9   1.0  done  60.2  60.2
Sepang        0.9   1.1  done  60.6  60.6
Sepang        0.9   1.2  done  62.2  62.2
Sepang        0.9   1.3  done  61.8  61.8
Sepang        0.9   1.4  done  61.2  61.2
Sepang        0.9   1.5  done  61.2  61.2
Sepang        0.8   0.5  done  68.8  68.8
Sepang        0.8   0.6  done  68.3  68.3
Sepang        0.8   0.7  done  69.5  69.5
Sepang        0.8   0.8  done  71.1  71.1
Sepang        0.8   0.9  done  70.6  70.6
Sepang        0.8   1.0  done  71.0  71.0
Sepang        0.8   1.1  done  71.5  71.5
Sepang        0.8   1.2  done  70.3  70.3
Sepang        0.8   1.3  done  72.4  72.4
Sepang        0.8   1.5  done  71.9  71.9
Sepang        0.7   0.6  done  65.7  65.7
Sepang        0.7   0.7  done  66.6  66.6
Sepang        0.7   0.8  done  67.0  67.0
Sepang        0.7   0.9  done  66.4  66.4
Sepang        0.7   1.0  done  67.0  67.0
Sepang        0.7   1.1  done  68.5  68.5
Sepang        0.7   1.2  done  67.5  67.5
Sepang        0.7   1.3  done  68.5  68.5
Sepang        0.7   1.4  done  67.2  67.2
Sepang        0.7   1.5  done  68.6  68.6
Sepang        0.6   0.5  done  58.1  58.1
Sepang        0.6   0.6  done  60.1  60.1
Sepang        0.6   0.7  done  59.6  59.6
Sepang        0.6   0.8  done  60.5  60.5
Sepang        0.6   0.9  done  60.2  60.2
Sepang        0.6   1.0  done  61.8  61.8
Sepang        0.6   1.1  done  60.8  60.8
Sepang        0.6   1.2  done  61.5  61.5
Sepang        0.6   1.3  done  62.1  62.1
Sepang        0.6   1.4  done  62.2  62.2
Sepang        0.6   1.5  done  61.8  61.8
Sepang        0.5   0.5  done  65.4  65.4
Sepang        0.5   0.6  done  66.0  66.0
Sepang        0.5   0.7  done  68.8  68.8
Sepang        0.5   0.8  done  69.8  69.8
Sepang        0.5   0.9  done  70.7  70.7
Sepang        0.5   1.0  done  72.0  72.0
Sepang        0.5   1.1  done  71.2  71.2
Sepang        0.5   1.2  done  70.8  70.8
Sepang        0.5   1.3  done  70.4  70.4
Sepang        0.5   1.4  done  70.7  70.7
Sepang        0.5   1.5  done  69.4  69.4
Sepang        0.4   0.5  done  130.2  130.2
Sepang        0.4   0.6  done  139.4  139.4
Sepang        0.4   0.7  done  145.4  145.4
Sepang        0.4   0.8  done  148.2  148.2
Sepang        0.4   0.9  done  154.5  154.5
Sepang        0.4   1.0  done  146.5  146.5
Sepang        0.4   1.2  done  142.8  142.8
Sepang        0.4   1.3  done  137.5  137.5
Sepang        0.4   1.4  done  131.0  131.0
Sepang        0.4   1.5  done  129.0  129.0
Sepang        0.3   0.5  done  136.8  136.8
Sepang        0.3   0.6  done  141.8  141.8
"""

# Retry results (8 crashes + 9 truncated/never-ran from original)
# Actual lap times from bracket notation in summary table
raw_retry = """
YasMarina     0.6   0.7  done  68.6  67.8
YasMarina     0.6   0.8  done  69.6  68.8
YasMarina     0.5   1.2  done  78.2  77.9
YasMarina     0.3   1.0  done  125.8  122.2
Sepang        0.9   0.5  done  57.3  56.0
Sepang        0.8   1.4  done  71.6  71.9
Sepang        0.7   0.5  done  66.4  63.7
Sepang        0.4   1.1  done  148.2  149.2
Sepang        0.3   0.7  done  143.5  141.5
Sepang        0.3   0.8  done  141.9  142.4
Sepang        0.3   0.9  done  140.7  141.1
Sepang        0.3   1.0  CRASHED
Sepang        0.3   1.1  done  138.7  136.8
Sepang        0.3   1.2  done  137.2  137.2
Sepang        0.3   1.3  done  133.8  134.1
Sepang        0.3   1.4  done  132.3  130.1
Sepang        0.3   1.5  done  130.2  126.7
"""

# Last retry: Sepang 0.3 lw=1.0 (resolved the remaining crash)
# 7007 steps, avg lap = 7007 * 0.04 / 2 = 140.1s
raw_last_retry = """
Sepang        0.3   1.0  done  140.1  140.1
"""

# ── Structured parsing ───────────────────────────────────────────────────────

tracks_order = ["YasMarina", "Sepang"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
lws_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# result[track][(mu, lw)] = {"status": ..., "avg_lap": ...}
results = {t: {} for t in tracks_order}


def parse_block(raw):
    """Parse a raw text block; later calls override earlier entries."""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        track = parts[0]
        mu = float(parts[1])
        lw = float(parts[2])

        if track not in results:
            results[track] = {}

        if "CRASHED" in line:
            results[track][(mu, lw)] = {"status": "crashed", "avg_lap": None}
        elif "COLLISION" in line:
            results[track][(mu, lw)] = {"status": "collision", "avg_lap": None}
        else:
            lap_times = [float(parts[-2]), float(parts[-1])]
            avg_lap = np.mean(lap_times)
            results[track][(mu, lw)] = {"status": "done", "avg_lap": avg_lap}


# Parse in order: original → retry → last retry
# Later entries override earlier ones (crashes get replaced by retry results)
parse_block(raw_original)
parse_block(raw_retry)
parse_block(raw_last_retry)


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

    # Red bounding box around mu=0.4 and mu=0.3 rows (OOD region)
    ood_row_start = mus_order.index(0.4)
    ax.add_patch(plt.Rectangle(
        (-0.5, ood_row_start - 0.5), n_lw, 2,
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
    "MCTS + Krasser SVGP (C_alpha, ARD kernel): Avg Lap Time vs Real Track Width & Friction\n"
    "(lane_width=0.5, heading backward check enabled, n_simulations=2000, 2 laps)",
    fontsize=13, fontweight="bold", y=1.06)

plt.savefig(_DIR / "heatmap_calpha_results.png", dpi=150, bbox_inches="tight")
plt.savefig(_DIR / "heatmap_calpha_results.pdf", bbox_inches="tight")
print(f"Saved {_DIR / 'heatmap_calpha_results.png'} and {_DIR / 'heatmap_calpha_results.pdf'}")
