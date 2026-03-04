"""Generate heatmaps from MCTS experiment results: avg lap time, colored by outcome.

Merged from 4 result files:
  1. nohup_heading_cost_new_real_track_width_all_maps.txt  (original mu=1.0..0.3)
  2. nohup_retry_crashed_mcts_known.txt                   (retry of 23 crashes)
  3. nohup_retry_crashed_mcts_known_remain.txt             (retry of 2 remaining)
  4. nohup_mcts_known_dyn_0.2_0.1_mu.txt                  (mu=0.2, 0.1)
"""

from pathlib import Path as _Path
_DIR = _Path(__file__).resolve().parent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Parse results ────────────────────────────────────────────────────────────
# Format: Track  mu  lw  ... lap1 lap2   (or CRASHED / COLLISION xxx)
# Later entries override earlier ones, so retries override original crashes.

raw_original = """
Sepang        1.0   0.5  CRASHED
Sepang        1.0   0.6  done  49.6 51.4
Sepang        1.0   0.7  done  50.4 51.5
Sepang        1.0   0.8  done  49.8 48.1
Sepang        1.0   0.9  done  51.2 49.5
Sepang        1.0   1.0  done  49.6 51.5
Sepang        1.0   1.1  done  51.4 47.5
Sepang        1.0   1.2  done  52.5 50.9
Sepang        1.0   1.3  done  51.2 50.3
Sepang        1.0   1.4  CRASHED
Sepang        1.0   1.5  CRASHED
Sepang        0.9   0.5  COLLISION 57.7
Sepang        0.9   0.6  done  62.3 60.3
Sepang        0.9   0.7  done  62.9 58.3
Sepang        0.9   0.8  done  59.0 59.5
Sepang        0.9   0.9  done  60.2 58.1
Sepang        0.9   1.0  done  59.4 60.6
Sepang        0.9   1.1  done  60.3 54.6
Sepang        0.9   1.2  done  56.4 57.8
Sepang        0.9   1.3  done  58.9 54.3
Sepang        0.9   1.4  done  59.3 57.4
Sepang        0.9   1.5  done  59.6 54.8
Sepang        0.8   0.5  done  67.4 63.2
Sepang        0.8   0.6  done  67.8 65.0
Sepang        0.8   0.7  done  71.8 70.6
Sepang        0.8   0.8  done  70.9 70.0
Sepang        0.8   0.9  done  69.7 68.2
Sepang        0.8   1.0  done  66.4 69.1
Sepang        0.8   1.1  done  68.1 65.3
Sepang        0.8   1.2  done  67.4 67.4
Sepang        0.8   1.3  done  66.2 66.4
Sepang        0.8   1.4  done  67.4 65.2
Sepang        0.8   1.5  done  69.3 66.8
Sepang        0.7   0.5  done  65.0 63.3
Sepang        0.7   0.6  done  66.6 64.0
Sepang        0.7   0.7  CRASHED
Sepang        0.7   0.8  done  66.6 64.3
Sepang        0.7   0.9  done  67.7 65.8
Sepang        0.7   1.0  done  66.8 62.9
Sepang        0.7   1.1  done  65.0 65.1
Sepang        0.7   1.2  done  65.0 63.4
Sepang        0.7   1.3  done  64.4 60.9
Sepang        0.7   1.4  done  63.3 62.9
Sepang        0.7   1.5  CRASHED
Sepang        0.6   0.5  done  59.0 57.7
Sepang        0.6   0.6  done  61.0 58.0
Sepang        0.6   0.7  done  58.3 58.0
Sepang        0.6   0.8  done  58.7 57.5
Sepang        0.6   0.9  done  58.7 57.2
Sepang        0.6   1.0  done  57.5 55.9
Sepang        0.6   1.1  done  58.0 56.4
Sepang        0.6   1.2  done  57.5 54.6
Sepang        0.6   1.3  done  55.0 55.9
Sepang        0.6   1.4  done  55.5 55.3
Sepang        0.6   1.5  done  56.0 54.9
Sepang        0.5   0.5  done  66.0 66.8
Sepang        0.5   0.6  done  71.0 65.7
Sepang        0.5   0.7  done  67.2 67.9
Sepang        0.5   0.8  done  66.9 65.7
Sepang        0.5   0.9  done  66.4 65.1
Sepang        0.5   1.0  done  66.3 63.6
Sepang        0.5   1.1  done  62.5 64.6
Sepang        0.5   1.2  done  63.4 62.0
Sepang        0.5   1.3  done  61.0 60.6
Sepang        0.5   1.4  done  61.2 59.3
Sepang        0.5   1.5  done  61.4 60.4
Sepang        0.4   0.5  done  161.6 161.7
Sepang        0.4   0.6  done  163.4 162.4
Sepang        0.4   0.7  done  154.2 148.2
Sepang        0.4   0.8  done  148.8 151.3
Sepang        0.4   0.9  done  139.2 135.2
Sepang        0.4   1.0  CRASHED
Sepang        0.4   1.1  done  117.3 117.2
Sepang        0.4   1.2  done  110.5 104.8
Sepang        0.4   1.3  done  98.2 93.0
Sepang        0.4   1.4  done  94.1 92.3
Sepang        0.4   1.5  done  93.7 90.0
Sepang        0.3   0.5  done  160.0 157.5
Sepang        0.3   0.6  done  156.4 156.0
Sepang        0.3   0.7  done  150.2 150.4
Sepang        0.3   0.8  done  141.0 139.7
Sepang        0.3   0.9  done  132.9 130.4
Sepang        0.3   1.0  done  122.6 123.7
Sepang        0.3   1.1  done  116.0 114.7
Sepang        0.3   1.2  CRASHED
Sepang        0.3   1.3  done  107.6 107.1
Sepang        0.3   1.4  done  105.7 106.2
Sepang        0.3   1.5  done  105.9 104.5
Shanghai      1.0   0.5  COLLISION 58.4
Shanghai      1.0   0.6  done  54.4 54.0
Shanghai      1.0   0.7  done  56.4 55.5
Shanghai      1.0   0.8  done  55.2 53.7
Shanghai      1.0   0.9  done  53.4 52.8
Shanghai      1.0   1.0  done  55.4 54.4
Shanghai      1.0   1.1  done  53.2 53.4
Shanghai      1.0   1.2  done  54.2 52.6
Shanghai      1.0   1.3  done  52.1 52.2
Shanghai      1.0   1.4  done  53.0 50.0
Shanghai      1.0   1.5  COLLISION 88.4
Shanghai      0.9   0.5  COLLISION 58.6
Shanghai      0.9   0.6  CRASHED
Shanghai      0.9   0.7  done  62.7 60.4
Shanghai      0.9   0.8  done  62.4 60.6
Shanghai      0.9   0.9  done  61.8 60.8
Shanghai      0.9   1.0  done  59.7 60.0
Shanghai      0.9   1.1  done  61.1 60.0
Shanghai      0.9   1.2  done  61.4 60.9
Shanghai      0.9   1.3  done  61.4 60.1
Shanghai      0.9   1.4  done  61.1 58.4
Shanghai      0.9   1.5  done  60.1 58.8
Shanghai      0.8   0.5  COLLISION 29.9
Shanghai      0.8   0.6  COLLISION 188.3
Shanghai      0.8   0.7  done  67.7 65.7
Shanghai      0.8   0.8  done  67.4 67.4
Shanghai      0.8   0.9  CRASHED
Shanghai      0.8   1.0  CRASHED
Shanghai      0.8   1.1  done  65.0 67.0
Shanghai      0.8   1.2  done  66.2 65.7
Shanghai      0.8   1.3  done  69.1 66.9
Shanghai      0.8   1.4  done  67.6 65.5
Shanghai      0.8   1.5  CRASHED
Shanghai      0.7   0.5  COLLISION 97.8
Shanghai      0.7   0.6  done  66.4 62.6
Shanghai      0.7   0.7  done  65.2 63.5
Shanghai      0.7   0.8  done  66.2 64.3
Shanghai      0.7   0.9  done  66.7 65.8
Shanghai      0.7   1.0  done  66.6 62.4
Shanghai      0.7   1.1  done  65.0 63.7
Shanghai      0.7   1.2  done  65.7 62.8
Shanghai      0.7   1.3  done  63.8 61.0
Shanghai      0.7   1.4  done  64.9 62.4
Shanghai      0.7   1.5  done  63.4 61.7
Shanghai      0.6   0.5  done  58.5 58.0
Shanghai      0.6   0.6  done  58.8 59.6
Shanghai      0.6   0.7  CRASHED
Shanghai      0.6   0.8  done  59.5 59.3
Shanghai      0.6   0.9  done  60.0 58.8
Shanghai      0.6   1.0  done  57.8 59.2
Shanghai      0.6   1.1  done  57.4 58.3
Shanghai      0.6   1.2  done  57.8 56.0
Shanghai      0.6   1.3  done  58.6 55.5
Shanghai      0.6   1.4  done  57.4 57.2
Shanghai      0.6   1.5  done  57.5 56.4
Shanghai      0.5   0.5  COLLISION 188.6
Shanghai      0.5   0.6  done  70.2 67.3
Shanghai      0.5   0.7  done  66.6 66.8
Shanghai      0.5   0.8  done  69.3 65.4
Shanghai      0.5   0.9  done  68.0 66.5
Shanghai      0.5   1.0  done  66.5 63.8
Shanghai      0.5   1.1  CRASHED
Shanghai      0.5   1.2  COLLISION 188.3
Shanghai      0.5   1.3  done  63.0 61.7
Shanghai      0.5   1.4  done  62.8 59.7
Shanghai      0.5   1.5  done  61.1 60.8
Shanghai      0.4   0.5  done  162.1 158.8
Shanghai      0.4   0.6  done  166.9 163.5
Shanghai      0.4   0.7  done  162.9 161.4
Shanghai      0.4   0.8  done  155.6 154.0
Shanghai      0.4   0.9  done  142.4 131.4
Shanghai      0.4   1.0  done  135.0 136.1
Shanghai      0.4   1.1  done  106.4 113.6
Shanghai      0.4   1.2  done  105.4 105.4
Shanghai      0.4   1.3  done  103.5 98.0
Shanghai      0.4   1.4  done  100.1 90.2
Shanghai      0.4   1.5  done  90.0 89.7
Shanghai      0.3   0.5  done  159.5 160.1
Shanghai      0.3   0.6  CRASHED
Shanghai      0.3   0.7  done  152.4 152.7
Shanghai      0.3   0.8  done  144.1 143.0
Shanghai      0.3   0.9  done  134.3 135.9
Shanghai      0.3   1.0  CRASHED
Shanghai      0.3   1.1  done  120.0 119.2
Shanghai      0.3   1.2  done  114.8 113.5
Shanghai      0.3   1.3  done  112.8 112.6
Shanghai      0.3   1.4  done  111.1 108.3
Shanghai      0.3   1.5  done  110.2 108.9
YasMarina     1.0   0.5  done  58.9 58.4
YasMarina     1.0   0.6  done  60.9 55.2
YasMarina     1.0   0.7  done  57.1 54.9
YasMarina     1.0   0.8  done  55.2 53.8
YasMarina     1.0   0.9  done  58.6 54.3
YasMarina     1.0   1.0  done  53.8 51.4
YasMarina     1.0   1.1  done  54.3 51.4
YasMarina     1.0   1.2  CRASHED
YasMarina     1.0   1.3  done  53.0 52.2
YasMarina     1.0   1.4  done  52.2 49.9
YasMarina     1.0   1.5  done  51.0 47.7
YasMarina     0.9   0.5  CRASHED
YasMarina     0.9   0.6  done  67.8 68.5
YasMarina     0.9   0.7  done  67.4 67.3
YasMarina     0.9   0.8  done  67.4 62.8
YasMarina     0.9   0.9  done  65.6 63.6
YasMarina     0.9   1.0  done  67.7 64.9
YasMarina     0.9   1.1  done  64.8 65.3
YasMarina     0.9   1.2  done  65.6 61.7
YasMarina     0.9   1.3  done  67.4 63.6
YasMarina     0.9   1.4  done  63.3 62.2
YasMarina     0.9   1.5  done  64.5 63.5
YasMarina     0.8   0.5  done  78.7 75.1
YasMarina     0.8   0.6  done  76.7 79.2
YasMarina     0.8   0.7  done  75.6 74.2
YasMarina     0.8   0.8  done  76.4 74.6
YasMarina     0.8   0.9  done  77.8 75.8
YasMarina     0.8   1.0  done  75.2 74.5
YasMarina     0.8   1.1  done  75.2 73.0
YasMarina     0.8   1.2  done  74.1 71.6
YasMarina     0.8   1.3  done  73.1 70.2
YasMarina     0.8   1.4  done  72.2 70.8
YasMarina     0.8   1.5  done  72.0 73.3
YasMarina     0.7   0.5  done  76.6 73.8
YasMarina     0.7   0.6  done  77.6 74.8
YasMarina     0.7   0.7  done  77.1 75.2
YasMarina     0.7   0.8  done  78.0 75.2
YasMarina     0.7   0.9  done  75.0 75.9
YasMarina     0.7   1.0  done  75.8 72.9
YasMarina     0.7   1.1  done  71.8 68.4
YasMarina     0.7   1.2  done  71.7 68.8
YasMarina     0.7   1.3  done  71.7 70.5
YasMarina     0.7   1.4  done  70.6 69.6
YasMarina     0.7   1.5  CRASHED
YasMarina     0.6   0.5  done  71.4 70.6
YasMarina     0.6   0.6  done  70.4 69.4
YasMarina     0.6   0.7  done  69.1 67.5
YasMarina     0.6   0.8  done  69.5 69.0
YasMarina     0.6   0.9  done  68.7 67.3
YasMarina     0.6   1.0  done  68.3 65.2
YasMarina     0.6   1.1  done  64.2 63.7
YasMarina     0.6   1.2  done  63.5 63.4
YasMarina     0.6   1.3  done  62.4 64.0
YasMarina     0.6   1.4  CRASHED
YasMarina     0.6   1.5  done  61.8 60.2
YasMarina     0.5   0.5  done  78.6 76.3
YasMarina     0.5   0.6  done  79.0 73.9
YasMarina     0.5   0.7  done  76.2 76.6
YasMarina     0.5   0.8  done  76.5 74.8
YasMarina     0.5   0.9  done  74.6 73.0
YasMarina     0.5   1.0  done  71.5 73.6
YasMarina     0.5   1.1  done  68.0 68.0
YasMarina     0.5   1.2  done  67.4 64.1
YasMarina     0.5   1.3  done  64.1 62.0
YasMarina     0.5   1.4  done  63.7 61.7
YasMarina     0.5   1.5  CRASHED
YasMarina     0.4   0.5  CRASHED
YasMarina     0.4   0.6  done  138.3 137.0
YasMarina     0.4   0.7  done  134.3 133.2
YasMarina     0.4   0.8  done  120.9 127.6
YasMarina     0.4   0.9  done  125.9 116.0
YasMarina     0.4   1.0  done  118.3 115.6
YasMarina     0.4   1.1  done  102.5 103.0
YasMarina     0.4   1.2  done  101.2 103.7
YasMarina     0.4   1.3  CRASHED
YasMarina     0.4   1.4  done  90.4 88.2
YasMarina     0.4   1.5  done  89.1 86.2
YasMarina     0.3   0.5  done  136.4 136.7
YasMarina     0.3   0.6  done  132.9 131.0
YasMarina     0.3   0.7  done  125.9 127.5
YasMarina     0.3   0.8  done  122.5 121.8
YasMarina     0.3   0.9  done  115.5 115.4
YasMarina     0.3   1.0  done  109.2 110.0
YasMarina     0.3   1.1  done  108.0 105.2
YasMarina     0.3   1.2  done  104.3 103.8
YasMarina     0.3   1.3  CRASHED
YasMarina     0.3   1.4  done  102.7 100.0
YasMarina     0.3   1.5  done  98.8 99.6
"""

# Retry results (override crashes from original)
raw_retry = """
Sepang        1.0   0.5  done  48.8 50.4
Sepang        1.0   1.4  done  48.2 46.7
Sepang        1.0   1.5  done  48.7 50.1
Sepang        0.7   1.5  done  63.7 62.6
Sepang        0.4   1.0  done  135.1 131.2
Sepang        0.3   1.2  done  110.0 111.1
Shanghai      0.9   0.6  done  61.2 59.5
Shanghai      0.8   0.9  done  69.2 67.9
Shanghai      0.8   1.0  done  70.6 67.4
Shanghai      0.8   1.5  done  67.1 65.2
Shanghai      0.6   0.7  done  59.3 58.4
Shanghai      0.5   1.1  done  65.6 65.6
Shanghai      0.3   0.6  done  158.7 158.8
Shanghai      0.3   1.0  done  125.9 126.6
YasMarina     1.0   1.2  done  54.4 50.2
YasMarina     0.9   0.5  done  66.9 66.6
YasMarina     0.7   1.5  done  68.4 68.4
YasMarina     0.6   1.4  done  60.3 60.3
YasMarina     0.5   1.5  done  66.3 63.0
YasMarina     0.4   0.5  done  139.8 139.4
YasMarina     0.3   1.3  done  101.2 100.2
"""

# Second retry (remaining 2 crashes from first retry)
raw_remain = """
Sepang        0.7   0.7  done  68.4 64.8
YasMarina     0.4   1.3  done  91.1 91.5
"""

# New mu=0.2 and mu=0.1 results (Sepang + YasMarina only, no Shanghai)
raw_new_mu = """
Sepang        0.2   0.5  done  134.0 134.1
Sepang        0.2   0.6  done  128.6 129.3
Sepang        0.2   0.7  done  119.0 118.8
Sepang        0.2   0.8  done  111.0 109.4
Sepang        0.2   0.9  done  103.4 105.4
Sepang        0.2   1.0  done  101.4 101.4
Sepang        0.2   1.1  done  101.2 98.9
Sepang        0.2   1.2  done  98.9 98.4
Sepang        0.2   1.3  done  99.1 98.2
Sepang        0.2   1.4  done  97.8 96.7
Sepang        0.2   1.5  done  98.7 98.4
Sepang        0.1   0.5  COLLISION 102.8
Sepang        0.1   0.6  CRASHED
Sepang        0.1   0.7  done  85.0 84.6
Sepang        0.1   0.8  done  86.4 82.7
Sepang        0.1   0.9  done  83.0 80.9
Sepang        0.1   1.0  done  81.4 81.1
Sepang        0.1   1.1  done  81.3 77.1
Sepang        0.1   1.2  done  79.9 78.3
Sepang        0.1   1.3  done  79.7 77.3
Sepang        0.1   1.4  done  81.0 77.6
Sepang        0.1   1.5  done  80.1 77.7
YasMarina     0.2   0.5  done  117.4 117.4
YasMarina     0.2   0.6  done  110.8 111.0
YasMarina     0.2   0.7  done  103.7 104.9
YasMarina     0.2   0.8  done  98.0 100.8
YasMarina     0.2   0.9  done  95.8 96.4
YasMarina     0.2   1.0  done  95.3 96.2
YasMarina     0.2   1.1  done  94.0 92.8
YasMarina     0.2   1.2  done  94.5 93.5
YasMarina     0.2   1.3  done  95.7 93.2
YasMarina     0.2   1.4  done  94.6 93.4
YasMarina     0.2   1.5  done  94.5 91.5
YasMarina     0.1   0.5  done  86.9 83.2
YasMarina     0.1   0.6  done  81.2 80.4
YasMarina     0.1   0.7  COLLISION 79.5
YasMarina     0.1   0.8  done  77.6 76.5
YasMarina     0.1   0.9  done  77.1 76.1
YasMarina     0.1   1.0  done  76.9 76.7
YasMarina     0.1   1.1  done  76.7 73.9
YasMarina     0.1   1.2  done  75.6 75.4
YasMarina     0.1   1.3  done  75.7 75.4
YasMarina     0.1   1.4  done  76.4 75.1
YasMarina     0.1   1.5  done  78.1 75.4
"""

# ── Structured parsing ───────────────────────────────────────────────────────

tracks_order = ["Sepang", "Shanghai", "YasMarina"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
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


# Parse in order: original → retry → remain → new mu values
# Later entries override earlier ones (crashes get replaced by retry results)
parse_block(raw_original)
parse_block(raw_retry)
parse_block(raw_remain)
parse_block(raw_new_mu)


# ── Plot ─────────────────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

fig, axes = plt.subplots(1, 3, figsize=(22, 9), constrained_layout=True)

for ax, track in zip(axes, tracks_order):
    n_mu = len(mus_order)
    n_lw = len(lws_order)

    # Build data matrix and status matrix
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

    # Draw all cells as colored rectangles based on outcome
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

# Legend
legend_elements = [
    Patch(facecolor=COLOR_GREEN, edgecolor="white", label="Completed (avg lap time shown)"),
    Patch(facecolor=COLOR_RED, edgecolor="white", label="Collision"),
    Patch(facecolor=COLOR_BLUE, edgecolor="white", label="Seg Fault"),
    Patch(facecolor=COLOR_GRAY, edgecolor="white", label="Not tested"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("MCTS + Known Dynamics + GP Action Prior: Avg Lap Time vs Real Track Width & Friction\n"
             "(lane_width=0.5, heading backward check enabled, n_simulations=1000, 2 laps)",
             fontsize=13, fontweight="bold", y=1.06)

plt.savefig(_DIR / "heatmap_results.png", dpi=150, bbox_inches="tight")
plt.savefig(_DIR / "heatmap_results.pdf", bbox_inches="tight")
print(f"Saved {_DIR / 'heatmap_results.png'} and {_DIR / 'heatmap_results.pdf'}")
