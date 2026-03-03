"""Generate heatmaps from MCTS experiment results: avg lap time, colored by outcome."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Parse results ────────────────────────────────────────────────────────────

raw = """
Sepang        1.0000   0.5     0      1     42  CRASHED
Sepang        1.0000   0.6     0      1     42    2526 No             2    199.9% done            10.11ms  49.6 51.4
Sepang        1.0000   0.7     0      1     42    2549 No             2    199.9% done            10.23ms  50.4 51.5
Sepang        1.0000   0.8     0      1     42    2447 No             2    199.9% done             9.97ms  49.8 48.1
Sepang        1.0000   0.9     0      1     42    2517 No             2    199.9% done             9.75ms  51.2 49.5
Sepang        1.0000   1.0     0      1     42    2527 No             2    199.9% done             9.84ms  49.6 51.5
Sepang        1.0000   1.1     0      1     42    2473 No             2    199.8% done             9.69ms  51.4 47.5
Sepang        1.0000   1.2     0      1     42    2585 No             2    199.9% done             9.65ms  52.5 50.9
Sepang        1.0000   1.3     0      1     42    2537 No             2    199.8% done             9.50ms  51.2 50.3
Sepang        1.0000   1.4     0      1     42  CRASHED
Sepang        1.0000   1.5     0      1     42  CRASHED
Sepang        0.9000   0.5     0      1     42  COLLISION 57.7
Sepang        0.9000   0.6     0      1     42    3065 No             2    199.9% done            10.10ms  62.3 60.3
Sepang        0.9000   0.7     0      1     42    3030 No             2    199.9% done             9.91ms  62.9 58.3
Sepang        0.9000   0.8     0      1     42    2961 No             2    199.9% done             9.91ms  59.0 59.5
Sepang        0.9000   0.9     0      1     42    2958 No             2    199.9% done             9.64ms  60.2 58.1
Sepang        0.9000   1.0     0      1     42    3000 No             2    199.8% done             9.60ms  59.4 60.6
Sepang        0.9000   1.1     0      1     42    2872 No             2    199.8% done             9.48ms  60.3 54.6
Sepang        0.9000   1.2     0      1     42    2856 No             2    199.9% done             9.41ms  56.4 57.8
Sepang        0.9000   1.3     0      1     42    2831 No             2    199.9% done             9.31ms  58.9 54.3
Sepang        0.9000   1.4     0      1     42    2917 No             2    199.8% done             9.29ms  59.3 57.4
Sepang        0.9000   1.5     0      1     42    2859 No             2    199.9% done             9.38ms  59.6 54.8
Sepang        0.8000   0.5     0      1     42    3265 No             2    199.9% done             9.92ms  67.4 63.2
Sepang        0.8000   0.6     0      1     42    3321 No             2    199.9% done             9.76ms  67.8 65.0
Sepang        0.8000   0.7     0      1     42    3558 No             2    199.9% done             9.78ms  71.8 70.6
Sepang        0.8000   0.8     0      1     42    3522 No             2    199.9% done             9.59ms  70.9 70.0
Sepang        0.8000   0.9     0      1     42    3447 No             2    199.9% done             9.56ms  69.7 68.2
Sepang        0.8000   1.0     0      1     42    3389 No             2    199.8% done             9.43ms  66.4 69.1
Sepang        0.8000   1.1     0      1     42    3334 No             2    199.9% done             9.29ms  68.1 65.3
Sepang        0.8000   1.2     0      1     42    3368 No             2    199.9% done             9.23ms  67.4 67.4
Sepang        0.8000   1.3     0      1     42    3313 No             2    199.9% done             9.14ms  66.2 66.4
Sepang        0.8000   1.4     0      1     42    3316 No             2    199.8% done             9.14ms  67.4 65.2
Sepang        0.8000   1.5     0      1     42    3403 No             2    199.9% done             9.04ms  69.3 66.8
Sepang        0.7000   0.5     0      1     42    3208 No             2    199.9% done             9.96ms  65.0 63.3
Sepang        0.7000   0.6     0      1     42    3264 No             2    199.9% done             9.70ms  66.6 64.0
Sepang        0.7000   0.7     0      1     42  CRASHED
Sepang        0.7000   0.8     0      1     42    3274 No             2    199.9% done             9.53ms  66.6 64.3
Sepang        0.7000   0.9     0      1     42    3338 No             2    199.9% done             9.40ms  67.7 65.8
Sepang        0.7000   1.0     0      1     42    3241 No             2    199.9% done             9.37ms  66.8 62.9
Sepang        0.7000   1.1     0      1     42    3251 No             2    199.8% done             9.20ms  65.0 65.1
Sepang        0.7000   1.2     0      1     42    3210 No             2    199.9% done             9.08ms  65.0 63.4
Sepang        0.7000   1.3     0      1     42    3133 No             2    199.8% done             9.13ms  64.4 60.9
Sepang        0.7000   1.4     0      1     42    3155 No             2    199.8% done             9.08ms  63.3 62.9
Sepang        0.7000   1.5     0      1     42  CRASHED
Sepang        0.6000   0.5     0      1     42    2916 No             2    199.9% done             9.90ms  59.0 57.7
Sepang        0.6000   0.6     0      1     42    2976 No             2    199.9% done             9.88ms  61.0 58.0
Sepang        0.6000   0.7     0      1     42    2907 No             2    199.9% done             9.61ms  58.3 58.0
Sepang        0.6000   0.8     0      1     42    2906 No             2    199.9% done             9.46ms  58.7 57.5
Sepang        0.6000   0.9     0      1     42    2898 No             2    199.9% done             9.28ms  58.7 57.2
Sepang        0.6000   1.0     0      1     42    2835 No             2    199.8% done             9.23ms  57.5 55.9
Sepang        0.6000   1.1     0      1     42    2859 No             2    199.9% done             8.97ms  58.0 56.4
Sepang        0.6000   1.2     0      1     42    2803 No             2    199.9% done             9.00ms  57.5 54.6
Sepang        0.6000   1.3     0      1     42    2773 No             2    199.8% done             8.90ms  55.0 55.9
Sepang        0.6000   1.4     0      1     42    2770 No             2    199.8% done             8.78ms  55.5 55.3
Sepang        0.6000   1.5     0      1     42    2771 No             2    199.9% done             8.78ms  56.0 54.9
Sepang        0.5000   0.5     0      1     42    3318 No             2    199.9% done            10.01ms  66.0 66.8
Sepang        0.5000   0.6     0      1     42    3416 No             2    199.9% done             9.58ms  71.0 65.7
Sepang        0.5000   0.7     0      1     42    3376 No             2    199.9% done             9.27ms  67.2 67.9
Sepang        0.5000   0.8     0      1     42    3315 No             2    199.9% done             9.23ms  66.9 65.7
Sepang        0.5000   0.9     0      1     42    3289 No             2    199.9% done             9.07ms  66.4 65.1
Sepang        0.5000   1.0     0      1     42    3248 No             2    199.9% done             8.91ms  66.3 63.6
Sepang        0.5000   1.1     0      1     42    3178 No             2    199.9% done             8.92ms  62.5 64.6
Sepang        0.5000   1.2     0      1     42    3134 No             2    199.9% done             8.94ms  63.4 62.0
Sepang        0.5000   1.3     0      1     42    3040 No             2    199.9% done             8.70ms  61.0 60.6
Sepang        0.5000   1.4     0      1     42    3012 No             2    199.9% done             8.82ms  61.2 59.3
Sepang        0.5000   1.5     0      1     42    3045 No             2    199.9% done             8.68ms  61.4 60.4
Sepang        0.4000   0.5     0      1     42    8084 No             2    199.9% done             8.78ms  161.6 161.7
Sepang        0.4000   0.6     0      1     42    8146 No             2    199.9% done             8.71ms  163.4 162.4
Sepang        0.4000   0.7     0      1     42    7559 No             2    199.9% done             8.53ms  154.2 148.2
Sepang        0.4000   0.8     0      1     42    7502 No             2    199.9% done             8.73ms  148.8 151.3
Sepang        0.4000   0.9     0      1     42    6862 No             2    199.9% done             8.48ms  139.2 135.2
Sepang        0.4000   1.0     0      1     42  CRASHED
Sepang        0.4000   1.1     0      1     42    5861 No             2    199.9% done             8.59ms  117.3 117.2
Sepang        0.4000   1.2     0      1     42    5382 No             2    199.9% done             8.64ms  110.5 104.8
Sepang        0.4000   1.3     0      1     42    4781 No             2    199.9% done             8.73ms  98.2 93.0
Sepang        0.4000   1.4     0      1     42    4661 No             2    199.9% done             8.72ms  94.1 92.3
Sepang        0.4000   1.5     0      1     42    4594 No             2    199.9% done             8.59ms  93.7 90.0
Sepang        0.3000   0.5     0      1     42    7937 No             2    199.9% done             8.60ms  160.0 157.5
Sepang        0.3000   0.6     0      1     42    7809 No             2    199.9% done             8.63ms  156.4 156.0
Sepang        0.3000   0.7     0      1     42    7513 No             2    199.9% done             9.69ms  150.2 150.4
Sepang        0.3000   0.8     0      1     42    7016 No             2    199.9% done             8.68ms  141.0 139.7
Sepang        0.3000   0.9     0      1     42    6582 No             2    199.9% done             8.68ms  132.9 130.4
Sepang        0.3000   1.0     0      1     42    6158 No             2    199.9% done             8.78ms  122.6 123.7
Sepang        0.3000   1.1     0      1     42    5768 No             2    199.9% done             8.85ms  116.0 114.7
Sepang        0.3000   1.2     0      1     42  CRASHED
Sepang        0.3000   1.3     0      1     42    5366 No             2    199.9% done             8.80ms  107.6 107.1
Sepang        0.3000   1.4     0      1     42    5298 No             2    199.9% done             8.83ms  105.7 106.2
Sepang        0.3000   1.5     0      1     42    5260 No             2    199.9% done             8.73ms  105.9 104.5
Shanghai      1.0000   0.5     0      1     42  COLLISION 58.4
Shanghai      1.0000   0.6     0      1     42    2710 No             2    199.9% done             9.75ms  54.4 54.0
Shanghai      1.0000   0.7     0      1     42    2798 No             2    199.9% done             9.69ms  56.4 55.5
Shanghai      1.0000   0.8     0      1     42    2724 No             2    199.9% done             9.58ms  55.2 53.7
Shanghai      1.0000   0.9     0      1     42    2656 No             2    199.9% done             9.54ms  53.4 52.8
Shanghai      1.0000   1.0     0      1     42    2746 No             2    199.8% done             9.64ms  55.4 54.4
Shanghai      1.0000   1.1     0      1     42    2664 No             2    199.9% done             9.44ms  53.2 53.4
Shanghai      1.0000   1.2     0      1     42    2668 No             2    199.9% done             9.49ms  54.2 52.6
Shanghai      1.0000   1.3     0      1     42    2608 No             2    199.9% done             9.42ms  52.1 52.2
Shanghai      1.0000   1.4     0      1     42    2575 No             2    199.9% done             9.29ms  53.0 50.0
Shanghai      1.0000   1.5     0      1     42  COLLISION 88.4
Shanghai      0.9000   0.5     0      1     42  COLLISION 58.6
Shanghai      0.9000   0.6     0      1     42  CRASHED
Shanghai      0.9000   0.7     0      1     42    3076 No             2    199.9% done             9.72ms  62.7 60.4
Shanghai      0.9000   0.8     0      1     42    3075 No             2    199.9% done             9.75ms  62.4 60.6
Shanghai      0.9000   0.9     0      1     42    3066 No             2    199.9% done             9.70ms  61.8 60.8
Shanghai      0.9000   1.0     0      1     42    2991 No             2    199.9% done             9.71ms  59.7 60.0
Shanghai      0.9000   1.1     0      1     42    3026 No             2    199.9% done             9.51ms  61.1 60.0
Shanghai      0.9000   1.2     0      1     42    3058 No             2    199.8% done             9.43ms  61.4 60.9
Shanghai      0.9000   1.3     0      1     42    3039 No             2    199.9% done             9.40ms  61.4 60.1
Shanghai      0.9000   1.4     0      1     42    2989 No             2    199.9% done             9.37ms  61.1 58.4
Shanghai      0.9000   1.5     0      1     42    2973 No             2    199.8% done             9.39ms  60.1 58.8
Shanghai      0.8000   0.5     0      1     42  COLLISION 29.9
Shanghai      0.8000   0.6     0      1     42  COLLISION 188.3
Shanghai      0.8000   0.7     0      1     42    3334 No             2    199.8% done             9.73ms  67.7 65.7
Shanghai      0.8000   0.8     0      1     42    3369 No             2    199.9% done             9.61ms  67.4 67.4
Shanghai      0.8000   0.9     0      1     42  CRASHED
Shanghai      0.8000   1.0     0      1     42  CRASHED
Shanghai      0.8000   1.1     0      1     42    3301 No             2    199.8% done             9.41ms  65.0 67.0
Shanghai      0.8000   1.2     0      1     42    3296 No             2    199.9% done             9.44ms  66.2 65.7
Shanghai      0.8000   1.3     0      1     42    3400 No             2    199.9% done             9.24ms  69.1 66.9
Shanghai      0.8000   1.4     0      1     42    3327 No             2    199.9% done             9.24ms  67.6 65.5
Shanghai      0.8000   1.5     0      1     42  CRASHED
Shanghai      0.7000   0.5     0      1     42  COLLISION 97.8
Shanghai      0.7000   0.6     0      1     42    3225 No             2    199.9% done             9.82ms  66.4 62.6
Shanghai      0.7000   0.7     0      1     42    3218 No             2    199.9% done             9.70ms  65.2 63.5
Shanghai      0.7000   0.8     0      1     42    3262 No             2    199.9% done             9.45ms  66.2 64.3
Shanghai      0.7000   0.9     0      1     42    3314 No             2    199.9% done             9.41ms  66.7 65.8
Shanghai      0.7000   1.0     0      1     42    3224 No             2    199.9% done             9.41ms  66.6 62.4
Shanghai      0.7000   1.1     0      1     42    3219 No             2    199.9% done             9.26ms  65.0 63.7
Shanghai      0.7000   1.2     0      1     42    3213 No             2    199.9% done             9.26ms  65.7 62.8
Shanghai      0.7000   1.3     0      1     42    3121 No             2    199.9% done             9.14ms  63.8 61.0
Shanghai      0.7000   1.4     0      1     42    3184 No             2    199.9% done             9.12ms  64.9 62.4
Shanghai      0.7000   1.5     0      1     42    3128 No             2    199.9% done             9.07ms  63.4 61.7
Shanghai      0.6000   0.5     0      1     42    2912 No             2    199.9% done             9.94ms  58.5 58.0
Shanghai      0.6000   0.6     0      1     42    2961 No             2    199.9% done             9.97ms  58.8 59.6
Shanghai      0.6000   0.7     0      1     42  CRASHED
Shanghai      0.6000   0.8     0      1     42    2971 No             2    199.9% done             9.70ms  59.5 59.3
Shanghai      0.6000   0.9     0      1     42    2972 No             2    199.8% done             9.49ms  60.0 58.8
Shanghai      0.6000   1.0     0      1     42    2923 No             2    199.9% done             9.36ms  57.8 59.2
Shanghai      0.6000   1.1     0      1     42    2892 No             2    199.9% done             9.21ms  57.4 58.3
Shanghai      0.6000   1.2     0      1     42    2845 No             2    199.9% done             9.09ms  57.8 56.0
Shanghai      0.6000   1.3     0      1     42    2852 No             2    199.9% done             9.03ms  58.6 55.5
Shanghai      0.6000   1.4     0      1     42    2865 No             2    199.9% done             9.06ms  57.4 57.2
Shanghai      0.6000   1.5     0      1     42    2847 No             2    199.9% done             8.95ms  57.5 56.4
Shanghai      0.5000   0.5     0      1     42  COLLISION 188.6
Shanghai      0.5000   0.6     0      1     42    3438 No             2    199.9% done             9.63ms  70.2 67.3
Shanghai      0.5000   0.7     0      1     42    3337 No             2    199.9% done             9.46ms  66.6 66.8
Shanghai      0.5000   0.8     0      1     42    3368 No             2    199.9% done             9.34ms  69.3 65.4
Shanghai      0.5000   0.9     0      1     42    3362 No             2    199.9% done             9.17ms  68.0 66.5
Shanghai      0.5000   1.0     0      1     42    3259 No             2    199.9% done             9.02ms  66.5 63.8
Shanghai      0.5000   1.1     0      1     42  CRASHED
Shanghai      0.5000   1.2     0      1     42  COLLISION 188.3
Shanghai      0.5000   1.3     0      1     42    3116 No             2    199.9% done             8.88ms  63.0 61.7
Shanghai      0.5000   1.4     0      1     42    3063 No             2    199.9% done             8.80ms  62.8 59.7
Shanghai      0.5000   1.5     0      1     42    3048 No             2    199.9% done             8.81ms  61.1 60.8
Shanghai      0.4000   0.5     0      1     42    8021 No             2    199.9% done             8.82ms  162.1 158.8
Shanghai      0.4000   0.6     0      1     42    8260 No             2    199.9% done             8.92ms  166.9 163.5
Shanghai      0.4000   0.7     0      1     42    8108 No             2    199.9% done             8.55ms  162.9 161.4
Shanghai      0.4000   0.8     0      1     42    7738 No             2    199.9% done             8.98ms  155.6 154.0
Shanghai      0.4000   0.9     0      1     42    6845 No             2    199.9% done             8.52ms  142.4 131.4
Shanghai      0.4000   1.0     0      1     42    6776 No             2    199.9% done             9.07ms  135.0 136.1
Shanghai      0.4000   1.1     0      1     42    5499 No             2    199.9% done             8.67ms  106.4 113.6
Shanghai      0.4000   1.2     0      1     42    5270 No             2    199.9% done             8.63ms  105.4 105.4
Shanghai      0.4000   1.3     0      1     42    5036 No             2    199.9% done             8.90ms  103.5 98.0
Shanghai      0.4000   1.4     0      1     42    4757 No             2    199.9% done             8.71ms  100.1 90.2
Shanghai      0.4000   1.5     0      1     42    4492 No             2    199.9% done             9.20ms  90.0 89.7
Shanghai      0.3000   0.5     0      1     42    7990 No             2    199.9% done             8.89ms  159.5 160.1
Shanghai      0.3000   0.6     0      1     42  CRASHED
Shanghai      0.3000   0.7     0      1     42    7627 No             2    199.9% done             8.95ms  152.4 152.7
Shanghai      0.3000   0.8     0      1     42    7176 No             2    199.9% done             8.52ms  144.1 143.0
Shanghai      0.3000   0.9     0      1     42    6754 No             2    199.9% done             8.81ms  134.3 135.9
Shanghai      0.3000   1.0     0      1     42  CRASHED
Shanghai      0.3000   1.1     0      1     42    5980 No             2    199.9% done             8.87ms  120.0 119.2
Shanghai      0.3000   1.2     0      1     42    5707 No             2    199.9% done             8.83ms  114.8 113.5
Shanghai      0.3000   1.3     0      1     42    5633 No             2    199.9% done             8.89ms  112.8 112.6
Shanghai      0.3000   1.4     0      1     42    5486 No             2    199.9% done             8.95ms  111.1 108.3
Shanghai      0.3000   1.5     0      1     42    5479 No             2    199.9% done             9.04ms  110.2 108.9
YasMarina     1.0000   0.5     0      1     42    2933 No             2    199.9% done             9.54ms  58.9 58.4
YasMarina     1.0000   0.6     0      1     42    2903 No             2    199.9% done             9.63ms  60.9 55.2
YasMarina     1.0000   0.7     0      1     42    2800 No             2    199.8% done             9.48ms  57.1 54.9
YasMarina     1.0000   0.8     0      1     42    2722 No             2    199.9% done             9.52ms  55.2 53.8
YasMarina     1.0000   0.9     0      1     42    2823 No             2    199.8% done             9.32ms  58.6 54.3
YasMarina     1.0000   1.0     0      1     42    2632 No             2    199.8% done             9.38ms  53.8 51.4
YasMarina     1.0000   1.1     0      1     42    2641 No             2    199.9% done             9.38ms  54.3 51.4
YasMarina     1.0000   1.2     0      1     42  CRASHED
YasMarina     1.0000   1.3     0      1     42    2630 No             2    199.8% done             9.23ms  53.0 52.2
YasMarina     1.0000   1.4     0      1     42    2554 No             2    199.8% done             9.31ms  52.2 49.9
YasMarina     1.0000   1.5     0      1     42    2468 No             2    199.8% done             9.56ms  51.0 47.7
YasMarina     0.9000   0.5     0      1     42  CRASHED
YasMarina     0.9000   0.6     0      1     42    3407 No             2    199.9% done             9.79ms  67.8 68.5
YasMarina     0.9000   0.7     0      1     42    3369 No             2    199.8% done             9.68ms  67.4 67.3
YasMarina     0.9000   0.8     0      1     42    3255 No             2    199.8% done             9.67ms  67.4 62.8
YasMarina     0.9000   0.9     0      1     42    3230 No             2    199.9% done             9.43ms  65.6 63.6
YasMarina     0.9000   1.0     0      1     42    3315 No             2    199.8% done             9.39ms  67.7 64.9
YasMarina     0.9000   1.1     0      1     42    3252 No             2    199.9% done             9.39ms  64.8 65.3
YasMarina     0.9000   1.2     0      1     42    3182 No             2    199.9% done             9.30ms  65.6 61.7
YasMarina     0.9000   1.3     0      1     42    3275 No             2    199.8% done             9.25ms  67.4 63.6
YasMarina     0.9000   1.4     0      1     42    3139 No             2    199.9% done             9.20ms  63.3 62.2
YasMarina     0.9000   1.5     0      1     42    3200 No             2    199.9% done             9.26ms  64.5 63.5
YasMarina     0.8000   0.5     0      1     42    3844 No             2    199.9% done             9.64ms  78.7 75.1
YasMarina     0.8000   0.6     0      1     42    3897 No             2    199.9% done             9.54ms  76.7 79.2
YasMarina     0.8000   0.7     0      1     42    3744 No             2    199.9% done             9.39ms  75.6 74.2
YasMarina     0.8000   0.8     0      1     42    3775 No             2    199.8% done             9.30ms  76.4 74.6
YasMarina     0.8000   0.9     0      1     42    3839 No             2    199.9% done             9.19ms  77.8 75.8
YasMarina     0.8000   1.0     0      1     42    3741 No             2    199.9% done             9.18ms  75.2 74.5
YasMarina     0.8000   1.1     0      1     42    3704 No             2    199.9% done             9.19ms  75.2 73.0
YasMarina     0.8000   1.2     0      1     42    3643 No             2    199.9% done             9.16ms  74.1 71.6
YasMarina     0.8000   1.3     0      1     42    3583 No             2    199.9% done             9.12ms  73.1 70.2
YasMarina     0.8000   1.4     0      1     42    3575 No             2    199.8% done             9.13ms  72.2 70.8
YasMarina     0.8000   1.5     0      1     42    3632 No             2    199.9% done             9.15ms  72.0 73.3
YasMarina     0.7000   0.5     0      1     42    3761 No             2    199.9% done             9.85ms  76.6 73.8
YasMarina     0.7000   0.6     0      1     42    3810 No             2    199.9% done             9.63ms  77.6 74.8
YasMarina     0.7000   0.7     0      1     42    3807 No             2    199.9% done             9.53ms  77.1 75.2
YasMarina     0.7000   0.8     0      1     42    3829 No             2    199.9% done             9.30ms  78.0 75.2
YasMarina     0.7000   0.9     0      1     42    3772 No             2    199.9% done             9.18ms  75.0 75.9
YasMarina     0.7000   1.0     0      1     42    3717 No             2    199.9% done             9.15ms  75.8 72.9
YasMarina     0.7000   1.1     0      1     42    3506 No             2    199.8% done             9.18ms  71.8 68.4
YasMarina     0.7000   1.2     0      1     42    3511 No             2    199.8% done             9.09ms  71.7 68.8
YasMarina     0.7000   1.3     0      1     42    3556 No             2    199.9% done             9.01ms  71.7 70.5
YasMarina     0.7000   1.4     0      1     42    3503 No             2    199.9% done             9.00ms  70.6 69.6
YasMarina     0.7000   1.5     0      1     42  CRASHED
YasMarina     0.6000   0.5     0      1     42    3548 No             2    199.9% done             9.74ms  71.4 70.6
YasMarina     0.6000   0.6     0      1     42    3495 No             2    199.8% done             9.64ms  70.4 69.4
YasMarina     0.6000   0.7     0      1     42    3416 No             2    199.8% done             9.37ms  69.1 67.5
YasMarina     0.6000   0.8     0      1     42    3463 No             2    199.9% done             9.30ms  69.5 69.0
YasMarina     0.6000   0.9     0      1     42    3399 No             2    199.9% done             9.09ms  68.7 67.3
YasMarina     0.6000   1.0     0      1     42    3338 No             2    199.9% done             9.10ms  68.3 65.2
YasMarina     0.6000   1.1     0      1     42    3198 No             2    199.9% done             9.03ms  64.2 63.7
YasMarina     0.6000   1.2     0      1     42    3172 No             2    199.9% done             8.94ms  63.5 63.4
YasMarina     0.6000   1.3     0      1     42    3158 No             2    199.8% done             8.93ms  62.4 64.0
YasMarina     0.6000   1.4     0      1     42  CRASHED
YasMarina     0.6000   1.5     0      1     42    3052 No             2    199.8% done             9.02ms  61.8 60.2
YasMarina     0.5000   0.5     0      1     42    3871 No             2    199.9% done             9.34ms  78.6 76.3
YasMarina     0.5000   0.6     0      1     42    3823 No             2    199.9% done             9.18ms  79.0 73.9
YasMarina     0.5000   0.7     0      1     42    3820 No             2    199.9% done             8.99ms  76.2 76.6
YasMarina     0.5000   0.8     0      1     42    3781 No             2    199.8% done             9.02ms  76.5 74.8
YasMarina     0.5000   0.9     0      1     42    3691 No             2    199.8% done             8.85ms  74.6 73.0
YasMarina     0.5000   1.0     0      1     42    3627 No             2    199.9% done             8.98ms  71.5 73.6
YasMarina     0.5000   1.1     0      1     42    3401 No             2    199.9% done             8.82ms  68.0 68.0
YasMarina     0.5000   1.2     0      1     42    3289 No             2    199.9% done             8.80ms  67.4 64.1
YasMarina     0.5000   1.3     0      1     42    3153 No             2    199.9% done             8.81ms  64.1 62.0
YasMarina     0.5000   1.4     0      1     42    3134 No             2    199.9% done             8.70ms  63.7 61.7
YasMarina     0.5000   1.5     0      1     42  CRASHED
YasMarina     0.4000   0.5     0      1     42  CRASHED
YasMarina     0.4000   0.6     0      1     42    6883 No             2    199.9% done             8.58ms  138.3 137.0
YasMarina     0.4000   0.7     0      1     42    6687 No             2    199.9% done             8.42ms  134.3 133.2
YasMarina     0.4000   0.8     0      1     42    6213 No             2    199.9% done             9.10ms  120.9 127.6
YasMarina     0.4000   0.9     0      1     42    6047 No             2    199.9% done             8.66ms  125.9 116.0
YasMarina     0.4000   1.0     0      1     42    5847 No             2    199.9% done             9.08ms  118.3 115.6
YasMarina     0.4000   1.1     0      1     42    5137 No             2    199.9% done             9.12ms  102.5 103.0
YasMarina     0.4000   1.2     0      1     42    5123 No             2    199.9% done             9.91ms  101.2 103.7
YasMarina     0.4000   1.3     0      1     42  CRASHED
YasMarina     0.4000   1.4     0      1     42    4465 No             2    199.9% done             9.62ms  90.4 88.2
YasMarina     0.4000   1.5     0      1     42    4383 No             2    199.9% done             9.48ms  89.1 86.2
YasMarina     0.3000   0.5     0      1     42    6826 No             2    199.9% done             8.98ms  136.4 136.7
YasMarina     0.3000   0.6     0      1     42    6596 No             2    199.9% done             9.00ms  132.9 131.0
YasMarina     0.3000   0.7     0      1     42    6335 No             2    199.9% done             9.34ms  125.9 127.5
YasMarina     0.3000   0.8     0      1     42    6107 No             2    199.9% done             9.56ms  122.5 121.8
YasMarina     0.3000   0.9     0      1     42    5772 No             2    199.9% done             9.32ms  115.5 115.4
YasMarina     0.3000   1.0     0      1     42    5481 No             2    199.9% done             9.17ms  109.2 110.0
YasMarina     0.3000   1.1     0      1     42    5329 No             2    199.9% done             9.27ms  108.0 105.2
YasMarina     0.3000   1.2     0      1     42    5203 No             2    199.9% done             9.22ms  104.3 103.8
YasMarina     0.3000   1.3     0      1     42  CRASHED
YasMarina     0.3000   1.4     0      1     42    5067 No             2    199.9% done             9.39ms  102.7 100.0
YasMarina     0.3000   1.5     0      1     42    4960 No             2    199.9% done             9.37ms  98.8 99.6
"""

# ── Structured parsing ───────────────────────────────────────────────────────

tracks_order = ["Sepang", "Shanghai", "YasMarina"]
mus_order = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
lws_order = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# result[track][(mu, lw)] = {"status": ..., "avg_lap": ..., "lap_times": [...]}
results = {t: {} for t in tracks_order}

for line in raw.strip().split("\n"):
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    track = parts[0]
    mu = float(parts[1])
    lw = float(parts[2])

    if "CRASHED" in line:
        results[track][(mu, lw)] = {"status": "crashed", "avg_lap": None}
    elif "COLLISION" in line:
        results[track][(mu, lw)] = {"status": "collision", "avg_lap": None}
    else:
        # Last two numbers are lap times
        lap_times = [float(parts[-2]), float(parts[-1])]
        avg_lap = np.mean(lap_times)
        results[track][(mu, lw)] = {"status": "done", "avg_lap": avg_lap}


# ── Plot ─────────────────────────────────────────────────────────────────────

COLOR_GREEN = "#2ECC71"
COLOR_RED = "#D9534F"
COLOR_BLUE = "#4A90D9"
COLOR_GRAY = "#CCCCCC"

fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

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
                    fontsize=6.5, fontweight="bold", color=text_color)

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
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("MCTS + GP Action Prior: Avg Lap Time vs Real Track Width & Friction\n"
             "(lane_width=0.5, heading backward check enabled, no soft penalty)",
             fontsize=13, fontweight="bold", y=1.06)

plt.savefig("heatmap_results.png", dpi=150, bbox_inches="tight")
plt.savefig("heatmap_results.pdf", bbox_inches="tight")
print("Saved heatmap_results.png and heatmap_results.pdf")
