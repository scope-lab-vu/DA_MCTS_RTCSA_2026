"""Render the Budget-Aware MCTS algorithm as a clean PDF/PNG using matplotlib."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

_DIR = Path(__file__).resolve().parent

lines = [
    ("bold",  "Algorithm 1: Budget-Aware MCTS with Adaptive Action Sampling"),
    ("space", ""),
    ("text",  "Input:  Time budget T, tree root x\u2080, progressive widening exponent \u03b1"),
    ("text",  "Output: Best action a*"),
    ("space", ""),
    ("comment", "% Hyperparameters: W_low = 15, W_high = 50, Nw = 100, Nr = 200"),
    ("rule",  ""),
    ("comment", "% Initialize with equal sampling (no bias at episode start)"),
    ("code",  " 1:  p \u2190 (p_agg, p_slow, p_feas) = (1/3, 1/3, 1/3)"),
    ("code",  " 2:  t_start \u2190 Clock();  t_deadline \u2190 t_start + T;  n \u2190 0"),
    ("space", ""),
    ("header", "\u2500\u2500 Phase 1: Warmup (measure iteration rate) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),
    ("code",  " 3:  while n < Nw and Clock() < t_deadline do"),
    ("code",  " 4:      GrowTree(x\u2080, p);  n \u2190 n + 1"),
    ("code",  " 5:  end while"),
    ("space", ""),
    ("header", "\u2500\u2500 Phase 2: Estimate budget & adapt weights \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),
    ("code",  " 6:  t\u0304_iter \u2190 (Clock() \u2212 t_start) / n                     \u25b7 avg time per iter"),
    ("code",  " 7:  N_total \u2190 n + \u230a(t_deadline \u2212 Clock()) / t\u0304_iter\u230b       \u25b7 estimated total iters"),
    ("code",  " 8:  \u0174 \u2190 N_total^\u03b1                                         \u25b7 expected root width"),
    ("code",  " 9:  p \u2190 ComputeWeights(\u0174)"),
    ("space", ""),
    ("header", "\u2500\u2500 Phase 3: Adaptive search with re-calibration \u2500\u2500\u2500\u2500\u2500\u2500"),
    ("code",  "10:  while Clock() < t_deadline do"),
    ("code",  "11:      GrowTree(x\u2080, p);  n \u2190 n + 1"),
    ("code",  "12:      if n mod Nr = 0 then                               \u25b7 re-calibrate"),
    ("code",  "13:          t\u0304_iter \u2190 (Clock() \u2212 t_start) / n"),
    ("code",  "14:          N_total \u2190 n + \u230a(t_deadline \u2212 Clock()) / t\u0304_iter\u230b"),
    ("code",  "15:          \u0174 \u2190 N_total^\u03b1"),
    ("code",  "16:          p \u2190 ComputeWeights(\u0174)"),
    ("code",  "17:      end if"),
    ("code",  "18:  end while"),
    ("space", ""),
    ("code",  "19:  return a* \u2190 argmax over children(x\u2080) of visits(a)"),
    ("rule",  ""),
    ("bold",  "function ComputeWeights(\u0174):"),
    ("code",  "20:  \u03c4 \u2190 clamp((\u0174 \u2212 15) / 35,  0,  1)                      \u25b7 budget interpolant"),
    ("space", ""),
    ("comment", "    Tight budget (\u03c4 \u2192 0): favor braking for safety"),
    ("comment", "    Ample budget (\u03c4 \u2192 1): favor GP prior + aggressive for speed"),
    ("space", ""),
    ("code",  "21:  p_slow \u2190 0.70 \u2212 0.60 \u00b7 \u03c4                              \u25b7 0.70 \u2192 0.10"),
    ("code",  "22:  p_feas \u2190 0.20 + 0.35 \u00b7 \u03c4                              \u25b7 0.20 \u2192 0.55"),
    ("code",  "23:  p_agg  \u2190 1 \u2212 p_slow \u2212 p_feas                          \u25b7 0.10 \u2192 0.35"),
    ("code",  "24:  return (p_agg, p_slow, p_feas)"),
    ("bold",  "end function"),
    ("rule",  ""),
    ("bold",  "function SampleAction(p):"),
    ("comment", "                                    \u25b7 used inside GrowTree on curves"),
    ("code",  "25:  Draw u ~ Uniform(0, 1)"),
    ("code",  "26:  if u < p_agg then"),
    ("code",  "27:      a_accel ~ Uniform(0, a_max)                        \u25b7 aggressive (accelerate)"),
    ("code",  "28:  else if u < p_agg + p_slow then"),
    ("code",  "29:      a_accel ~ Uniform(\u2212a_max, 0)                       \u25b7 slow (brake)"),
    ("code",  "30:  else"),
    ("code",  "31:      a_accel ~ N(\u03bc_GP, \u03c3\u00b2_GP)                            \u25b7 feasible (GP prior)"),
    ("code",  "32:  end if"),
    ("code",  "33:  a_steer ~ N(\u03bc_GP^sv, \u03c3\u00b2_GP^sv)                        \u25b7 steering from GP prior"),
    ("code",  "34:  return (a_accel, a_steer)"),
    ("bold",  "end function"),
    ("rule",  ""),
    ("space", ""),
    ("bold",  "Hyperparameters:"),
    ("space", ""),
    ("text",  "  Symbol      Value     Description"),
    ("text",  "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),
    ("text",  "  T           35 ms     MCTS time budget per planning step"),
    ("text",  "  \u03b1           0.5       Action progressive widening exponent"),
    ("text",  "  W_low       15        Root width below which budget is \"tight\""),
    ("text",  "  W_high      50        Root width above which budget is \"ample\""),
    ("text",  "  Nw          100       Warmup iterations (equal 1/3 sampling)"),
    ("text",  "  Nr          200       Re-calibration interval"),
    ("space", ""),
    ("bold",  "Endpoint sampling distributions:"),
    ("space", ""),
    ("text",  "  Budget regime            p_agg    p_slow    p_feas    Behavior"),
    ("text",  "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),
    ("text",  "  Tight (\u0174 \u2264 15)           0.10     0.70      0.20      Mostly braking \u2192 safety"),
    ("text",  "  Ample (\u0174 \u2265 50)           0.35     0.10      0.55      GP + accel \u2192 speed"),
    ("text",  "  Episode start            1/3      1/3       1/3       Equal (before calibration)"),
]

STYLES = {
    "bold":    {"fontsize": 11, "fontweight": "bold", "color": "#1a1a1a", "fontfamily": "monospace"},
    "code":    {"fontsize": 9.5, "fontweight": "normal", "color": "#2c3e50", "fontfamily": "monospace"},
    "text":    {"fontsize": 9.5, "fontweight": "normal", "color": "#333333", "fontfamily": "monospace"},
    "comment": {"fontsize": 9, "fontweight": "normal", "color": "#27ae60", "fontstyle": "italic", "fontfamily": "monospace"},
    "header":  {"fontsize": 9.5, "fontweight": "bold", "color": "#8e44ad", "fontfamily": "monospace"},
    "space":   {},
    "rule":    {},
}

fig, ax = plt.subplots(figsize=(12, 19))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

y = 0.98
line_height = 0.0105
space_height = 0.005

for kind, text in lines:
    if kind == "space":
        y -= space_height
        continue
    if kind == "rule":
        ax.axhline(y=y, xmin=0.03, xmax=0.97, color="#CCCCCC", linewidth=0.8)
        y -= space_height
        continue
    style = STYLES[kind]
    ax.text(0.04, y, text, transform=ax.transAxes, **style)
    y -= line_height

border = mpatches.FancyBboxPatch(
    (0.02, y - 0.008), 0.96, 0.98 - y + 0.008,
    boxstyle="round,pad=0.01", facecolor="#FAFAFA",
    edgecolor="#333333", linewidth=1.5, zorder=-1,
    transform=ax.transAxes)
ax.add_patch(border)

out_png = _DIR / "budget_aware_mcts_algorithm.png"
out_pdf = _DIR / "budget_aware_mcts_algorithm.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
