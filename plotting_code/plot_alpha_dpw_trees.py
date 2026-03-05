"""Visualize how alpha_dpw affects MCTS-DPW tree growth.

Simulates a simplified DPW tree-building process (abstract rewards)
and draws the tree at iterations 1, 10, 100, 1000 for several
alpha_dpw values.  Best path (most-visited) is highlighted in red.

Progressive widening rule (from mcts_native.hpp SPWNative::select):
    if  N^alpha  >=  num_children   →  sample NEW action
    else                             →  pick best existing child (UCB)
"""

import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

_DIR = Path(__file__).resolve().parent

# ── Simplified MCTS-DPW tree ─────────────────────────────────────────────────

class Node:
    __slots__ = ("id", "children", "visits", "total_reward", "depth", "parent")

    _counter = 0

    def __init__(self, depth=0, parent=None):
        Node._counter += 1
        self.id = Node._counter
        self.children: list["Node"] = []
        self.visits = 0
        self.total_reward = 0.0
        self.depth = depth
        self.parent = parent

    @property
    def avg_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0.0


def reset_counter():
    Node._counter = 0


def build_tree(alpha_dpw: float, n_iterations: int, max_depth: int = 6,
               K: float = 1.0, seed: int = 42) -> Node:
    """Simulate DPW tree growth for `n_iterations` iterations."""
    rng = random.Random(seed)
    reset_counter()
    root = Node(depth=0)
    root.visits = 1  # root starts with 1 visit (standard MCTS)

    for _ in range(n_iterations):
        node = root
        path = [node]

        # ── Selection + expansion ────────────────────────────────────
        while node.depth < max_depth:
            n_children = len(node.children)
            # Progressive widening check
            if math.pow(node.visits, alpha_dpw) >= n_children:
                # Expand: add a new child
                child = Node(depth=node.depth + 1, parent=node)
                node.children.append(child)
                node = child
                path.append(node)
                break
            else:
                # UCB select among existing children
                best = None
                best_score = -float("inf")
                log_n = math.log(node.visits) if node.visits > 0 else 0
                for c in node.children:
                    if c.visits == 0:
                        score = float("inf")
                    else:
                        score = c.avg_reward + K * math.sqrt(log_n / c.visits)
                    if score > best_score:
                        best_score = score
                        best = c
                node = best
                path.append(node)

        # ── Simulation (random reward) ───────────────────────────────
        reward = rng.gauss(0.5, 0.3)

        # ── Backpropagation ──────────────────────────────────────────
        for n in path:
            n.visits += 1
            n.total_reward += reward

    return root


def get_best_path(root: Node) -> set:
    """Trace the most-visited path from root to leaf. Returns set of node ids."""
    ids = set()
    node = root
    ids.add(node.id)
    while node.children:
        best = max(node.children, key=lambda c: c.visits)
        ids.add(best.id)
        node = best
    return ids


# ── Hierarchical layout (Buchheim-style spacing) ─────────────────────────────

def layout_tree(root: Node):
    """Recursive layout that gives each subtree proportional horizontal space.

    Returns dict {node_id: (x, y)}.
    """
    positions = {}

    def _subtree_width(node, depth_limit=7):
        """Count leaf-level slots (or 1 if leaf)."""
        if not node.children or node.depth >= depth_limit:
            return 1
        return sum(_subtree_width(c, depth_limit) for c in node.children)

    def _layout(node, x_left, x_right, depth_limit=7):
        x_center = (x_left + x_right) / 2.0
        positions[node.id] = (x_center, -node.depth)

        if not node.children or node.depth >= depth_limit:
            return

        total_w = _subtree_width(node, depth_limit)
        span = x_right - x_left
        cursor = x_left
        for c in node.children:
            cw = _subtree_width(c, depth_limit)
            frac = cw / total_w
            cx_left = cursor
            cx_right = cursor + span * frac
            _layout(c, cx_left, cx_right, depth_limit)
            cursor = cx_right

    total = _subtree_width(root)
    _layout(root, -total / 2, total / 2)
    return positions


def _normalize_positions(positions):
    """Scale positions into [-1, 1] for x and [-1, 0] for y."""
    if not positions:
        return positions
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    return {
        nid: ((x - x_min) / x_range * 2 - 1,
              (y - y_min) / y_range * (-1.0))
        for nid, (x, y) in positions.items()
    }


def draw_tree(ax, root: Node, max_nodes_draw: int = 2000):
    """Draw the tree on a matplotlib Axes."""
    best_path = get_best_path(root)
    positions = layout_tree(root)
    positions = _normalize_positions(positions)

    # Collect edges and nodes via BFS
    edges_normal = []
    edges_best = []
    nodes_normal = []
    nodes_best = []
    queue = [root]
    count = 0
    while queue and count < max_nodes_draw:
        node = queue.pop(0)
        on_best = node.id in best_path
        if on_best:
            nodes_best.append(node)
        else:
            nodes_normal.append(node)
        count += 1
        for c in node.children:
            parent_best = node.id in best_path and c.id in best_path
            if parent_best:
                edges_best.append((node, c))
            else:
                edges_normal.append((node, c))
            queue.append(c)

    # Draw normal edges
    for parent, child in edges_normal:
        if parent.id not in positions or child.id not in positions:
            continue
        px, py = positions[parent.id]
        cx, cy = positions[child.id]
        ax.plot([px, cx], [py, cy], color="#B0BEC5", linewidth=0.4,
                alpha=0.6, zorder=1, solid_capstyle="round")

    # Draw best edges on top
    for parent, child in edges_best:
        if parent.id not in positions or child.id not in positions:
            continue
        px, py = positions[parent.id]
        cx, cy = positions[child.id]
        ax.plot([px, cx], [py, cy], color="#E74C3C", linewidth=2.5,
                alpha=1.0, zorder=4, solid_capstyle="round")

    # Draw normal nodes
    for node in nodes_normal:
        if node.id not in positions:
            continue
        x, y = positions[node.id]
        ax.scatter(x, y, s=12, c="#5DADE2", edgecolors="#2E86C1",
                   linewidths=0.3, zorder=2)

    # Draw best-path nodes on top
    for node in nodes_best:
        if node.id not in positions:
            continue
        x, y = positions[node.id]
        ax.scatter(x, y, s=50, c="#E74C3C", edgecolors="#C0392B",
                   linewidths=0.8, zorder=5)

    # Stats text
    total_nodes = _count_nodes(root)
    max_d = _max_depth(root) - root.depth
    max_w = _max_width(root)
    root_children = len(root.children)
    stat = f"nodes={total_nodes}   depth={max_d}   root_children={root_children}   max_width={max_w}"
    ax.text(0.5, 0.02, stat, transform=ax.transAxes,
            ha="center", fontsize=7, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#DDDDDD", alpha=0.8))

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 0.15)
    ax.axis("off")


def _count_nodes(root):
    count = 1
    for c in root.children:
        count += _count_nodes(c)
    return count


def _max_depth(root):
    if not root.children:
        return root.depth
    return max(_max_depth(c) for c in root.children)


def _max_width(root):
    depth_counts = defaultdict(int)
    queue = [root]
    while queue:
        node = queue.pop(0)
        depth_counts[node.depth] += 1
        for c in node.children:
            queue.append(c)
    return max(depth_counts.values()) if depth_counts else 0


# ── Main ─────────────────────────────────────────────────────────────────────

alphas = [0.1, 0.3, 0.5, 0.8]
iterations = [1, 10, 100, 1000]

fig, axes = plt.subplots(len(alphas), len(iterations),
                         figsize=(22, 18), constrained_layout=True)

for row, alpha in enumerate(alphas):
    for col, n_iter in enumerate(iterations):
        ax = axes[row, col]
        tree = build_tree(alpha, n_iter, max_depth=6, K=1.0, seed=42)
        draw_tree(ax, tree)

    # Row label
    axes[row, 0].annotate(
        f"α = {alpha}", xy=(-0.22, 0.5),
        xycoords="axes fraction", fontsize=13, fontweight="bold",
        ha="center", va="center", rotation=90,
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8F0FE", ec="#5DADE2", lw=1.5))

# Column headers
for col, n_iter in enumerate(iterations):
    axes[0, col].set_title(f"Iteration {n_iter}", fontsize=13,
                           fontweight="bold", pad=12)

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
           markeredgecolor="#C0392B", markersize=10, label="Best path (most visited)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#5DADE2",
           markeredgecolor="#2E86C1", markersize=7, label="Other nodes"),
    Line2D([0], [0], color="#E74C3C", linewidth=2.5, label="Best-path edges"),
    Line2D([0], [0], color="#B0BEC5", linewidth=1.0, label="Other edges"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

fig.suptitle(
    "Effect of α_dpw on MCTS-DPW Tree Growth\n"
    "Rule: if  N^α ≥ num_children → sample new action (explore),  else → UCB select (exploit)\n"
    "Low α → narrow & deep (exploit-heavy)     |     High α → wide & shallow (explore-heavy)",
    fontsize=14, fontweight="bold", y=1.05)

out_png = _DIR / "alpha_dpw_tree_visualization.png"
out_pdf = _DIR / "alpha_dpw_tree_visualization.pdf"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
