"""Elegant arc diagram - nodes on line, arcs show similarity.

Very clean, modern visualization style.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def create_data(seed=42):
    rng = np.random.default_rng(seed)
    W_list = []

    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    W_list.append([0.46, 0.46, 0.08])
    W_list.append([0.47, 0.07, 0.46])
    W_list.append([0.07, 0.47, 0.46])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T
    return W, S


def draw_arc(ax, x1, x2, height_scale=0.5, **kwargs):
    """Draw a semicircular arc between two x positions."""
    center = (x1 + x2) / 2
    width = abs(x2 - x1)
    height = width * height_scale

    # Draw arc using path
    theta = np.linspace(0, np.pi, 50)
    x = center + (width / 2) * np.cos(theta)
    y = height * np.sin(theta) / 2

    ax.plot(x, y, **kwargs)


def main():
    W, S = create_data()
    n = len(W)

    # Order nodes by dominant factor
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    # X positions with gaps between groups
    positions = {}
    x = 0
    prev_dom = -1

    for idx, node in enumerate(order):
        dom = dominant[node]
        if dom != prev_dom and prev_dom >= 0:
            x += 0.3  # gap between groups
        positions[node] = x
        x += 1
        prev_dom = dom

    # Normalize to [0, 1]
    max_x = max(positions.values())
    positions = {k: v / max_x for k, v in positions.items()}

    fig, ax = plt.subplots(figsize=(5.0, 2.2))

    # Draw arcs for similarities
    threshold = 0.18
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                x1, x2 = positions[i], positions[j]
                if x1 > x2:
                    x1, x2 = x2, x1

                alpha = 0.2 + 0.6 * (S[i, j] - threshold) / (1 - threshold)
                lw = 0.4 + 0.8 * (S[i, j] - threshold) / (1 - threshold)

                draw_arc(ax, x1, x2, height_scale=0.8,
                        color=LIGHT_GRAY, linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes
    node_y = 0
    node_radius = 0.025

    for node in range(n):
        x = positions[node]
        dom = dominant[node]
        max_w = W[node].max()

        if max_w < 0.55:
            circle = mpatches.Circle(
                (x, node_y), node_radius,
                facecolor='white',
                edgecolor=CHARCOAL,
                linewidth=1.0,
                zorder=3
            )
        else:
            circle = mpatches.Circle(
                (x, node_y), node_radius,
                facecolor=FACTOR_COLORS[dom],
                edgecolor='white',
                linewidth=0.6,
                zorder=3
            )
        ax.add_patch(circle)

    # Factor labels below
    # Find center of each group
    for k in range(3):
        group_nodes = [node for node in range(n) if dominant[node] == k]
        if group_nodes:
            center_x = np.mean([positions[node] for node in group_nodes])
            ax.text(center_x, -0.12, f'F{k+1}', fontsize=9, ha='center', va='top',
                    color=FACTOR_COLORS[k], fontweight='medium')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.18, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_arc.svg", format='svg',
                bbox_inches='tight', pad_inches=0.03)
    fig.savefig(output_dir / "elegant_arc.png", dpi=200,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    print("Saved: elegant_arc.svg")


if __name__ == "__main__":
    main()
