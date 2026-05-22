"""Polished ternary plot for W matrix visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
FAINT_GRAY = '#f0f0f0'

FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],
    [0.31, 0.44, 0.75],
    [0.44, 0.69, 0.25],
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']


def blend_color(weights):
    weights = np.array(weights) / np.sum(weights)
    return weights @ FACTOR_COLORS


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
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    return W, order


def barycentric_to_cartesian(w, vertices):
    """Convert barycentric coordinates to cartesian."""
    return w[0] * vertices[0] + w[1] * vertices[1] + w[2] * vertices[2]


def draw_ternary_polished(ax, W, item_labels=None, show_grid=True):
    """Publication-quality ternary plot."""
    n = len(W)

    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    scale = 0.82
    cy = 0.42  # center y
    vertices = np.array([
        [0.5, cy + h * scale * 0.6],    # top (factor 0)
        [0.5 - scale/2, cy - h * scale * 0.4],  # bottom-left (factor 1)
        [0.5 + scale/2, cy - h * scale * 0.4],  # bottom-right (factor 2)
    ])

    # Subtle grid lines
    if show_grid:
        for level in [0.33, 0.67]:
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                w1 = np.zeros(3)
                w1[i] = level
                w1[j] = 1 - level
                p1 = barycentric_to_cartesian(w1, vertices)

                w2 = np.zeros(3)
                w2[i] = level
                w2[k] = 1 - level
                p2 = barycentric_to_cartesian(w2, vertices)

                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='#e8e8e8', lw=0.6, zorder=0)

    # Triangle fill (very subtle)
    triangle = mpatches.Polygon(vertices, closed=True, facecolor='#fafafa',
                                 edgecolor='none', zorder=0)
    ax.add_patch(triangle)

    # Triangle edges
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([vertices[i][0], vertices[j][0]],
                [vertices[i][1], vertices[j][1]],
                color=LIGHT_GRAY, lw=1.8, zorder=1, solid_capstyle='round')

    # Corner markers (small circles at vertices)
    for i in range(3):
        marker = mpatches.Circle(vertices[i], 0.018, facecolor=FACTOR_HEX[i],
                                  edgecolor='white', lw=1, zorder=2)
        ax.add_patch(marker)

    # Factor labels
    factor_labels = ['animate', 'round', 'natural']
    label_offsets = [
        (0, 0.065),       # top
        (-0.07, -0.045),  # bottom-left
        (0.07, -0.045),   # bottom-right
    ]
    for i, (v, label, (dx, dy)) in enumerate(zip(vertices, factor_labels, label_offsets)):
        ax.text(v[0] + dx, v[1] + dy, label, fontsize=9, ha='center', va='center',
                color=FACTOR_HEX[i], fontweight='semibold', style='italic')

    # Compute initial positions
    positions = np.array([barycentric_to_cartesian(W[i], vertices) for i in range(n)])

    # Collision resolution - push overlapping nodes apart
    node_radius = 0.024
    min_dist = node_radius * 2.2
    for _ in range(80):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 0.001:
                    push = (min_dist - dist) / 2 * diff / dist * 0.5
                    positions[i] += push
                    positions[j] -= push
                    moved = True
        if not moved:
            break

    # Draw nodes
    for i in range(n):
        color = blend_color(W[i])
        circle = mpatches.Circle(positions[i], node_radius, facecolor=color,
                                  edgecolor='white', lw=1.2, zorder=3)
        ax.add_patch(circle)

    # Label placement
    if item_labels:
        center = np.mean(vertices, axis=0)
        placed_labels = []

        for i in range(n):
            if i >= len(item_labels):
                continue

            pos = positions[i]
            direction = pos - center
            norm = np.linalg.norm(direction)
            if norm > 0.001:
                direction = direction / norm

            # Base label position
            label_dist = 0.05
            label_pos = pos + direction * label_dist

            # Adjust for edge proximity
            if pos[1] < center[1] - 0.1:  # near bottom
                label_pos[1] = pos[1] - 0.04
                va = 'top'
            elif pos[1] > center[1] + 0.15:  # near top
                label_pos[1] = pos[1] + 0.04
                va = 'bottom'
            else:
                va = 'center'

            # Horizontal alignment
            if direction[0] > 0.2:
                ha = 'left'
                label_pos[0] = pos[0] + 0.035
            elif direction[0] < -0.2:
                ha = 'right'
                label_pos[0] = pos[0] - 0.035
            else:
                ha = 'center'

            ax.text(label_pos[0], label_pos[1], item_labels[i], fontsize=5.5,
                    ha=ha, va=va, color=CHARCOAL, alpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.92)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, order = create_data()
    W_ord = W[order]

    item_labels = ['dog', 'cat', 'bird', 'fish',
                   'ball', 'orange', 'apple', 'wheel',
                   'tree', 'rock', 'leaf', 'cloud',
                   'turtle', 'egg', 'snail']
    item_labels_ord = [item_labels[i] for i in order]

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Main figure
    fig, ax = plt.subplots(figsize=(5, 4.5))
    draw_ternary_polished(ax, W_ord, item_labels_ord)
    fig.savefig(output_dir / "w_ternary_polished.svg", format='svg',
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print("Saved: w_ternary_polished.svg")

    # Clean version without labels (for flexibility)
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    draw_ternary_polished(ax2, W_ord, item_labels=None)
    fig2.savefig(output_dir / "w_ternary_clean.svg", format='svg',
                 bbox_inches='tight', pad_inches=0.08)
    plt.close(fig2)
    print("Saved: w_ternary_clean.svg")


if __name__ == "__main__":
    main()
