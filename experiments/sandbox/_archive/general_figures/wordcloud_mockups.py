"""Word cloud visualizations - connected graph with gradient blobs and contour rings."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Colors matching the original
ORANGE = '#E07040'
RED = '#D64545'
GREEN = '#4CAF50'
PURPLE = '#7E57C2'


def setup_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


# Node positions and edges - carefully aligned to avoid text/edge overlap
# Format: (x, y, label, color, label_offset_x, label_offset_y)
NODES = [
    (0.12, 0.75, 'Green', ORANGE, -0.06, 0),
    (0.28, 0.75, 'Grass', ORANGE, 0, 0.05),
    (0.48, 0.68, 'Savanna', RED, 0, -0.055),
    (0.42, 0.82, 'Lion', RED, 0, 0.05),
    (0.62, 0.75, 'Tiger', RED, 0.06, 0),
    (0.18, 0.45, 'Drink', GREEN, -0.06, 0),
    (0.32, 0.52, 'Water', GREEN, 0, 0.05),
    (0.45, 0.38, 'Beer', GREEN, 0, -0.055),
    (0.22, 0.28, 'Wine', GREEN, 0, -0.055),
    (0.62, 0.45, 'Man', PURPLE, 0.055, 0),
    (0.78, 0.52, 'Boss', PURPLE, 0.055, 0),
    (0.72, 0.32, 'Work', PURPLE, 0, -0.055),
]

EDGES = [
    (0, 1),   # Green - Grass
    (1, 2),   # Grass - Savanna
    (2, 3),   # Savanna - Lion
    (2, 4),   # Savanna - Tiger
    (5, 6),   # Drink - Water
    (6, 7),   # Water - Beer
    (5, 8),   # Drink - Wine
    (7, 9),   # Beer - Man
    (9, 10),  # Man - Boss
    (9, 11),  # Man - Work
]

# Cluster centers for background effects
CLUSTERS = [
    (0.20, 0.75, ORANGE),   # Nature
    (0.52, 0.75, RED),      # Animals
    (0.30, 0.40, GREEN),    # Drinks
    (0.70, 0.43, PURPLE),   # Work
]


def draw_edges(ax):
    """Draw connecting edges."""
    for i, j in EDGES:
        x1, y1 = NODES[i][0], NODES[i][1]
        x2, y2 = NODES[j][0], NODES[j][1]
        ax.plot([x1, x2], [y1, y2], color='#444', lw=1.2, zorder=1)


def draw_nodes(ax, node_size=0.028):
    """Draw nodes with labels positioned to avoid edges."""
    for x, y, label, color, lx, ly in NODES:
        circle = mpatches.Circle((x, y), node_size, facecolor=color,
                                edgecolor='white', lw=0.8, zorder=3)
        ax.add_patch(circle)
        # Position label based on offset
        if lx != 0:  # Label to left or right
            ha = 'right' if lx < 0 else 'left'
            ax.text(x + lx, y, label, fontsize=7, ha=ha, va='center', color='#333', zorder=4)
        else:  # Label above or below
            va = 'bottom' if ly > 0 else 'top'
            ax.text(x, y + ly, label, fontsize=7, ha='center', va=va, color='#333', zorder=4)


def draw_gradient_blobs(ax):
    """Connected graph with soft gradient blobs."""
    # Draw gradient blobs first (background)
    for cx, cy, color in CLUSTERS:
        for r, alpha in [(0.22, 0.06), (0.16, 0.10), (0.10, 0.15), (0.05, 0.20)]:
            circle = mpatches.Circle((cx, cy), r, facecolor=color,
                                    alpha=alpha, edgecolor='none', zorder=0)
            ax.add_patch(circle)

    draw_edges(ax)
    draw_nodes(ax)


def draw_contour_rings(ax):
    """Connected graph with contour rings."""
    # Draw contour rings first (background)
    for cx, cy, color in CLUSTERS:
        for r, lw, alpha in [(0.20, 0.6, 0.3), (0.14, 1.0, 0.5), (0.08, 1.4, 0.7)]:
            circle = mpatches.Circle((cx, cy), r, facecolor='none',
                                    edgecolor=color, lw=lw, alpha=alpha, zorder=0)
            ax.add_patch(circle)
        # Soft center fill
        circle = mpatches.Circle((cx, cy), 0.04, facecolor=color,
                                alpha=0.2, edgecolor='none', zorder=0)
        ax.add_patch(circle)

    draw_edges(ax)
    draw_nodes(ax)


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    versions = [
        (draw_gradient_blobs, "wordcloud_gradient"),
        (draw_contour_rings, "wordcloud_contour"),
    ]

    for func, name in versions:
        fig, ax = plt.subplots(figsize=(5, 5))
        setup_ax(ax)
        func(ax)
        fig.savefig(output_dir / f"{name}.svg", format='svg',
                   bbox_inches='tight', pad_inches=0.05, transparent=True)
        plt.close(fig)
        print(f"Saved: {name}.svg")


if __name__ == "__main__":
    main()
