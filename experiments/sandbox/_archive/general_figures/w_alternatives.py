"""Alternative visualizations for W matrix (factor loadings)."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

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


def draw_ternary(ax, W, item_labels=None):
    """Ternary/simplex plot - each item is a point in a triangle.

    Corners represent pure factors, position = barycentric coordinates from W.
    This is mathematically honest since W rows sum to 1.
    """
    n = len(W)

    # Triangle vertices (equilateral)
    v0 = np.array([0.5, 0.9])   # top (factor 0 - animate)
    v1 = np.array([0.1, 0.15])  # bottom-left (factor 1 - round)
    v2 = np.array([0.9, 0.15])  # bottom-right (factor 2 - natural)

    # Draw triangle edges
    for va, vb in [(v0, v1), (v1, v2), (v2, v0)]:
        ax.plot([va[0], vb[0]], [va[1], vb[1]], color=LIGHT_GRAY, lw=1.5, zorder=1)

    # Factor labels at corners
    factor_labels = ['animate', 'round', 'natural']
    offsets = [(0, 0.06), (-0.08, -0.06), (0.08, -0.06)]
    for i, (v, label, (dx, dy)) in enumerate(zip([v0, v1, v2], factor_labels, offsets)):
        ax.text(v[0] + dx, v[1] + dy, label, fontsize=8, ha='center',
                color=FACTOR_HEX[i], fontweight='medium', style='italic')

    # Plot items as points using barycentric coordinates
    for i in range(n):
        # Barycentric: position = w0*v0 + w1*v1 + w2*v2
        pos = W[i, 0] * v0 + W[i, 1] * v1 + W[i, 2] * v2
        color = blend_color(W[i])

        circle = mpatches.Circle(pos, 0.025, facecolor=color,
                                  edgecolor='white', lw=0.8, zorder=3)
        ax.add_patch(circle)

        # Optional item labels
        if item_labels and i < len(item_labels):
            ax.text(pos[0] + 0.035, pos[1], item_labels[i], fontsize=5,
                    ha='left', va='center', color=CHARCOAL, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_parallel_coords(ax, W, item_labels=None):
    """Parallel coordinates - each factor is an axis, items are lines."""
    n, k = W.shape

    # Draw vertical axes
    axis_x = [0.15, 0.5, 0.85]
    for j, x in enumerate(axis_x):
        ax.plot([x, x], [0.1, 0.9], color=FACTOR_HEX[j], lw=2, alpha=0.6)
        ax.text(x, 0.95, ['animate', 'round', 'natural'][j], fontsize=7,
                ha='center', color=FACTOR_HEX[j], fontweight='medium', style='italic')
        # Tick marks
        for tick in [0, 0.5, 1.0]:
            y = 0.1 + tick * 0.8
            ax.plot([x - 0.02, x + 0.02], [y, y], color=FACTOR_HEX[j], lw=1)
            if j == 0:
                ax.text(x - 0.05, y, f'{tick:.1f}', fontsize=5, ha='right',
                        va='center', color=MEDIUM_GRAY)

    # Draw lines for each item
    for i in range(n):
        color = blend_color(W[i])
        y_vals = 0.1 + W[i] * 0.8

        # Draw line through all axes
        ax.plot(axis_x, y_vals, color=color, lw=1.2, alpha=0.7, zorder=2)

        # Small circle at each axis intersection
        for j, x in enumerate(axis_x):
            ax.scatter(x, y_vals[j], c=[color], s=15, edgecolor='white',
                      linewidth=0.4, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_bipartite(ax, W, item_labels=None):
    """Bipartite graph - items on left, factors on right, edges weighted by loadings."""
    n, k = W.shape

    # Item positions (left side)
    item_y = np.linspace(0.9, 0.1, n)
    item_x = 0.15

    # Factor positions (right side)
    factor_y = [0.75, 0.5, 0.25]
    factor_x = 0.85

    # Draw edges (only significant ones)
    for i in range(n):
        for j in range(k):
            if W[i, j] > 0.1:  # threshold for visibility
                alpha = W[i, j] * 0.8
                lw = 0.5 + W[i, j] * 2
                ax.plot([item_x + 0.03, factor_x - 0.06],
                       [item_y[i], factor_y[j]],
                       color=FACTOR_HEX[j], alpha=alpha, lw=lw, zorder=1)

    # Draw item nodes
    for i in range(n):
        color = blend_color(W[i])
        circle = mpatches.Circle((item_x, item_y[i]), 0.022, facecolor=color,
                                  edgecolor='white', lw=0.6, zorder=3)
        ax.add_patch(circle)

        if item_labels and i < len(item_labels):
            ax.text(item_x - 0.05, item_y[i], item_labels[i], fontsize=4.5,
                    ha='right', va='center', color=CHARCOAL)

    # Draw factor nodes
    factor_labels = ['animate', 'round', 'natural']
    for j in range(k):
        circle = mpatches.Circle((factor_x, factor_y[j]), 0.045,
                                  facecolor=FACTOR_HEX[j], edgecolor='white', lw=1, zorder=3)
        ax.add_patch(circle)
        ax.text(factor_x + 0.08, factor_y[j], factor_labels[j], fontsize=7,
                ha='left', va='center', color=FACTOR_HEX[j], fontweight='medium', style='italic')

    # Labels
    ax.text(item_x, 0.02, 'items', fontsize=7, ha='center', color=MEDIUM_GRAY)
    ax.text(factor_x, 0.02, 'factors', fontsize=7, ha='center', color=MEDIUM_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_factor_bars(ax, W, item_labels=None):
    """Stacked bar chart - each item is a horizontal bar showing factor composition."""
    n, k = W.shape
    bar_height = 0.75 / n

    for i in range(n):
        y = 0.9 - (i + 1) * bar_height
        x_start = 0.25

        # Stacked bars
        for j in range(k):
            width = W[i, j] * 0.55
            rect = mpatches.Rectangle((x_start, y + 0.1 * bar_height),
                                       width, bar_height * 0.8,
                                       facecolor=FACTOR_HEX[j], edgecolor='none')
            ax.add_patch(rect)
            x_start += width

        # Item label
        if item_labels and i < len(item_labels):
            ax.text(0.22, y + bar_height * 0.5, item_labels[i], fontsize=4.5,
                    ha='right', va='center', color=CHARCOAL)

        # Colored dot
        color = blend_color(W[i])
        circle = mpatches.Circle((0.1, y + bar_height * 0.5), 0.018,
                                  facecolor=color, edgecolor='white', lw=0.5)
        ax.add_patch(circle)

    # Factor legend at bottom
    legend_x = [0.35, 0.55, 0.75]
    for j, x in enumerate(legend_x):
        rect = mpatches.Rectangle((x, 0.02), 0.08, 0.04,
                                   facecolor=FACTOR_HEX[j], edgecolor='none')
        ax.add_patch(rect)
        ax.text(x + 0.04, 0.08, ['animate', 'round', 'natural'][j], fontsize=5,
                ha='center', color=FACTOR_HEX[j], style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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

    # 1. Ternary plot
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    draw_ternary(ax1, W_ord, item_labels_ord)
    fig1.text(0.5, 0.95, 'Ternary Plot (Simplex)', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig1.savefig(output_dir / "w_ternary.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig1)
    print("Saved: w_ternary.svg")

    # 2. Parallel coordinates
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    draw_parallel_coords(ax2, W_ord, item_labels_ord)
    fig2.text(0.5, 0.98, 'Parallel Coordinates', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig2.savefig(output_dir / "w_parallel.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)
    print("Saved: w_parallel.svg")

    # 3. Bipartite graph
    fig3, ax3 = plt.subplots(figsize=(4.5, 5))
    draw_bipartite(ax3, W_ord, item_labels_ord)
    fig3.text(0.5, 0.96, 'Bipartite Graph', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig3.savefig(output_dir / "w_bipartite.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig3)
    print("Saved: w_bipartite.svg")

    # 4. Stacked bars
    fig4, ax4 = plt.subplots(figsize=(4, 5))
    draw_factor_bars(ax4, W_ord, item_labels_ord)
    fig4.text(0.5, 0.96, 'Factor Composition Bars', fontsize=10, ha='center',
              color=CHARCOAL, fontweight='medium')
    fig4.savefig(output_dir / "w_bars.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig4)
    print("Saved: w_bars.svg")

    # 5. Combined comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    draw_ternary(axes[0, 0], W_ord, item_labels_ord)
    axes[0, 0].set_title('Ternary (Simplex)', fontsize=9, color=CHARCOAL, pad=5)

    draw_parallel_coords(axes[0, 1], W_ord)
    axes[0, 1].set_title('Parallel Coordinates', fontsize=9, color=CHARCOAL, pad=5)

    draw_bipartite(axes[1, 0], W_ord, item_labels_ord)
    axes[1, 0].set_title('Bipartite Graph', fontsize=9, color=CHARCOAL, pad=5)

    draw_factor_bars(axes[1, 1], W_ord, item_labels_ord)
    axes[1, 1].set_title('Factor Composition', fontsize=9, color=CHARCOAL, pad=5)

    fig.suptitle('Alternative Visualizations for Factor Loadings (W)',
                 fontsize=11, color=CHARCOAL, fontweight='medium', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "w_alternatives_comparison.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: w_alternatives_comparison.svg")


if __name__ == "__main__":
    main()
