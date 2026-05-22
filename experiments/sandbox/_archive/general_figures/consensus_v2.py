"""Consensus visualization v2 - Full process: unstable → cluster dims → select rank → stable."""

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


def draw_unstable_w_matrix(ax, W_true, rng):
    """Unstable W matrix - same structure but with noisy/flickering values."""
    n, k = W_true.shape
    cell_h = 0.85 / n
    cell_w = 0.22

    # Draw multiple ghost layers to show instability
    for _ in range(4):
        W_layer = W_true + rng.normal(0, 0.3, W_true.shape)
        W_layer = np.clip(W_layer, 0.02, 1)
        W_layer = W_layer / W_layer.sum(axis=1, keepdims=True)

        # Different permutation each layer
        perm_layer = rng.permutation(3)
        W_layer = W_layer[:, perm_layer]

        for i in range(n):
            y = 0.92 - (i + 1) * cell_h
            offset_x = rng.uniform(-0.03, 0.03)
            offset_y = rng.uniform(-0.01, 0.01)

            for j in range(k):
                val = W_layer[i, j]
                rect = mpatches.Rectangle(
                    (0.38 + j * cell_w + offset_x, y + offset_y),
                    cell_w * 0.88, cell_h * 0.88,
                    facecolor=FACTOR_HEX[perm_layer[j]], alpha=val * 0.25, edgecolor='none')
                ax.add_patch(rect)

            # Ghost node
            color = blend_color(W_layer[i])
            circle = mpatches.Circle((0.15 + offset_x, y + cell_h * 0.44 + offset_y), 0.038,
                                      facecolor=color, alpha=0.25, edgecolor='none')
            ax.add_patch(circle)

    # Factor labels (wrong/inconsistent)
    ax.text(0.50, 0.02, '?', fontsize=9, ha='center', color=MEDIUM_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_stacked_embeddings(ax, n_stacks=5):
    """Stacked W matrices representing multiple runs."""
    for i in range(n_stacks):
        offset = i * 0.08
        alpha = 0.4 if i < n_stacks - 1 else 1.0
        lw = 0.6 if i < n_stacks - 1 else 1.2
        rect = mpatches.Rectangle(
            (0.1 + offset, 0.15 + offset), 0.5, 0.55,
            facecolor='white' if i < n_stacks - 1 else '#f8f8f8',
            edgecolor=LIGHT_GRAY if i < n_stacks - 1 else CHARCOAL,
            lw=lw, alpha=alpha
        )
        ax.add_patch(rect)

    # Curly brace or label
    ax.text(0.55, 0.06, 'n runs', fontsize=7, ha='center', color=MEDIUM_GRAY, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_dimension_clustering(ax, rng):
    """Dimensions from multiple runs cluster together (k-means)."""
    # 3 clusters representing the true dimensions
    cluster_centers = [(0.22, 0.72), (0.72, 0.68), (0.47, 0.22)]
    n_points_per_cluster = 8  # dimensions from multiple runs

    for k, ((cx, cy), col) in enumerate(zip(cluster_centers, FACTOR_HEX)):
        # Cluster ellipse
        ellipse = mpatches.Ellipse((cx, cy), 0.32, 0.28, angle=rng.uniform(-15, 15),
                                    facecolor=col, alpha=0.12, edgecolor=col, lw=1.2)
        ax.add_patch(ellipse)

        # Points inside (dimensions from different runs)
        for _ in range(n_points_per_cluster):
            px = cx + rng.normal(0, 0.07)
            py = cy + rng.normal(0, 0.06)
            ax.scatter(px, py, c=[col], s=25, alpha=0.75, edgecolor='white', linewidth=0.4)

        # Cluster label
        ax.text(cx, cy - 0.22, f'Dim {k+1}', fontsize=7, ha='center', color=col, fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_rank_selection(ax):
    """Bar chart showing rank selection - choose optimal k."""
    # Stability or quality metric per rank
    ranks = [2, 3, 4, 5]
    values = [0.5, 0.92, 0.75, 0.45]  # rank 3 is best
    selected = 1

    bar_width = 0.15
    for i, (r, v) in enumerate(zip(ranks, values)):
        x = 0.2 + i * 0.2
        is_selected = i == selected
        color = FACTOR_HEX[1] if is_selected else LIGHT_GRAY
        lw = 2 if is_selected else 0

        rect = mpatches.Rectangle(
            (x, 0.25), bar_width, v * 0.55,
            facecolor=color, edgecolor=CHARCOAL if is_selected else 'none', lw=lw
        )
        ax.add_patch(rect)

        # Rank label
        ax.text(x + bar_width/2, 0.18, str(r), fontsize=7, ha='center', color=MEDIUM_GRAY)

    # Axis label
    ax.text(0.5, 0.08, 'k (rank)', fontsize=7, ha='center', color=MEDIUM_GRAY, style='italic')

    # Arrow pointing to selected
    ax.annotate('', xy=(0.2 + selected * 0.2 + bar_width/2, 0.88),
               xytext=(0.2 + selected * 0.2 + bar_width/2, 0.95),
               arrowprops=dict(arrowstyle='->', color=FACTOR_HEX[1], lw=1.5))
    ax.text(0.2 + selected * 0.2 + bar_width/2, 0.97, 'k*', fontsize=8,
            ha='center', color=FACTOR_HEX[1], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_stable_w_matrix(ax, W_true):
    """Final stable consensus - W matrix with nodes linked to rows (matching earlier style)."""
    n, k = W_true.shape
    cell_h = 0.85 / n
    cell_w = 0.22

    # W matrix cells
    for i in range(n):
        y = 0.92 - (i + 1) * cell_h
        for j in range(k):
            val = W_true[i, j]
            rect = mpatches.Rectangle(
                (0.38 + j * cell_w, y),
                cell_w * 0.88, cell_h * 0.88,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        # Node circle on left
        color = blend_color(W_true[i])
        circle = mpatches.Circle((0.15, y + cell_h * 0.44), 0.038,
                                  facecolor=color, edgecolor='white', lw=0.8)
        ax.add_patch(circle)

        # Connection line
        ax.plot([0.19, 0.36], [y + cell_h * 0.44, y + cell_h * 0.44],
                color=LIGHT_GRAY, lw=0.5, alpha=0.6)

    # Factor labels at bottom
    for j in range(k):
        ax.text(0.38 + (j + 0.5) * cell_w, 0.02, f'F{j+1}',
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def create_data(seed=42):
    """Same data as overview_v3 - 15 nodes."""
    rng = np.random.default_rng(seed)
    W_list = []

    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    # Overlap items
    W_list.append([0.46, 0.46, 0.08])
    W_list.append([0.47, 0.07, 0.46])
    W_list.append([0.07, 0.47, 0.46])

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)

    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    return W, order


def main():
    rng = np.random.default_rng(42)

    W_true, order = create_data()
    W_ordered = W_true[order]

    fig = plt.figure(figsize=(12, 3.8))

    # Title
    fig.text(0.5, 0.95, 'Consensus Embedding', fontsize=11, ha='center',
             color=CHARCOAL, fontweight='medium')

    y_main = 0.14
    h_main = 0.72

    # 1. Unstable W matrix (shows variability across runs)
    ax1 = fig.add_axes([0.02, y_main, 0.16, h_main])
    draw_unstable_w_matrix(ax1, W_ordered, rng)
    fig.text(0.10, 0.06, 'Unstable\n(single run)', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.19, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # 2. Stacked embeddings (n runs)
    ax2 = fig.add_axes([0.21, y_main, 0.14, h_main])
    draw_stacked_embeddings(ax2)
    fig.text(0.28, 0.06, 'Multiple\nruns', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.365, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # 3. Dimension clustering (k-means)
    ax3 = fig.add_axes([0.39, y_main, 0.17, h_main])
    draw_dimension_clustering(ax3, rng)
    fig.text(0.475, 0.06, 'Cluster\ndimensions', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.575, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # 4. Rank selection
    ax4 = fig.add_axes([0.59, y_main, 0.13, h_main])
    draw_rank_selection(ax4)
    fig.text(0.655, 0.06, 'Select\nrank', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.735, 0.50, '→', fontsize=16, color=MEDIUM_GRAY, ha='center')

    # 5. Stable consensus (W matrix style matching earlier figures)
    ax5 = fig.add_axes([0.76, y_main, 0.22, h_main])
    draw_stable_w_matrix(ax5, W_ordered)
    fig.text(0.87, 0.06, 'Stable\nconsensus', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Box around final result
    box = mpatches.FancyBboxPatch((0.75, y_main - 0.02), 0.24, h_main + 0.04,
                                   boxstyle="round,pad=0.01", facecolor='none',
                                   edgecolor=FACTOR_HEX[1], lw=1.5, transform=fig.transFigure)
    fig.add_artist(box)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "consensus_v2.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: consensus_v2.svg")


if __name__ == "__main__":
    main()
