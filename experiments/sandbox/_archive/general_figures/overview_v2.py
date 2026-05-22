"""Overview figure v2 - using consistent style from figure_rsm and figure_graph."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

# Colors matching figure_rsm and figure_graph
FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],  # orange-coral (#E07040)
    [0.31, 0.44, 0.75],  # blue-indigo (#5070C0)
    [0.44, 0.69, 0.25],  # yellow-green (#70B040)
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']


def create_data(seed=42):
    """Same data generation as figure_rsm/figure_graph."""
    rng = np.random.default_rng(seed)
    W_list = []

    for _ in range(4):
        W_list.append([rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92), rng.uniform(0.03, 0.09)])
    for _ in range(4):
        W_list.append([rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09), rng.uniform(0.82, 0.92)])

    # Overlap items
    W_list.append([0.46, 0.46, 0.08])  # F1-F2
    W_list.append([0.47, 0.07, 0.46])  # F1-F3
    W_list.append([0.07, 0.47, 0.46])  # F2-F3

    W = np.array(W_list)
    W = W / W.sum(axis=1, keepdims=True)
    S = W @ W.T

    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    return W, S, order


def blend_color(weights):
    """Blend factor colors based on membership weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    rgb = weights @ FACTOR_COLORS
    return rgb


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_graph(ax, W, S, seed=42):
    """Draw graph with blended node colors."""
    n = len(W)
    rng = np.random.default_rng(seed)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    # Layout
    factor_centers = {
        0: np.array([-0.5, -0.3]),
        1: np.array([0.5, -0.3]),
        2: np.array([0.0, 0.5]),
    }
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        offset = rng.uniform(-0.1, 0.1, 2)
        pos[i] = center + offset
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=60, seed=seed)

    # Draw edges
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        alpha = 0.15 + 0.5 * strength
        lw = 0.3 + 0.5 * strength
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, linewidth=lw, alpha=alpha, zorder=1, clip_on=False)

    # Draw nodes
    node_radius = 0.08
    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), node_radius, facecolor=color,
                                  edgecolor='white', linewidth=0.8, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.7, 0.85)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_rsm_mini(ax, S_ordered, W_ordered, cmap):
    """Draw small RSM with membership bars."""
    n = len(S_ordered)
    cell_size = 1.0 / n

    # RSM cells
    for i in range(n):
        for j in range(n):
            color = cmap(S_ordered[i, j])
            rect = mpatches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor='none'
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_membership_bar_left(ax, W_ordered):
    """Left membership bar."""
    n = len(W_ordered)
    cell_size = 1.0 / n
    for i in range(n):
        for k in range(3):
            if W_ordered[i, k] > 0.05:
                rect = mpatches.Rectangle(
                    (0, 1 - (i + 1) * cell_size), 1, cell_size,
                    facecolor=FACTOR_HEX[k], alpha=W_ordered[i, k], edgecolor='none'
                )
                ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_w_matrix(ax, W_ordered):
    """Draw W matrix with colored columns."""
    n, k = W_ordered.shape
    cell_h = 0.9 / n
    cell_w = 0.8 / k

    for i in range(n):
        for j in range(k):
            val = W_ordered[i, j]
            rect = mpatches.Rectangle(
                (0.1 + j * cell_w, 0.95 - (i + 1) * cell_h),
                cell_w * 0.9, cell_h * 0.9,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none'
            )
            ax.add_patch(rect)

    # Column labels
    for j in range(k):
        ax.text(0.1 + (j + 0.5) * cell_w, 0.02, f'F{j+1}',
                fontsize=6, ha='center', color=FACTOR_HEX[j])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_soft_clusters(ax, W, seed=42):
    """Draw nodes in embedding space showing soft cluster membership."""
    n = len(W)
    rng = np.random.default_rng(seed)

    # Position nodes based on W (using barycentric coordinates)
    corners = np.array([
        [0.5, 0.92],   # F1 top
        [0.12, 0.15],  # F2 bottom-left
        [0.88, 0.15],  # F3 bottom-right
    ])

    for i in range(n):
        w = W[i]
        pos = w @ corners
        # Add small jitter
        pos += rng.uniform(-0.04, 0.04, 2)

        color = blend_color(w)
        circle = mpatches.Circle(pos, 0.045, facecolor=color,
                                 edgecolor='white', linewidth=0.6, zorder=3, clip_on=False)
        ax.add_patch(circle)

    # Factor labels at corners
    ax.text(0.5, 0.98, 'F₁', fontsize=8, ha='center', color=FACTOR_HEX[0], fontweight='medium')
    ax.text(0.05, 0.10, 'F₂', fontsize=8, ha='center', color=FACTOR_HEX[1], fontweight='medium')
    ax.text(0.95, 0.10, 'F₃', fontsize=8, ha='center', color=FACTOR_HEX[2], fontweight='medium')

    # Light triangle outline
    triangle = plt.Polygon(corners, fill=False, edgecolor=LIGHT_GRAY, lw=0.8)
    ax.add_patch(triangle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_arrow(fig, x, y, direction='right'):
    """Draw arrow between panels."""
    if direction == 'right':
        fig.text(x, y, '→', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')
    else:
        fig.text(x, y, '↓', fontsize=14, color=MEDIUM_GRAY, ha='center', va='center')


def main():
    W, S, order = create_data()
    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]
    cmap = create_colormap()

    # Create figure
    fig = plt.figure(figsize=(11, 3.2))

    # Panel positions
    y_main = 0.18
    h_main = 0.72

    # 1. Graph
    ax1 = fig.add_axes([0.02, y_main, 0.18, h_main])
    draw_graph(ax1, W, S)
    fig.text(0.11, 0.05, 'Graph', fontsize=9, ha='center', color=MEDIUM_GRAY)

    draw_arrow(fig, 0.215, 0.54)

    # 2. RSM with left bar
    rsm_left = 0.26
    rsm_size = 0.12
    bar_w = 0.015

    ax_bar = fig.add_axes([rsm_left - bar_w - 0.005, y_main, bar_w, h_main * rsm_size / 0.12])
    draw_membership_bar_left(ax_bar, W_ordered)

    ax2 = fig.add_axes([rsm_left, y_main, rsm_size, h_main * rsm_size / 0.12])
    draw_rsm_mini(ax2, S_ordered, W_ordered, cmap)
    fig.text(rsm_left + rsm_size / 2, 0.05, 'Similarity', fontsize=9, ha='center', color=MEDIUM_GRAY)

    draw_arrow(fig, 0.42, 0.54)

    # 3. SRF box with equation
    ax3 = fig.add_axes([0.45, y_main + 0.1, 0.10, h_main - 0.2])
    box = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.03",
                                   facecolor='#f0f7fc', edgecolor=FACTOR_HEX[1], lw=1.2)
    ax3.add_patch(box)
    ax3.text(0.5, 0.55, 'S ≈ WWᵀ', fontsize=8, ha='center', va='center', color=CHARCOAL)
    ax3.text(0.5, 0.35, 'W ≥ 0', fontsize=7, ha='center', va='center', color=MEDIUM_GRAY)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    fig.text(0.50, 0.05, 'SRF', fontsize=9, ha='center', color=MEDIUM_GRAY)

    draw_arrow(fig, 0.575, 0.54)

    # 4. W matrix
    ax4 = fig.add_axes([0.60, y_main, 0.08, h_main])
    draw_w_matrix(ax4, W_ordered)
    fig.text(0.64, 0.05, 'Embedding W', fontsize=9, ha='center', color=MEDIUM_GRAY)

    draw_arrow(fig, 0.705, 0.54)

    # 5. Soft cluster visualization
    ax5 = fig.add_axes([0.74, y_main - 0.02, 0.24, h_main + 0.04])
    draw_soft_clusters(ax5, W)
    fig.text(0.86, 0.05, 'Soft Clusters', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v2.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: overview_v2.svg")


if __name__ == "__main__":
    main()
