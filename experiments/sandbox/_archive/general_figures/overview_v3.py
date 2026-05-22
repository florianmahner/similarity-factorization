"""Overview v3 - Full two-stage pipeline: Embedding + Consensus."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'

FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],
    [0.31, 0.44, 0.75],
    [0.44, 0.69, 0.25],
])
FACTOR_HEX = ['#E07040', '#5070C0', '#70B040']


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

    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))

    return W, S, order


def blend_color(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ FACTOR_COLORS


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_graph_compact(ax, W, S, seed=42):
    """Compact graph view."""
    n = len(W)
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    factor_centers = {0: np.array([-0.45, -0.25]), 1: np.array([0.45, -0.25]), 2: np.array([0.0, 0.45])}
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.08, 0.08, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.4, iterations=50, seed=seed)

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, linewidth=0.3 + 0.4 * strength,
                alpha=0.15 + 0.45 * strength, zorder=1, clip_on=False)

    for i in range(n):
        x, y = pos[i]
        circle = mpatches.Circle((x, y), 0.065, facecolor=blend_color(W[i]),
                                  edgecolor='white', linewidth=0.6, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.6, 0.75)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_matrix_detailed(ax, W_ordered):
    """Detailed W matrix with row connections to nodes."""
    n, k = W_ordered.shape
    cell_h = 0.85 / n
    cell_w = 0.25

    for i in range(n):
        for j in range(k):
            val = W_ordered[i, j]
            rect = mpatches.Rectangle(
                (0.35 + j * cell_w, 0.92 - (i + 1) * cell_h),
                cell_w * 0.85, cell_h * 0.85,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none'
            )
            ax.add_patch(rect)

        y_pos = 0.92 - (i + 0.5) * cell_h
        color = blend_color(W_ordered[i])
        circle = mpatches.Circle((0.15, y_pos), 0.025, facecolor=color,
                                  edgecolor='white', linewidth=0.5, zorder=3)
        ax.add_patch(circle)
        ax.plot([0.18, 0.33], [y_pos, y_pos], color=LIGHT_GRAY, lw=0.5, alpha=0.5)

    for j in range(k):
        ax.text(0.35 + (j + 0.45) * cell_w, 0.02, f'F{j+1}',
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_rsm_mini(ax, S_ordered, cmap):
    n = len(S_ordered)
    cell_size = 1.0 / n
    for i in range(n):
        for j in range(n):
            rect = mpatches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size), cell_size, cell_size,
                facecolor=cmap(S_ordered[i, j]), edgecolor='none')
            ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_cv_curves(ax):
    """Cross-validation curves for rank selection."""
    x = np.linspace(0.08, 0.92, 50)
    train = 0.82 * np.exp(-4 * x) + 0.08
    val = 0.65 * np.exp(-2.5 * x) + 0.12 + 0.22 * (x - 0.38) ** 2

    ax.plot(x, train, color=CHARCOAL, lw=1.2, label='Train')
    ax.plot(x, val, color=FACTOR_HEX[0], lw=1.2, label='Val')

    k_star = 0.38
    ax.axvline(k_star, color=LIGHT_GRAY, ls='--', lw=0.8)
    ax.scatter([k_star], [np.interp(k_star, x, val)], color=FACTOR_HEX[0], s=25, zorder=5)
    ax.text(k_star + 0.05, 0.12, 'k*', fontsize=7, color=MEDIUM_GRAY, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_stacked_matrices(ax, n_stacks=5):
    """Multiple W matrices stacked (repeats)."""
    for i in range(n_stacks):
        offset = i * 0.06
        alpha = 0.3 if i < n_stacks - 1 else 1.0
        rect = mpatches.Rectangle(
            (0.15 + offset, 0.12 + offset), 0.45, 0.65,
            facecolor='white' if i < n_stacks - 1 else '#f5f5f5',
            edgecolor=LIGHT_GRAY if i < n_stacks - 1 else CHARCOAL,
            lw=0.8 if i < n_stacks - 1 else 1.2,
            alpha=alpha
        )
        ax.add_patch(rect)

    ax.text(0.55, 0.05, 'repeats', fontsize=7, color=MEDIUM_GRAY, ha='center', style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_dimension_clustering(ax, seed=42):
    """Clustering of embedding dimensions."""
    rng = np.random.default_rng(seed)

    # 3 clusters of "dimensions"
    cluster_centers = [(0.25, 0.7), (0.7, 0.65), (0.5, 0.25)]

    for (cx, cy), col in zip(cluster_centers, FACTOR_HEX):
        # Draw cluster blob outline
        theta = np.linspace(0, 2 * np.pi, 50)
        r = 0.15 + 0.03 * np.sin(5 * theta)
        blob_x = cx + r * np.cos(theta)
        blob_y = cy + r * np.sin(theta) * 0.7
        ax.fill(blob_x, blob_y, facecolor=col, alpha=0.15, edgecolor=col, lw=1)

        # Points inside
        for _ in range(7):
            px = cx + rng.uniform(-0.1, 0.1)
            py = cy + rng.uniform(-0.08, 0.08)
            ax.scatter(px, py, c=[col], s=12, alpha=0.8, edgecolor='white', linewidth=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_rank_selection(ax):
    """Bar chart for rank selection."""
    ranks = [0.35, 0.85, 0.55, 0.25]
    selected = 1

    for i, h in enumerate(ranks):
        color = FACTOR_HEX[1] if i == selected else LIGHT_GRAY
        rect = mpatches.Rectangle(
            (0.15 + i * 0.2, 0.2), 0.14, h * 0.6,
            facecolor=color, edgecolor='none'
        )
        ax.add_patch(rect)

    ax.text(0.5, 0.08, 'k', fontsize=8, ha='center', color=MEDIUM_GRAY, style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_consensus_w(ax, W_ordered):
    """Final consensus W matrix."""
    n, k = W_ordered.shape
    cell_h = 0.75 / n
    cell_w = 0.22

    # Box around
    box = mpatches.FancyBboxPatch(
        (0.12, 0.08), 0.76, 0.84,
        boxstyle="round,pad=0.02",
        facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=1.5
    )
    ax.add_patch(box)

    for i in range(n):
        for j in range(k):
            val = W_ordered[i, j]
            rect = mpatches.Rectangle(
                (0.20 + j * cell_w, 0.85 - (i + 1) * cell_h),
                cell_w * 0.85, cell_h * 0.85,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none'
            )
            ax.add_patch(rect)

    ax.text(0.5, 0.95, 'W*', fontsize=9, ha='center', color=CHARCOAL, fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    W, S, order = create_data()
    W_ordered = W[order]
    S_ordered = S[np.ix_(order, order)]
    cmap = create_colormap()

    fig = plt.figure(figsize=(12, 6))

    # Stage labels
    fig.text(0.02, 0.95, 'Stage 1: Embedding', fontsize=10, color=CHARCOAL, fontweight='bold')
    fig.text(0.02, 0.47, 'Stage 2: Consensus', fontsize=10, color=CHARCOAL, fontweight='bold')

    # Dividing line
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.50, 0.50], color=LIGHT_GRAY, lw=1, linestyle='-'))

    # === STAGE 1 ===
    y1 = 0.56
    h1 = 0.38

    # 1. Graph
    ax1 = fig.add_axes([0.02, y1, 0.16, h1])
    draw_graph_compact(ax1, W, S)
    fig.text(0.10, y1 - 0.04, 'Graph', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.185, y1 + h1 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 2. RSM
    ax2 = fig.add_axes([0.21, y1 + 0.02, 0.11, h1 - 0.04])
    draw_rsm_mini(ax2, S_ordered, cmap)
    fig.text(0.265, y1 - 0.04, 'Similarity', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.34, y1 + h1 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 3. Cross-validation
    ax3 = fig.add_axes([0.37, y1 + 0.04, 0.12, h1 - 0.08])
    draw_cv_curves(ax3)
    fig.text(0.43, y1 - 0.04, 'Rank Selection', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.505, y1 + h1 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 4. SRF box
    ax4 = fig.add_axes([0.53, y1 + 0.06, 0.10, h1 - 0.12])
    box = mpatches.FancyBboxPatch((0.08, 0.15), 0.84, 0.7, boxstyle="round,pad=0.04",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=1.5)
    ax4.add_patch(box)
    ax4.text(0.5, 0.55, 'S ≈ WWᵀ', fontsize=9, ha='center', va='center', color=CHARCOAL)
    ax4.text(0.5, 0.38, 'W ≥ 0', fontsize=7, ha='center', va='center', color=MEDIUM_GRAY)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    fig.text(0.58, y1 - 0.04, 'SRF', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.655, y1 + h1 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 5. W matrix
    ax5 = fig.add_axes([0.68, y1, 0.16, h1])
    draw_w_matrix_detailed(ax5, W_ordered)
    fig.text(0.76, y1 - 0.04, 'Embedding W', fontsize=8, ha='center', color=MEDIUM_GRAY)

    # Arrow down to stage 2
    fig.text(0.88, y1 + h1 / 2, '↓', fontsize=16, color=MEDIUM_GRAY, ha='center', rotation=0)
    fig.text(0.92, y1 + h1 / 2 - 0.02, 'repeat', fontsize=7, color=MEDIUM_GRAY, ha='left', style='italic')

    # === STAGE 2 ===
    y2 = 0.06
    h2 = 0.38

    # 1. Stacked matrices
    ax6 = fig.add_axes([0.02, y2, 0.16, h2])
    draw_stacked_matrices(ax6)
    fig.text(0.10, y2 - 0.02, 'Multiple Runs', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.195, y2 + h2 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 2. Dimension clustering
    ax7 = fig.add_axes([0.22, y2, 0.18, h2])
    draw_dimension_clustering(ax7)
    fig.text(0.31, y2 - 0.02, 'Cluster Dimensions', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.42, y2 + h2 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 3. Rank selection
    ax8 = fig.add_axes([0.45, y2 + 0.02, 0.14, h2 - 0.04])
    draw_rank_selection(ax8)
    fig.text(0.52, y2 - 0.02, 'Select Rank', fontsize=8, ha='center', color=MEDIUM_GRAY)

    fig.text(0.61, y2 + h2 / 2, '→', fontsize=14, color=MEDIUM_GRAY, ha='center')

    # 4. Consensus W
    ax9 = fig.add_axes([0.64, y2, 0.16, h2])
    draw_consensus_w(ax9, W_ordered)
    fig.text(0.72, y2 - 0.02, 'Consensus W*', fontsize=8, ha='center', color=MEDIUM_GRAY)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v3.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: overview_v3.svg")


if __name__ == "__main__":
    main()
