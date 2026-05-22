"""Overview v5 - Simple: Input → Transform → Output (single clean row)."""

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
    # Normalize so diagonal is 1
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    return W, S


def blend_color(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return weights @ FACTOR_COLORS


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


def draw_similarity_matrix(ax, S, cmap):
    """Draw similarity matrix as input."""
    n = len(S)
    cell = 1.0 / n
    for i in range(n):
        for j in range(n):
            rect = mpatches.Rectangle(
                (j * cell, 1 - (i + 1) * cell), cell, cell,
                facecolor=cmap(S[i, j]), edgecolor='none')
            ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_matrix(ax, W):
    """Draw W matrix with nodes on left, factors as columns."""
    n, k = W.shape
    cell_h = 0.9 / n
    cell_w = 0.20

    # W matrix cells
    for i in range(n):
        y = 0.95 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.40 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        # Node circle on left
        color = blend_color(W[i])
        circle = mpatches.Circle((0.18, y + cell_h * 0.45), 0.035,
                                  facecolor=color, edgecolor='white', lw=0.6)
        ax.add_patch(circle)

        # Connection line
        ax.plot([0.22, 0.38], [y + cell_h * 0.45, y + cell_h * 0.45],
                color=LIGHT_GRAY, lw=0.4, alpha=0.5)

    # Labels
    ax.text(0.18, 0.02, 'nodes', fontsize=7, ha='center', color=MEDIUM_GRAY)
    for j in range(k):
        ax.text(0.40 + (j + 0.5) * cell_w, 0.02, f'F{j+1}',
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_graph_clustered(ax, W, S, seed=42):
    """Draw graph with nodes colored by soft membership."""
    n = len(W)
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    # Layout by cluster
    factor_centers = {0: np.array([-0.5, -0.3]), 1: np.array([0.5, -0.3]), 2: np.array([0.0, 0.5])}
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.1, 0.1, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=60, seed=seed)

    # Edges
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, lw=0.4 + 0.6 * strength,
                alpha=0.15 + 0.5 * strength, zorder=1, clip_on=False)

    # Nodes
    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.08, facecolor=color,
                                  edgecolor='white', lw=1, zorder=3, clip_on=False)
        ax.add_patch(circle)

    # Factor labels
    for (cx, cy), label, col in [
        ((-0.75, -0.6), 'F₁', FACTOR_HEX[0]),
        ((0.75, -0.6), 'F₂', FACTOR_HEX[1]),
        ((0.0, 0.85), 'F₃', FACTOR_HEX[2]),
    ]:
        ax.text(cx, cy, label, fontsize=9, ha='center', color=col, fontweight='medium')

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.8, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, S = create_data()
    cmap = create_colormap()

    n = len(W)

    # Order for W display (sorted by factor for clarity)
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    W_ord = W[order]

    # Scrambled permutation - breaks visible block structure
    # Simulates arbitrary ordering (e.g., alphabetical) where structure isn't apparent
    rng = np.random.default_rng(123)
    scramble = rng.permutation(n)
    S_scrambled = S[np.ix_(scramble, scramble)]

    fig = plt.figure(figsize=(11, 3.5))

    # Input: Similarity (scrambled - no visible block structure)
    ax1 = fig.add_axes([0.03, 0.15, 0.22, 0.75])
    draw_similarity_matrix(ax1, S_scrambled, cmap)
    fig.text(0.14, 0.05, 'Similarity S', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow + equation
    fig.text(0.30, 0.52, '→', fontsize=20, color=MEDIUM_GRAY)
    ax_eq = fig.add_axes([0.34, 0.30, 0.14, 0.45])
    box = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=1.5)
    ax_eq.add_patch(box)
    ax_eq.text(0.5, 0.6, 'S ≈ WWᵀ', fontsize=11, ha='center', va='center', color=CHARCOAL)
    ax_eq.text(0.5, 0.35, 'W ≥ 0', fontsize=9, ha='center', va='center', color=MEDIUM_GRAY)
    ax_eq.set_xlim(0, 1)
    ax_eq.set_ylim(0, 1)
    ax_eq.axis('off')
    fig.text(0.41, 0.05, 'SRF', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.505, 0.52, '→', fontsize=20, color=MEDIUM_GRAY)

    # W matrix
    ax2 = fig.add_axes([0.53, 0.12, 0.20, 0.78])
    draw_w_matrix(ax2, W_ord)
    fig.text(0.63, 0.05, 'Embedding W', fontsize=10, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.755, 0.52, '→', fontsize=20, color=MEDIUM_GRAY)

    # Output: Clustered graph
    ax3 = fig.add_axes([0.78, 0.10, 0.21, 0.82])
    draw_graph_clustered(ax3, W, S)
    fig.text(0.885, 0.05, 'Soft Clusters', fontsize=10, ha='center', color=CHARCOAL)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v5.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: overview_v5.svg")


if __name__ == "__main__":
    main()
