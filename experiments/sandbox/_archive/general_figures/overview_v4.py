"""Overview v4 - Elegant graph-to-embedding with side-by-side comparison."""

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


def draw_input_graph(ax, W, S, seed=42):
    """Input graph with uniform gray nodes."""
    n = len(W)
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    # Random layout (before we know structure)
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=seed + 10)

    # Edges
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, linewidth=0.4 + 0.5 * strength,
                alpha=0.2 + 0.4 * strength, zorder=1, clip_on=False)

    # Nodes - all same color (we don't know clusters yet)
    for i in range(n):
        x, y = pos[i]
        circle = mpatches.Circle((x, y), 0.08, facecolor='#8899aa',
                                  edgecolor='white', linewidth=0.8, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    return pos


def draw_output_graph(ax, W, S, seed=42):
    """Output graph with colored nodes arranged by cluster."""
    n = len(W)
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    threshold = 0.12
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    # Cluster-aware layout
    factor_centers = {0: np.array([-0.55, -0.35]), 1: np.array([0.55, -0.35]), 2: np.array([0.0, 0.6])}
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.1, 0.1, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=80, seed=seed)

    # Edges
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, linewidth=0.4 + 0.5 * strength,
                alpha=0.15 + 0.5 * strength, zorder=1, clip_on=False)

    # Nodes with blended colors
    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.08, facecolor=color,
                                  edgecolor='white', linewidth=0.8, zorder=3, clip_on=False)
        ax.add_patch(circle)

    # Factor labels
    for (cx, cy), label, color in [
        ((-0.8, -0.7), 'F₁', FACTOR_HEX[0]),
        ((0.8, -0.7), 'F₂', FACTOR_HEX[1]),
        ((0.0, 1.0), 'F₃', FACTOR_HEX[2]),
    ]:
        ax.text(cx, cy, label, fontsize=10, ha='center', va='center',
                color=color, fontweight='medium', clip_on=False)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_w_horizontal(ax, W_ordered):
    """Horizontal W visualization with stacked bars."""
    n, k = W_ordered.shape

    bar_height = 0.8 / n
    for i in range(n):
        y = 0.9 - (i + 0.5) * bar_height - 0.05

        # Stacked bar
        x_start = 0.15
        for j in range(k):
            width = W_ordered[i, j] * 0.7
            rect = mpatches.Rectangle((x_start, y - bar_height * 0.4), width, bar_height * 0.8,
                                       facecolor=FACTOR_HEX[j], edgecolor='none')
            ax.add_patch(rect)
            x_start += width

        # Node dot on left
        color = blend_color(W_ordered[i])
        circle = mpatches.Circle((0.08, y), 0.015, facecolor=color,
                                  edgecolor='white', linewidth=0.3, zorder=3)
        ax.add_patch(circle)

    # Legend
    for j, (color, label) in enumerate(zip(FACTOR_HEX, ['F₁', 'F₂', 'F₃'])):
        rect = mpatches.Rectangle((0.2 + j * 0.25, 0.02), 0.08, 0.04,
                                   facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(0.3 + j * 0.25, 0.04, label, fontsize=7, va='center', color=CHARCOAL)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    W, S, order = create_data()
    W_ordered = W[order]

    fig = plt.figure(figsize=(11, 4))

    # Title
    fig.text(0.5, 0.96, 'Similarity-based Representation Factorization', fontsize=11,
             ha='center', color=CHARCOAL, fontweight='medium')

    # Input graph
    ax1 = fig.add_axes([0.02, 0.08, 0.28, 0.82])
    draw_input_graph(ax1, W, S)
    fig.text(0.16, 0.02, 'Input: Graph / Similarity', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # Arrow with SRF label
    fig.text(0.33, 0.48, '→', fontsize=20, color=MEDIUM_GRAY, ha='center', va='center')
    fig.text(0.33, 0.38, 'SRF', fontsize=9, ha='center', color=FACTOR_HEX[1], fontweight='medium')

    # W matrix in middle
    ax2 = fig.add_axes([0.37, 0.10, 0.22, 0.78])
    draw_w_horizontal(ax2, W_ordered)
    fig.text(0.48, 0.02, 'Embedding W (soft membership)', fontsize=9, ha='center', color=MEDIUM_GRAY)

    # Arrow
    fig.text(0.62, 0.48, '→', fontsize=20, color=MEDIUM_GRAY, ha='center', va='center')

    # Output graph with clusters
    ax3 = fig.add_axes([0.67, 0.08, 0.32, 0.82])
    draw_output_graph(ax3, W, S)
    fig.text(0.83, 0.02, 'Output: Soft Clusters', fontsize=9, ha='center', color=MEDIUM_GRAY)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v4.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: overview_v4.svg")


if __name__ == "__main__":
    main()
