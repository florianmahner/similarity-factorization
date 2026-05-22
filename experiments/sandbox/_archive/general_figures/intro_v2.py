"""Intro figure v2 - Source → RSM → Graph → Interpretable Space."""

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

BRAIN_COLOR = '#7eb5d6'
DNN_COLOR = '#9b7eb5'
BEHAVIOR_COLOR = '#7eb58a'


def create_colormap():
    colors = ['#ffffff', '#f0f4f8', '#d4e1ed', '#a8c5db',
              '#6a9fc0', '#3d7ea6', '#1a5a7a', '#0d3d54']
    return mcolors.LinearSegmentedColormap.from_list('elegant', colors)


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
    S = W @ W.T
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((-W.max(axis=1), dominant))
    return W, S, order


def draw_source_icons(ax):
    """Draw brain, DNN, behavior icons stacked."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Brain
    brain = mpatches.Ellipse((0.5, 0.78), 0.35, 0.28, facecolor=BRAIN_COLOR, edgecolor='white', lw=1)
    ax.add_patch(brain)
    ax.text(0.5, 0.78, 'Brain', fontsize=7, ha='center', va='center', color='white', fontweight='medium')

    # DNN
    dnn = mpatches.FancyBboxPatch((0.25, 0.42), 0.5, 0.22, boxstyle="round,pad=0.02",
                                   facecolor=DNN_COLOR, edgecolor='white', lw=1)
    ax.add_patch(dnn)
    ax.text(0.5, 0.53, 'DNN', fontsize=7, ha='center', va='center', color='white', fontweight='medium')

    # Behavior
    behavior = mpatches.FancyBboxPatch((0.25, 0.12), 0.5, 0.22, boxstyle="round,pad=0.02",
                                        facecolor=BEHAVIOR_COLOR, edgecolor='white', lw=1)
    ax.add_patch(behavior)
    ax.text(0.5, 0.23, 'Behavior', fontsize=7, ha='center', va='center', color='white', fontweight='medium')


def draw_rsm(ax, S, cmap):
    """Draw similarity matrix."""
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


def draw_graph(ax, W, S, seed=42):
    """Draw graph with soft cluster coloring."""
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
    factor_centers = {0: np.array([-0.5, -0.3]), 1: np.array([0.5, -0.3]), 2: np.array([0.0, 0.55])}
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
                color=LIGHT_GRAY, lw=0.4 + 0.5 * strength,
                alpha=0.15 + 0.5 * strength, zorder=1, clip_on=False)

    # Nodes
    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.07, facecolor=color,
                                  edgecolor='white', lw=0.8, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.7, 0.9)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_interpretable_space(ax, W):
    """Show W matrix as interpretable factors with labels."""
    n, k = W.shape
    cell_h = 0.82 / n
    cell_w = 0.22

    # Factor labels (what we learn they represent)
    factor_labels = ['animate', 'round', 'natural']

    # Item labels (example items)
    item_labels = ['dog', 'cat', 'bird', 'fish',
                   'ball', 'orange', 'apple', 'wheel',
                   'tree', 'rock', 'leaf', 'cloud',
                   'turtle', 'egg', 'snail']

    # Draw W matrix
    for i in range(n):
        y = 0.92 - (i + 1) * cell_h
        for j in range(k):
            val = W[i, j]
            rect = mpatches.Rectangle(
                (0.28 + j * cell_w, y), cell_w * 0.9, cell_h * 0.9,
                facecolor=FACTOR_HEX[j], alpha=val, edgecolor='none')
            ax.add_patch(rect)

        # Item label on left
        if i < len(item_labels):
            ax.text(0.25, y + cell_h * 0.45, item_labels[i],
                    fontsize=5, ha='right', va='center', color=CHARCOAL)

        # Colored dot showing blend
        color = blend_color(W[i])
        circle = mpatches.Circle((0.08, y + cell_h * 0.45), 0.022,
                                  facecolor=color, edgecolor='white', lw=0.5)
        ax.add_patch(circle)

    # Factor labels at top
    for j in range(k):
        ax.text(0.28 + (j + 0.5) * cell_w, 0.96, factor_labels[j],
                fontsize=7, ha='center', color=FACTOR_HEX[j], fontweight='medium', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def main():
    W, S, order = create_data()
    W_ord = W[order]
    S_ord = S[np.ix_(order, order)]
    cmap = create_colormap()

    fig = plt.figure(figsize=(11, 3.5))

    y_main = 0.15
    h_main = 0.72

    # 1. Source icons
    ax1 = fig.add_axes([0.02, y_main, 0.12, h_main])
    draw_source_icons(ax1)
    fig.text(0.08, 0.06, 'Source', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.155, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 2. RSM
    ax2 = fig.add_axes([0.18, y_main + 0.05, 0.17, h_main - 0.05])
    draw_rsm(ax2, S_ord, cmap)
    fig.text(0.265, 0.06, 'Similarity (RSM)', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.375, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 3. Graph
    ax3 = fig.add_axes([0.40, y_main, 0.22, h_main])
    draw_graph(ax3, W_ord, S_ord)
    fig.text(0.51, 0.06, 'Graph', fontsize=9, ha='center', color=CHARCOAL)

    # Arrow
    fig.text(0.635, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 4. Interpretable factors (W matrix with labels)
    ax4 = fig.add_axes([0.67, y_main, 0.31, h_main])
    draw_interpretable_space(ax4, W_ord)
    fig.text(0.825, 0.06, 'Interpretable Factors', fontsize=9, ha='center', color=CHARCOAL)

    # Title
    fig.text(0.5, 0.95, 'From Representations to Interpretable Dimensions',
             fontsize=11, ha='center', color=CHARCOAL, fontweight='medium')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "intro_v2.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: intro_v2.svg")


if __name__ == "__main__":
    main()
