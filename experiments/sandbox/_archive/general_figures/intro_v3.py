"""Intro figure v3 - Source → RSM → Graph → Ternary Simplex."""

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

    brain = mpatches.Ellipse((0.5, 0.78), 0.35, 0.28, facecolor=BRAIN_COLOR, edgecolor='white', lw=1)
    ax.add_patch(brain)
    ax.text(0.5, 0.78, 'Brain', fontsize=7, ha='center', va='center', color='white', fontweight='medium')

    dnn = mpatches.FancyBboxPatch((0.25, 0.42), 0.5, 0.22, boxstyle="round,pad=0.02",
                                   facecolor=DNN_COLOR, edgecolor='white', lw=1)
    ax.add_patch(dnn)
    ax.text(0.5, 0.53, 'DNN', fontsize=7, ha='center', va='center', color='white', fontweight='medium')

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

    factor_centers = {0: np.array([-0.5, -0.3]), 1: np.array([0.5, -0.3]), 2: np.array([0.0, 0.55])}
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.1, 0.1, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=60, seed=seed)

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, lw=0.4 + 0.5 * strength,
                alpha=0.15 + 0.5 * strength, zorder=1, clip_on=False)

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


def barycentric_to_cartesian(w, vertices):
    return w[0] * vertices[0] + w[1] * vertices[1] + w[2] * vertices[2]


def draw_ternary_compact(ax, W):
    """Compact ternary plot for the intro figure."""
    n = len(W)

    # Triangle vertices
    h = np.sqrt(3) / 2
    vertices = np.array([
        [0.5, 0.88],   # top
        [0.1, 0.12],   # bottom-left
        [0.9, 0.12],   # bottom-right
    ])

    # Triangle fill
    triangle = mpatches.Polygon(vertices, closed=True, facecolor='#fafafa',
                                 edgecolor='none', zorder=0)
    ax.add_patch(triangle)

    # Subtle grid
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
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#e8e8e8', lw=0.5, zorder=0)

    # Edges
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([vertices[i][0], vertices[j][0]],
                [vertices[i][1], vertices[j][1]],
                color=LIGHT_GRAY, lw=1.5, zorder=1)

    # Corner markers
    for i in range(3):
        marker = mpatches.Circle(vertices[i], 0.025, facecolor=FACTOR_HEX[i],
                                  edgecolor='white', lw=1, zorder=2)
        ax.add_patch(marker)

    # Factor labels
    factor_labels = ['animate', 'round', 'natural']
    offsets = [(0, 0.08), (-0.08, -0.06), (0.08, -0.06)]
    for i, (v, label, (dx, dy)) in enumerate(zip(vertices, factor_labels, offsets)):
        ax.text(v[0] + dx, v[1] + dy, label, fontsize=7, ha='center',
                color=FACTOR_HEX[i], fontweight='medium', style='italic')

    # Compute positions
    positions = np.array([barycentric_to_cartesian(W[i], vertices) for i in range(n)])

    # Collision resolution
    node_radius = 0.032
    min_dist = node_radius * 2.3
    for _ in range(60):
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 0.001:
                    push = (min_dist - dist) / 2 * diff / dist * 0.4
                    positions[i] += push
                    positions[j] -= push

    # Draw nodes
    for i in range(n):
        color = blend_color(W[i])
        circle = mpatches.Circle(positions[i], node_radius, facecolor=color,
                                  edgecolor='white', lw=1, zorder=3)
        ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
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

    fig.text(0.155, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 2. RSM
    ax2 = fig.add_axes([0.18, y_main + 0.05, 0.17, h_main - 0.05])
    draw_rsm(ax2, S_ord, cmap)
    fig.text(0.265, 0.06, 'Similarity (RSM)', fontsize=9, ha='center', color=CHARCOAL)

    fig.text(0.375, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 3. Graph
    ax3 = fig.add_axes([0.40, y_main, 0.22, h_main])
    draw_graph(ax3, W_ord, S_ord)
    fig.text(0.51, 0.06, 'Graph', fontsize=9, ha='center', color=CHARCOAL)

    fig.text(0.635, 0.52, '→', fontsize=18, color=MEDIUM_GRAY, ha='center')

    # 4. Ternary simplex
    ax4 = fig.add_axes([0.67, y_main, 0.31, h_main])
    draw_ternary_compact(ax4, W_ord)
    fig.text(0.825, 0.06, 'Factor Space (Simplex)', fontsize=9, ha='center', color=CHARCOAL)

    fig.text(0.5, 0.95, 'From Representations to Interpretable Dimensions',
             fontsize=11, ha='center', color=CHARCOAL, fontweight='medium')

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "intro_v3.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: intro_v3.svg")


if __name__ == "__main__":
    main()
