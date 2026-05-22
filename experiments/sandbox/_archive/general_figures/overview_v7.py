"""Overview v7 - Before/After: Unknown structure → Revealed soft clusters."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#c8c8c8'
MEDIUM_GRAY = '#888888'
UNKNOWN_GRAY = '#8899aa'

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
    return W, S


def blend_color(weights):
    weights = np.array(weights) / np.sum(weights)
    return weights @ FACTOR_COLORS


def create_graph(S, threshold=0.12):
    n = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])
    return G


def draw_before_graph(ax, G, seed=42):
    """Graph with unknown structure - random layout, gray nodes."""
    pos = nx.spring_layout(G, k=0.9, iterations=100, seed=seed + 99)

    threshold = 0.12
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, lw=0.5 + 0.6 * strength,
                alpha=0.25 + 0.4 * strength, zorder=1, clip_on=False)

    for i in G.nodes():
        x, y = pos[i]
        circle = mpatches.Circle((x, y), 0.085, facecolor=UNKNOWN_GRAY,
                                  edgecolor='white', lw=1.2, zorder=3, clip_on=False)
        ax.add_patch(circle)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_after_graph(ax, W, G, seed=42):
    """Graph with revealed structure - clustered layout, colored nodes."""
    n = len(W)

    # Cluster-aware layout
    factor_centers = {0: np.array([-0.6, -0.35]), 1: np.array([0.6, -0.35]), 2: np.array([0.0, 0.65])}
    rng = np.random.default_rng(seed)
    pos = {}
    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        pos[i] = center + rng.uniform(-0.12, 0.12, 2)
    pos = nx.spring_layout(G, pos=pos, k=0.55, iterations=80, seed=seed)

    threshold = 0.12
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        strength = (weight - threshold) / (1 - threshold)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=LIGHT_GRAY, lw=0.5 + 0.6 * strength,
                alpha=0.15 + 0.55 * strength, zorder=1, clip_on=False)

    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])
        circle = mpatches.Circle((x, y), 0.085, facecolor=color,
                                  edgecolor='white', lw=1.2, zorder=3, clip_on=False)
        ax.add_patch(circle)

    # Factor labels
    for (cx, cy), label, col in [
        ((-0.9, -0.75), 'F₁', FACTOR_HEX[0]),
        ((0.9, -0.75), 'F₂', FACTOR_HEX[1]),
        ((0.0, 1.05), 'F₃', FACTOR_HEX[2]),
    ]:
        ax.text(cx, cy, label, fontsize=11, ha='center', color=col, fontweight='bold')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, S = create_data()
    G = create_graph(S)

    fig = plt.figure(figsize=(10, 4.5))

    # Before
    ax1 = fig.add_axes([0.02, 0.08, 0.38, 0.82])
    draw_before_graph(ax1, G)
    fig.text(0.21, 0.02, 'Input: Similarity Graph', fontsize=11, ha='center', color=CHARCOAL)

    # Arrow + SRF
    ax_mid = fig.add_axes([0.42, 0.32, 0.16, 0.36])
    box = mpatches.FancyBboxPatch((0.08, 0.15), 0.84, 0.7, boxstyle="round,pad=0.05",
                                   facecolor='#f8fbfd', edgecolor=FACTOR_HEX[1], lw=2)
    ax_mid.add_patch(box)
    ax_mid.text(0.5, 0.58, 'S ≈ WWᵀ', fontsize=12, ha='center', va='center', color=CHARCOAL)
    ax_mid.text(0.5, 0.38, 'W ≥ 0', fontsize=10, ha='center', va='center', color=MEDIUM_GRAY)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis('off')

    fig.text(0.405, 0.50, '→', fontsize=22, color=MEDIUM_GRAY, ha='center')
    fig.text(0.595, 0.50, '→', fontsize=22, color=MEDIUM_GRAY, ha='center')
    fig.text(0.50, 0.18, 'SRF', fontsize=11, ha='center', color=CHARCOAL, fontweight='medium')

    # After
    ax2 = fig.add_axes([0.60, 0.08, 0.38, 0.82])
    draw_after_graph(ax2, W, G)
    fig.text(0.79, 0.02, 'Output: Soft Cluster Membership', fontsize=11, ha='center', color=CHARCOAL)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "overview_v7.svg", format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("Saved: overview_v7.svg")


if __name__ == "__main__":
    main()
