"""Elegant graph visualization matching the RSM style.

Clean force-directed layout with nodes grouped by factor.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

# Sophisticated muted palette (matching RSM)
CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#d0d0d0'
MEDIUM_GRAY = '#888888'
FAINT_GRAY = '#eeeeee'

# Muted factor colors (same as RSM)
FACTOR_COLORS = ['#c75b5b', '#5b8fc7', '#5bc77a']


def create_data(seed=42):
    """Create synthetic soft membership data."""
    rng = np.random.default_rng(seed)
    W_list = []

    # Pure cluster items (4 each)
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
    return W, S


def create_graph(S, threshold=0.15):
    """Create networkx graph from similarity matrix."""
    n = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])

    return G


def get_clustered_layout(G, W, seed=42):
    """Create layout that groups nodes by dominant factor."""
    n = len(W)

    # Initial positions based on factor membership
    # Place factor centers at triangle vertices
    factor_centers = {
        0: np.array([-0.6, -0.3]),   # F1 - left
        1: np.array([0.6, -0.3]),    # F2 - right
        2: np.array([0.0, 0.6]),     # F3 - top
    }

    # Initialize positions based on membership weights
    pos = {}
    rng = np.random.default_rng(seed)

    for i in range(n):
        # Weighted average of factor centers
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        # Add small random offset
        offset = rng.uniform(-0.15, 0.15, 2)
        pos[i] = center + offset

    # Refine with spring layout (keeping general structure)
    pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=50, seed=seed)

    return pos


def main():
    W, S = create_data()
    n = len(W)

    G = create_graph(S, threshold=0.12)
    pos = get_clustered_layout(G, W)

    # Create figure
    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    # Draw edges with opacity based on weight
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        alpha = 0.15 + 0.6 * weight  # scale to visible range

        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=LIGHT_GRAY,
            linewidth=0.6,
            alpha=alpha,
            zorder=1
        )

    # Draw nodes
    for i in range(n):
        x, y = pos[i]
        dominant = np.argmax(W[i])
        max_weight = W[i].max()

        # Size based on how "pure" the membership is
        size = 0.06 + 0.02 * max_weight

        if max_weight < 0.55:
            # Overlap node - hollow with dark stroke
            circle = mpatches.Circle(
                (x, y), size,
                facecolor='white',
                edgecolor=CHARCOAL,
                linewidth=1.2,
                zorder=3
            )
        else:
            # Pure node - filled with factor color
            circle = mpatches.Circle(
                (x, y), size,
                facecolor=FACTOR_COLORS[dominant],
                edgecolor='white',
                linewidth=1.0,
                zorder=3
            )
        ax.add_patch(circle)

    # Subtle factor labels
    label_positions = [
        (-0.75, -0.55, 'F₁'),
        (0.75, -0.55, 'F₂'),
        (0.0, 0.85, 'F₃'),
    ]
    for x, y, label in label_positions:
        ax.text(x, y, label, fontsize=9, ha='center', va='center',
                color=MEDIUM_GRAY, fontweight='medium')

    # Clean up
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.75, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Save
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "elegant_graph.svg", format='svg',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_dir / "elegant_graph.png", dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print("Saved: elegant_graph.svg")


if __name__ == "__main__":
    main()
