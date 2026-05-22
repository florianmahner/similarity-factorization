"""Force-directed graph - nodes colored by membership weights like RSM bars."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

CHARCOAL = '#3d3d3d'
LIGHT_GRAY = '#d0d0d0'
MEDIUM_GRAY = '#888888'

# Colors chosen for distinct blends:
# F1+F2 = warm purple, F2+F3 = teal, F1+F3 = olive
FACTOR_COLORS = np.array([
    [0.88, 0.44, 0.25],  # orange-coral (#E07040)
    [0.31, 0.44, 0.75],  # blue-indigo (#5070C0)
    [0.44, 0.69, 0.25],  # yellow-green (#70B040)
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
    """Blend factor colors based on membership weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    rgb = weights @ FACTOR_COLORS
    return rgb


def create_graph(S, threshold=0.12):
    n = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=S[i, j])
    return G


def get_layout(G, W, seed=42):
    """Layout that groups nodes by factor membership."""
    n = len(W)

    # Factor centers in triangular arrangement
    factor_centers = {
        0: np.array([-0.35, -0.20]),
        1: np.array([0.35, -0.20]),
        2: np.array([0.0, 0.38]),
    }

    # Initial positions based on membership
    pos = {}
    rng = np.random.default_rng(seed)

    for i in range(n):
        center = sum(W[i, k] * factor_centers[k] for k in range(3))
        offset = rng.uniform(-0.06, 0.06, 2)
        pos[i] = center + offset

    # Light spring refinement - keep communities intact
    pos = nx.spring_layout(G, pos=pos, k=0.25, iterations=30, seed=seed)

    # Scale down to make nodes closer
    center = np.mean(list(pos.values()), axis=0)
    for i in pos:
        pos[i] = center + (pos[i] - center) * 0.75

    return pos


def compute_cluster_centers(W, pos):
    """Compute actual cluster centers from node positions."""
    n = len(W)
    dominant = np.argmax(W, axis=1)
    centers = []
    for k in range(3):
        members = [i for i in range(n) if dominant[i] == k]
        if members:
            cx = np.mean([pos[i][0] for i in members])
            cy = np.mean([pos[i][1] for i in members])
            centers.append((cx, cy))
        else:
            centers.append((0, 0))
    return centers


def draw_gradient_blobs(ax, cluster_centers):
    """Draw soft gradient blobs behind each cluster."""
    for (cx, cy), color in zip(cluster_centers, FACTOR_HEX):
        for r, alpha in [(0.55, 0.04), (0.45, 0.07), (0.35, 0.10), (0.25, 0.14), (0.15, 0.20), (0.08, 0.28)]:
            circle = mpatches.Circle((cx, cy), r, facecolor=color,
                                    alpha=alpha, edgecolor='none', zorder=0,
                                    clip_on=False)
            ax.add_patch(circle)


def draw_contour_rings(ax, cluster_centers):
    """Draw contour rings behind each cluster."""
    for (cx, cy), color in zip(cluster_centers, FACTOR_HEX):
        for r, lw, alpha in [(0.50, 0.5, 0.20), (0.38, 0.8, 0.35), (0.26, 1.2, 0.50), (0.16, 1.5, 0.65)]:
            circle = mpatches.Circle((cx, cy), r, facecolor='none',
                                    edgecolor=color, lw=lw, alpha=alpha, zorder=0,
                                    clip_on=False)
            ax.add_patch(circle)
        # Soft center
        circle = mpatches.Circle((cx, cy), 0.10, facecolor=color,
                                alpha=0.25, edgecolor='none', zorder=0,
                                clip_on=False)
        ax.add_patch(circle)


def draw_graph(ax, W, G, pos, show_labels=True):
    """Draw the graph with edges and nodes."""
    n = len(W)
    threshold = 0.12

    # Get all edge weights to normalize properly
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    min_w, max_w = min(weights), max(weights)

    # Draw edges - strength based on similarity
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        # Normalize to 0-1 range based on actual weight distribution
        strength = (weight - min_w) / (max_w - min_w) if max_w > min_w else 0.5
        alpha = 0.35 + 0.55 * strength
        lw = 0.5 + 1.8 * strength

        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=MEDIUM_GRAY,
            linewidth=lw,
            alpha=alpha,
            zorder=1,
            clip_on=False
        )

    # Draw nodes - larger for dense layout
    node_radius = 0.075
    for i in range(n):
        x, y = pos[i]
        color = blend_color(W[i])

        circle = mpatches.Circle(
            (x, y), node_radius,
            facecolor=color,
            edgecolor='white',
            linewidth=1.0,
            zorder=3,
            clip_on=False
        )
        ax.add_patch(circle)

    # Factor labels
    if show_labels:
        label_positions = [
            (-0.45, -0.38, 'F₁', FACTOR_HEX[0]),
            (0.45, -0.38, 'F₂', FACTOR_HEX[1]),
            (0.0, 0.52, 'F₃', FACTOR_HEX[2]),
        ]
        for x, y, label, color in label_positions:
            ax.text(x, y, label, fontsize=10, ha='center', va='center',
                    color=color, fontweight='medium', clip_on=False)


def setup_ax(ax):
    ax.set_xlim(-0.60, 0.60)
    ax.set_ylim(-0.50, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    W, S, order = create_data()
    G = create_graph(S, threshold=0.12)
    pos = get_layout(G, W)

    # Compute cluster centers from actual node positions
    cluster_centers = compute_cluster_centers(W, pos)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    versions = [
        (None, "figure_graph", True),
        (None, "figure_graph_plain", False),  # No labels version
        (draw_gradient_blobs, "figure_graph_gradient", True),
        (draw_contour_rings, "figure_graph_contour", True),
    ]

    for bg_func, name, show_labels in versions:
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        setup_ax(ax)

        if bg_func:
            bg_func(ax, cluster_centers)

        draw_graph(ax, W, G, pos, show_labels=show_labels)

        fig.savefig(output_dir / f"{name}.svg", format='svg',
                   bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)
        print(f"Saved: {name}.svg")


if __name__ == "__main__":
    main()
