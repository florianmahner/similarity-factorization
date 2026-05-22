"""Mockup: Visualizing overlapping clusters from NMF."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import networkx as nx
import numpy as np

from src.utils.figure_theme import CMAP, GRAY, apply_theme

apply_theme()

# Cluster colors
COLORS = [CMAP[0], CMAP[1], CMAP[2]]  # red, blue, green


def create_synthetic_data(n_nodes: int = 12, n_factors: int = 3, seed: int = 42):
    """Create synthetic NMF-style soft membership matrix."""
    rng = np.random.default_rng(seed)

    # Create soft membership W matrix (nodes x factors)
    # Some nodes belong strongly to one factor, others are mixed
    W = np.zeros((n_nodes, n_factors))

    # Nodes 0-3: mostly factor 0 (red)
    W[0:4, 0] = rng.uniform(0.7, 1.0, 4)
    W[0:4, 1] = rng.uniform(0.0, 0.2, 4)
    W[0:4, 2] = rng.uniform(0.0, 0.1, 4)

    # Nodes 4-7: mostly factor 1 (blue)
    W[4:8, 1] = rng.uniform(0.7, 1.0, 4)
    W[4:8, 0] = rng.uniform(0.0, 0.2, 4)
    W[4:8, 2] = rng.uniform(0.0, 0.2, 4)

    # Nodes 8-11: mostly factor 2 (green)
    W[8:12, 2] = rng.uniform(0.7, 1.0, 4)
    W[8:12, 0] = rng.uniform(0.0, 0.1, 4)
    W[8:12, 1] = rng.uniform(0.0, 0.2, 4)

    # Add some overlap nodes
    # Node 3: red-blue overlap
    W[3, 0] = 0.6
    W[3, 1] = 0.5
    W[3, 2] = 0.1

    # Node 7: blue-green overlap
    W[7, 1] = 0.5
    W[7, 2] = 0.6
    W[7, 0] = 0.1

    # Node 2: slight red-green
    W[2, 0] = 0.7
    W[2, 2] = 0.3

    # Normalize rows to sum to 1 for visualization
    W = W / W.sum(axis=1, keepdims=True)

    # Create RSM from W @ W.T
    S = W @ W.T

    # Create graph with edges based on similarity
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if S[i, j] > 0.15:  # threshold for edge
                G.add_edge(i, j, weight=S[i, j])

    return W, S, G


def blend_colors(weights, colors):
    """Blend RGB colors based on weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()

    blended = np.zeros(3)
    for w, c in zip(weights, colors):
        # Convert hex to RGB
        rgb = np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]) / 255
        blended += w * rgb

    return blended


def draw_pie_node(ax, x, y, weights, colors, radius=0.08):
    """Draw a pie chart at position (x, y)."""
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Sort by weight for cleaner look
    order = np.argsort(weights)[::-1]

    start_angle = 90
    for idx in order:
        if weights[idx] < 0.05:
            continue
        angle = weights[idx] * 360
        wedge = mpatches.Wedge(
            (x, y), radius, start_angle, start_angle - angle,
            facecolor=colors[idx], edgecolor='white', linewidth=0.5
        )
        ax.add_patch(wedge)
        start_angle -= angle


def plot_approach_1_pie_nodes(W, S, G, ax):
    """Approach 1: Pie chart nodes showing membership proportions."""
    ax.set_title("Pie Chart Nodes", fontsize=10, fontweight='bold')

    pos = nx.spring_layout(G, seed=42, k=2)

    # Draw edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=GRAY['light'], linewidth=1, zorder=1)

    # Draw pie chart nodes
    for node in G.nodes():
        x, y = pos[node]
        draw_pie_node(ax, x, y, W[node], COLORS, radius=0.12)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_approach_2_blended_colors(W, S, G, ax):
    """Approach 2: Blended node colors."""
    ax.set_title("Blended Colors", fontsize=10, fontweight='bold')

    pos = nx.spring_layout(G, seed=42, k=2)

    # Draw edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=GRAY['light'], linewidth=1, zorder=1)

    # Draw nodes with blended colors
    for node in G.nodes():
        x, y = pos[node]
        color = blend_colors(W[node], COLORS)
        circle = mpatches.Circle((x, y), 0.1, facecolor=color,
                                  edgecolor='white', linewidth=1.5, zorder=2)
        ax.add_patch(circle)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_approach_3_rings(W, S, G, ax):
    """Approach 3: Concentric rings showing membership."""
    ax.set_title("Concentric Rings", fontsize=10, fontweight='bold')

    pos = nx.spring_layout(G, seed=42, k=2)

    # Draw edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=GRAY['light'], linewidth=1, zorder=1)

    # Draw nodes with concentric rings
    for node in G.nodes():
        x, y = pos[node]
        weights = W[node]

        # Sort factors by weight
        order = np.argsort(weights)[::-1]

        # Draw rings from outside in
        radii = [0.14, 0.10, 0.06]
        for i, idx in enumerate(order):
            if weights[idx] > 0.1:
                alpha = 0.3 + 0.7 * weights[idx]
                circle = mpatches.Circle(
                    (x, y), radii[i],
                    facecolor=COLORS[idx], alpha=alpha,
                    edgecolor='none', zorder=2 + i
                )
                ax.add_patch(circle)

        # White center dot
        circle = mpatches.Circle((x, y), 0.03, facecolor='white',
                                  edgecolor='none', zorder=10)
        ax.add_patch(circle)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_approach_4_dominant_with_halo(W, S, G, ax):
    """Approach 4: Dominant color with secondary halo."""
    ax.set_title("Dominant + Halo", fontsize=10, fontweight='bold')

    pos = nx.spring_layout(G, seed=42, k=2)

    # Draw edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=GRAY['light'], linewidth=1, zorder=1)

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        weights = W[node]
        order = np.argsort(weights)[::-1]

        dominant = order[0]
        secondary = order[1] if weights[order[1]] > 0.15 else None

        # Draw halo for secondary membership
        if secondary is not None:
            halo = mpatches.Circle(
                (x, y), 0.15,
                facecolor=COLORS[secondary], alpha=0.3,
                edgecolor='none', zorder=1
            )
            ax.add_patch(halo)

        # Draw main node
        circle = mpatches.Circle(
            (x, y), 0.09,
            facecolor=COLORS[dominant],
            edgecolor='white', linewidth=1.5, zorder=2
        )
        ax.add_patch(circle)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_rsm_with_loadings(W, S, ax):
    """RSM with factor loading bars on the side."""
    ax.set_title("RSM + Factor Loadings", fontsize=10, fontweight='bold')

    n = S.shape[0]

    # Reorder by dominant factor for cleaner blocks
    dominant = np.argmax(W, axis=1)
    order = np.lexsort((W.max(axis=1), dominant))[::-1]
    S_ordered = S[order][:, order]
    W_ordered = W[order]

    # Plot RSM
    im = ax.imshow(S_ordered, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')

    # Add factor loading bars on top
    bar_height = 0.8
    for i in range(n):
        bottom = -bar_height - 0.3
        for k in range(3):
            width = W_ordered[i, k] * 0.8
            ax.add_patch(mpatches.Rectangle(
                (i - 0.4, bottom + k * 0.25), 0.8, 0.2,
                facecolor=COLORS[k], alpha=W_ordered[i, k],
                edgecolor='none'
            ))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -bar_height - 0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Similarity', fontsize=8)


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Generate data
    W, S, G = create_synthetic_data()

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    # Graph approaches
    plot_approach_1_pie_nodes(W, S, G, axes[0, 0])
    plot_approach_2_blended_colors(W, S, G, axes[0, 1])
    plot_approach_3_rings(W, S, G, axes[0, 2])
    plot_approach_4_dominant_with_halo(W, S, G, axes[1, 0])

    # RSM with loadings
    plot_rsm_with_loadings(W, S, axes[1, 1])

    # Legend
    axes[1, 2].axis('off')
    for i, (color, label) in enumerate(zip(COLORS, ['Factor 1', 'Factor 2', 'Factor 3'])):
        axes[1, 2].add_patch(mpatches.Circle((0.2, 0.8 - i * 0.25), 0.08,
                                              facecolor=color, transform=axes[1, 2].transAxes))
        axes[1, 2].text(0.35, 0.8 - i * 0.25, label, transform=axes[1, 2].transAxes,
                        va='center', fontsize=10)
    axes[1, 2].text(0.5, 0.15, "Overlapping nodes show\nmixed cluster membership",
                    transform=axes[1, 2].transAxes, ha='center', va='center',
                    fontsize=9, style='italic', color=GRAY['medium'])

    plt.suptitle("Mockup: Visualizing Overlapping Clusters (NMF)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save as SVG
    output_path = output_dir / "graph_overlap_mockup.svg"
    fig.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
