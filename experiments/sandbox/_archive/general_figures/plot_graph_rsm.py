"""Visualize graph and RSM with community structure."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Ellipse


def generate_block_rsm(
    community_sizes: list[int],
    within_sim: float = 0.8,
    between_sim: float = 0.2,
    noise: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate RSM with block-diagonal community structure."""
    rng = np.random.default_rng(seed)
    n = sum(community_sizes)
    rsm = np.full((n, n), between_sim)

    idx = 0
    for size in community_sizes:
        rsm[idx : idx + size, idx : idx + size] = within_sim
        idx += size

    rsm += rng.normal(0, noise, (n, n))
    rsm = (rsm + rsm.T) / 2
    rsm = np.clip(rsm, 0, 1)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_to_graph(
    rsm: np.ndarray,
    threshold: float | None = None,
) -> nx.Graph:
    """Convert RSM to networkx graph.

    Args:
        rsm: Similarity matrix
        threshold: If provided, only include edges with similarity >= threshold.
                   If None, include all edges (fully connected).
    """
    n = rsm.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            weight = rsm[i, j]
            if threshold is None or weight >= threshold:
                G.add_edge(i, j, weight=weight)
    return G


def get_community_positions(
    community_sizes: list[int],
    community_centers: list[tuple[float, float]],
    radius: float = 0.4,
    seed: int = 42,
) -> dict[int, tuple[float, float]]:
    """Generate node positions clustered around community centers."""
    rng = np.random.default_rng(seed)
    pos = {}
    node_idx = 0

    for size, center in zip(community_sizes, community_centers):
        angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
        angles += rng.uniform(0, 0.5)
        for angle in angles:
            r = radius * (0.6 + 0.4 * rng.random())
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            pos[node_idx] = (x, y)
            node_idx += 1
    return pos


def draw_community_ellipse(
    ax: plt.Axes,
    nodes: list[int],
    pos: dict,
    color: str,
    alpha: float = 0.15,
    padding: float = 0.3,
) -> None:
    """Draw shaded ellipse around community nodes."""
    points = np.array([pos[n] for n in nodes])
    center = points.mean(axis=0)
    width = points[:, 0].ptp() + padding * 2
    height = points[:, 1].ptp() + padding * 2
    ellipse = Ellipse(
        center, width, height, alpha=alpha, facecolor=color, edgecolor="none"
    )
    ax.add_patch(ellipse)


def plot_graph_and_rsm(
    rsm: np.ndarray,
    community_sizes: list[int],
    community_colors: list[str],
    threshold: float | None = 0.5,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (6, 3),
) -> plt.Figure:
    """Plot graph with communities and corresponding RSM.

    Args:
        rsm: Similarity matrix
        community_sizes: Number of nodes per community
        community_colors: Color for each community
        threshold: Similarity threshold for showing edges.
                   If None, show all edges with varying opacity.
        output_path: Where to save figure (optional)
        figsize: Figure size
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_communities = len(community_sizes)
    community_centers = [
        (-1.0, 0.8),
        (1.0, 0.8),
        (0.0, -0.7),
    ][:n_communities]

    pos = get_community_positions(community_sizes, community_centers)

    # Assign community labels
    community_labels = []
    for i, size in enumerate(community_sizes):
        community_labels.extend([i] * size)

    node_colors = [community_colors[c] for c in community_labels]

    # === Left panel: Graph ===
    ax = axes[0]

    # Draw community regions
    node_idx = 0
    for i, size in enumerate(community_sizes):
        nodes = list(range(node_idx, node_idx + size))
        draw_community_ellipse(ax, nodes, pos, community_colors[i], alpha=0.2)
        node_idx += size

    G = rsm_to_graph(rsm, threshold=threshold)

    # Draw edges
    if threshold is not None:
        # Thresholded: solid within, dashed between
        within_edges = [
            (u, v)
            for u, v in G.edges()
            if community_labels[u] == community_labels[v]
        ]
        between_edges = [
            (u, v)
            for u, v in G.edges()
            if community_labels[u] != community_labels[v]
        ]

        nx.draw_networkx_edges(
            G, pos, edgelist=within_edges, width=1.5, alpha=0.7, edge_color="0.3", ax=ax
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=between_edges,
            width=1.0,
            alpha=0.4,
            edge_color="0.5",
            style="dashed",
            ax=ax,
        )
    else:
        # Fully connected: edge opacity/width by similarity
        for u, v, data in G.edges(data=True):
            weight = data["weight"]
            alpha = 0.1 + 0.6 * weight
            width = 0.3 + 2.0 * weight
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                edge_color="0.4",
                ax=ax,
            )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=250,
        edgecolors="white",
        linewidths=1.2,
        ax=ax,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.5, 1.5)

    # === Right panel: RSM ===
    ax = axes[1]

    # Use pcolormesh for true vector output (not rasterized like imshow)
    n = rsm.shape[0]
    x = np.arange(n + 1) - 0.5
    y = np.arange(n + 1) - 0.5
    im = ax.pcolormesh(x, y, rsm, cmap="RdYlBu_r", vmin=0, vmax=1, rasterized=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # Flip y-axis to match imshow orientation
    ax.set_aspect("equal")

    # Add community outlines
    node_idx = 0
    for i, size in enumerate(community_sizes):
        rect = plt.Rectangle(
            (node_idx - 0.5, node_idx - 0.5),
            size,
            size,
            fill=False,
            edgecolor=community_colors[i],
            linewidth=2,
        )
        ax.add_patch(rect)
        node_idx += size

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Similarity", fontsize=9)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    community_sizes = [3, 4, 3]
    community_colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    # Realistic RSM: higher noise, medium between-similarity for better color range
    rsm = generate_block_rsm(
        community_sizes, within_sim=0.75, between_sim=0.35, noise=0.15
    )

    # Realistic version: sparse graph with noisy RSM showing full color range
    plot_graph_and_rsm(
        rsm,
        community_sizes,
        community_colors,
        threshold=0.5,
        output_path=output_dir / "graph_rsm_realistic.pdf",
    )
    print(f"Saved realistic version to {output_dir / 'graph_rsm_realistic.pdf'}")

    plt.show()
