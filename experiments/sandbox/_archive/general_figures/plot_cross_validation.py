"""Visualize cross-validation procedure for similarity matrix factorization."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import networkx as nx
import numpy as np


COMMUNITY_SIZES = [3, 4, 3]
COMMUNITY_COLORS = ["#E74C3C", "#3498DB", "#2ECC71"]
HELD_OUT_COLOR = "#FF6B00"  # bright orange for held-out


def get_realistic_rsm() -> np.ndarray:
    """Return the realistic RSM with community structure and varied off-diagonal."""
    return np.array([
        [1.0,  0.82, 0.78,  0.48, 0.35, 0.30, 0.25,  0.55, 0.28, 0.22],
        [0.82, 1.0,  0.85,  0.52, 0.58, 0.32, 0.28,  0.30, 0.25, 0.20],
        [0.78, 0.85, 1.0,   0.35, 0.30, 0.45, 0.38,  0.32, 0.52, 0.28],
        [0.48, 0.52, 0.35,  1.0,  0.75, 0.80, 0.72,  0.38, 0.32, 0.28],
        [0.35, 0.58, 0.30,  0.75, 1.0,  0.78, 0.82,  0.35, 0.30, 0.25],
        [0.30, 0.32, 0.45,  0.80, 0.78, 1.0,  0.76,  0.55, 0.48, 0.42],
        [0.25, 0.28, 0.38,  0.72, 0.82, 0.76, 1.0,   0.42, 0.38, 0.52],
        [0.55, 0.30, 0.32,  0.38, 0.35, 0.55, 0.42,  1.0,  0.85, 0.80],
        [0.28, 0.25, 0.52,  0.32, 0.30, 0.48, 0.38,  0.85, 1.0,  0.82],
        [0.22, 0.20, 0.28,  0.28, 0.25, 0.42, 0.52,  0.80, 0.82, 1.0],
    ])


def get_community_positions(seed: int = 42) -> dict[int, tuple[float, float]]:
    """Generate node positions clustered around community centers."""
    rng = np.random.default_rng(seed)
    community_centers = [(-1.0, 0.8), (1.0, 0.8), (0.0, -0.7)]
    radius = 0.4
    pos = {}
    node_idx = 0

    for size, center in zip(COMMUNITY_SIZES, community_centers):
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
) -> None:
    """Draw shaded ellipse around community nodes."""
    points = np.array([pos[n] for n in nodes])
    center = points.mean(axis=0)
    width = points[:, 0].ptp() + 0.6
    height = points[:, 1].ptp() + 0.6
    ellipse = Ellipse(center, width, height, alpha=alpha, facecolor=color, edgecolor="none")
    ax.add_patch(ellipse)


def rsm_to_graph(rsm: np.ndarray, threshold: float = 0.5) -> nx.Graph:
    """Convert RSM to graph with threshold."""
    n = rsm.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rsm[i, j] >= threshold:
                G.add_edge(i, j, weight=rsm[i, j])
    return G


def get_community_labels() -> list[int]:
    """Get community label for each node."""
    labels = []
    for i, size in enumerate(COMMUNITY_SIZES):
        labels.extend([i] * size)
    return labels


def draw_graph_panel(
    ax: plt.Axes,
    rsm: np.ndarray,
    pos: dict,
    community_labels: list[int],
    held_out_edges: list[tuple[int, int]] | None = None,
    threshold: float = 0.5,
) -> None:
    """Draw graph with community structure."""
    n_nodes = rsm.shape[0]
    node_colors = [COMMUNITY_COLORS[c] for c in community_labels]

    # Draw community ellipses
    node_idx = 0
    for i, size in enumerate(COMMUNITY_SIZES):
        nodes = list(range(node_idx, node_idx + size))
        draw_community_ellipse(ax, nodes, pos, COMMUNITY_COLORS[i], alpha=0.2)
        node_idx += size

    # Get edges from RSM
    G = rsm_to_graph(rsm, threshold=threshold)
    all_edges = list(G.edges())

    if held_out_edges is None:
        held_out_edges = []

    # Convert to set of frozensets for direction-independent comparison
    held_out_set = {frozenset(e) for e in held_out_edges}
    train_edges = [e for e in all_edges if frozenset(e) not in held_out_set]
    # Get actual held-out edges that exist in graph
    held_out_in_graph = [e for e in all_edges if frozenset(e) in held_out_set]
    within_train = [(u, v) for u, v in train_edges if community_labels[u] == community_labels[v]]
    between_train = [(u, v) for u, v in train_edges if community_labels[u] != community_labels[v]]

    # Draw train edges (faded when there are held-out)
    train_alpha = 0.3 if held_out_in_graph else 0.7
    nx.draw_networkx_edges(G, pos, edgelist=within_train, width=1.2, alpha=train_alpha,
                           edge_color="0.4", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=between_train, width=0.8, alpha=train_alpha * 0.7,
                           edge_color="0.5", style="dashed", ax=ax)

    # Draw held-out edges using matplotlib directly (more visible)
    if held_out_in_graph:
        for u, v in held_out_in_graph:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color=HELD_OUT_COLOR, linewidth=5, alpha=1.0, solid_capstyle="round")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200,
                           edgecolors="white", linewidths=1.2, ax=ax)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)


def draw_edges_only(
    ax: plt.Axes,
    edges: list[tuple[int, int]],
    pos: dict,
    community_labels: list[int],
    color: str,
    alpha: float = 0.7,
) -> None:
    """Draw only specified edges (no nodes)."""
    for u, v in edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        # Dashed for between-community
        style = "-" if community_labels[u] == community_labels[v] else "--"
        ax.plot(x, y, color=color, linewidth=2.5, alpha=alpha, linestyle=style)


def plot_cross_validation_figure(output_path: Path | None = None) -> plt.Figure:
    """Create cross-validation illustration figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1], hspace=0.35, wspace=0.2)

    rsm = get_realistic_rsm()
    pos = get_community_positions()
    community_labels = get_community_labels()
    n = rsm.shape[0]
    node_colors = [COMMUNITY_COLORS[c] for c in community_labels]

    # Select held-out edges
    held_out_edges = [(0, 1), (3, 5), (7, 8), (1, 4), (5, 7)]
    G = rsm_to_graph(rsm, threshold=0.5)
    all_edges = list(G.edges())
    held_set = {frozenset(e) for e in held_out_edges}
    train_edges = [e for e in all_edges if frozenset(e) not in held_set]
    val_edges = [e for e in all_edges if frozenset(e) in held_set]

    # === Panel A: Full data ===
    ax = fig.add_subplot(gs[0, 0])
    draw_graph_panel(ax, rsm, pos, community_labels, held_out_edges=None)
    ax.set_title("Full data", fontsize=10, fontweight="bold")

    # === Panel B: Train edges only ===
    ax = fig.add_subplot(gs[0, 1])
    # Draw community ellipses
    node_idx = 0
    for i, size in enumerate(COMMUNITY_SIZES):
        nodes = list(range(node_idx, node_idx + size))
        draw_community_ellipse(ax, nodes, pos, COMMUNITY_COLORS[i], alpha=0.15)
        node_idx += size
    # Draw train edges
    draw_edges_only(ax, train_edges, pos, community_labels, color="0.3", alpha=0.7)
    # Draw nodes
    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=200, c=node_colors[node], edgecolors="white", linewidths=1.2, zorder=10)
    ax.set_title("Train edges", fontsize=10, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    # === Panel C: Validation edges only ===
    ax = fig.add_subplot(gs[0, 2])
    # Draw community ellipses (faded)
    node_idx = 0
    for i, size in enumerate(COMMUNITY_SIZES):
        nodes = list(range(node_idx, node_idx + size))
        draw_community_ellipse(ax, nodes, pos, COMMUNITY_COLORS[i], alpha=0.1)
        node_idx += size
    # Draw validation edges
    draw_edges_only(ax, val_edges, pos, community_labels, color=HELD_OUT_COLOR, alpha=1.0)
    # Draw nodes (faded)
    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=200, c=node_colors[node], edgecolors="white", linewidths=1.2, alpha=0.5, zorder=10)
    ax.set_title("Validation edges", fontsize=10, fontweight="bold", color=HELD_OUT_COLOR)
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    # === Panel D: RSM with held-out cells (shown as white/missing) ===
    ax = fig.add_subplot(gs[0, 3])

    # Create masked RSM for display
    rsm_display = rsm.copy()
    for (i, j) in held_out_edges:
        rsm_display[i, j] = np.nan
        rsm_display[j, i] = np.nan

    im = ax.imshow(rsm_display, cmap="RdYlBu_r", vmin=0, vmax=1)

    # Mark held-out cells with orange fill
    for (i, j) in held_out_edges:
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=HELD_OUT_COLOR, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        rect2 = plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                               facecolor=HELD_OUT_COLOR, edgecolor="white", linewidth=1)
        ax.add_patch(rect2)

    # Community outlines
    node_idx = 0
    for i, size in enumerate(COMMUNITY_SIZES):
        rect = plt.Rectangle((node_idx - 0.5, node_idx - 0.5), size, size,
                              fill=False, edgecolor=COMMUNITY_COLORS[i], linewidth=1.5)
        ax.add_patch(rect)
        node_idx += size

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Held-out entries", fontsize=10, fontweight="bold")

    # === Panel E: Train/validation curves ===
    ax = fig.add_subplot(gs[0, 4])

    ranks = np.arange(1, 20)
    train_mse = 0.8 * np.exp(-0.3 * ranks) + 0.02
    val_mse = 0.9 * np.exp(-0.25 * ranks) + 0.08 + 0.015 * (ranks > 8) * (ranks - 8)

    ax.plot(ranks, train_mse, color="0.3", lw=2, label="Train")
    ax.plot(ranks, val_mse, color=HELD_OUT_COLOR, lw=2, label="Validation")

    opt_rank = ranks[np.argmin(val_mse)]
    ax.axvline(opt_rank, color="0.5", ls=":", lw=1)
    ax.scatter([opt_rank], [val_mse[opt_rank - 1]], color=HELD_OUT_COLOR, s=50, zorder=5)

    ax.set_xlabel("Rank", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.set_title("Select rank", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)

    # === Bottom row: K-fold RSMs ===
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.axis("off")

    n_folds = 5
    fold_width = 0.16
    gap = 0.025
    start_x = 0.08

    for fold in range(n_folds):
        left = start_x + fold * (fold_width + gap)
        ax_rsm = ax_bottom.inset_axes([left, 0.15, fold_width, 0.75])

        # Create fold mask
        mask = np.zeros((n, n), dtype=bool)
        rng_fold = np.random.default_rng(fold)
        n_holdout = n * (n - 1) // 2 // n_folds
        upper_tri = [(i, j) for i in range(n) for j in range(i + 1, n)]
        holdout_pairs = rng_fold.choice(len(upper_tri), size=n_holdout, replace=False)
        for idx in holdout_pairs:
            i, j = upper_tri[idx]
            mask[i, j] = True
            mask[j, i] = True

        # Plot RSM with held-out as white
        rsm_fold = rsm.copy()
        rsm_fold[mask] = np.nan
        ax_rsm.imshow(rsm_fold, cmap="RdYlBu_r", vmin=0, vmax=1)

        # Overlay held-out cells with bright orange
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          facecolor=HELD_OUT_COLOR, alpha=0.9)
                    ax_rsm.add_patch(rect)

        ax_rsm.set_xticks([])
        ax_rsm.set_yticks([])
        for spine in ax_rsm.spines.values():
            spine.set_linewidth(0.5)
        ax_rsm.set_title(f"Fold {fold + 1}", fontsize=8)

    # Arrow and label
    ax_bottom.annotate("", xy=(0.92, 0.5), xytext=(0.06, 0.5),
                       arrowprops=dict(arrowstyle="->", color="0.4", lw=1.5))
    ax_bottom.text(0.5, 0.0, "K-fold cross-validation on similarity entries",
                   ha="center", fontsize=9, style="italic")

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    plot_cross_validation_figure(output_dir / "cross_validation.pdf")
    print(f"Saved to {output_dir / 'cross_validation.pdf'}")
