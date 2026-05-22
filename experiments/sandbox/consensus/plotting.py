"""
Visualization functions for consensus analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, setup_style, CYCLE
from src.utils.figure_theme import create_figure, despine


def plot_consensus_heatmap(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
    title: str = "Consensus Matrix",
) -> None:
    """Plot consensus matrix as heatmap, reordered by dendrogram."""
    setup_style()

    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    consensus_ordered = consensus[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(consensus_ordered, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')

    ax.set_xlabel('Samples (reordered)')
    ax.set_ylabel('Samples (reordered)')
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Co-clustering frequency')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_consensus_histogram(
    consensus: np.ndarray,
    output_path: Path,
) -> None:
    """Plot histogram of consensus values."""
    setup_style()

    triu_idx = np.triu_indices_from(consensus, k=1)
    entries = consensus[triu_idx]

    fig, ax = create_figure("single")
    ax.hist(entries, bins=50, color=TEAL, edgecolor='white', linewidth=0.5)
    ax.axvline(0.5, color=ROSE, linestyle='--', linewidth=1.5, label='Unstable')

    ax.set_xlabel('Consensus value')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    k: int | None = None,
    labels: list[str] | None = None,
    title: str = "Dendrogram",
) -> None:
    """
    Plot dendrogram with optional cut line at k clusters.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 5))

    color_threshold = 0
    if k is not None and k > 1:
        n = linkage_matrix.shape[0] + 1
        if k <= n:
            color_threshold = linkage_matrix[-(k), 2]

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6 if labels and len(labels) > 50 else 8,
        color_threshold=color_threshold,
        above_threshold_color=GRAY,
    )

    if k is not None:
        ax.axhline(y=color_threshold, color=ROSE, linestyle='--',
                   linewidth=1.5, label=f'Cut for k={k}')
        ax.legend(frameon=False)

    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance (1 - consensus)')
    ax.set_title(title)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_soft_memberships(
    memberships: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Plot soft memberships as stacked bar chart."""
    setup_style()

    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    memberships_ordered = memberships[order]

    n, k = memberships.shape

    fig, ax = plt.subplots(figsize=(12, 3))

    x = np.arange(n)
    bottom = np.zeros(n)
    colors = [CYCLE[i % len(CYCLE)] for i in range(k)]

    for cluster_id in range(k):
        ax.bar(x, memberships_ordered[:, cluster_id], bottom=bottom,
               width=1.0, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')
        bottom += memberships_ordered[:, cluster_id]

    ax.set_xlabel('Samples (reordered)')
    ax.set_ylabel('Membership')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc='upper right')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_cluster_splits(
    linkage_matrix: np.ndarray,
    cv_rank: int,
    output_path: Path,
    max_k: int = 6,
) -> None:
    """
    Show how clusters split as k increases beyond CV rank.
    """
    setup_style()

    n = linkage_matrix.shape[0] + 1
    k_range = list(range(cv_rank, min(max_k + 1, n)))

    if len(k_range) < 2:
        return

    # Get assignments at each k
    all_assignments = {k: fcluster(linkage_matrix, k, criterion='maxclust') for k in k_range}

    # Sort samples by assignment trajectory
    sort_keys = np.column_stack([all_assignments[k] for k in k_range])
    order = np.lexsort(sort_keys.T[::-1])

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.8 / len(k_range)

    for i, k in enumerate(k_range):
        assignments = all_assignments[k][order]
        x_offset = i * bar_width

        for cluster_id in range(1, k + 1):
            mask = assignments == cluster_id
            color = CYCLE[(cluster_id - 1) % len(CYCLE)]
            ax.barh(np.arange(n)[mask], bar_width, left=x_offset,
                    color=color, edgecolor='white', linewidth=0.1)

    ax.set_yticks([])
    ax.set_xticks([i * bar_width + bar_width / 2 for i in range(len(k_range))])
    ax.set_xticklabels([f'k={k}' for k in k_range])
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(f'Samples (n={n})')
    ax.set_title(f'Cluster splits beyond CV-optimal k={cv_rank}')
    ax.axvline(bar_width / 2, color='black', linestyle='-', linewidth=2, alpha=0.3)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_granularity_analysis(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    cv_rank: int,
    output_path: Path,
    max_k: int = 8,
) -> pd.DataFrame:
    """
    Analyze within/between cluster consensus at different k.

    Returns DataFrame with metrics.
    """
    setup_style()

    n = consensus.shape[0]
    results = []

    for k in range(2, min(max_k + 1, n)):
        assignments = fcluster(linkage_matrix, k, criterion='maxclust')

        # Within-cluster consensus
        within = []
        for cid in range(1, k + 1):
            members = np.where(assignments == cid)[0]
            if len(members) > 1:
                cluster_cons = consensus[np.ix_(members, members)]
                triu = cluster_cons[np.triu_indices(len(members), k=1)]
                within.append(np.mean(triu))

        # Between-cluster consensus
        between = []
        for c1 in range(1, k + 1):
            for c2 in range(c1 + 1, k + 1):
                m1 = np.where(assignments == c1)[0]
                m2 = np.where(assignments == c2)[0]
                if len(m1) > 0 and len(m2) > 0:
                    between.append(np.mean(consensus[np.ix_(m1, m2)]))

        sizes = [np.sum(assignments == c) for c in range(1, k + 1)]

        results.append({
            'k': k,
            'within_consensus': np.mean(within) if within else 0,
            'between_consensus': np.mean(between) if between else 0,
            'separation': (np.mean(within) - np.mean(between)) if within and between else 0,
            'min_cluster_size': min(sizes),
        })

    df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))

    ax = axes[0]
    ax.plot(df['k'], df['within_consensus'], 'o-', color=TEAL, label='Within', ms=5)
    ax.plot(df['k'], df['between_consensus'], 'o-', color=ROSE, label='Between', ms=5)
    ax.axvline(cv_rank, color=GRAY, linestyle='--', label=f'CV k={cv_rank}')
    ax.set_xlabel('k')
    ax.set_ylabel('Mean consensus')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Cohesion vs separation')
    despine(ax)

    ax = axes[1]
    ax.plot(df['k'], df['separation'], 'o-', color=CYAN, ms=5)
    ax.axvline(cv_rank, color=GRAY, linestyle='--')
    ax.axhline(0, color=GRAY, linestyle='-', linewidth=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('Separation')
    ax.set_title('Higher = better')
    despine(ax)

    ax = axes[2]
    ax.plot(df['k'], df['min_cluster_size'], 'o-', color=SAND, ms=5)
    ax.axvline(cv_rank, color=GRAY, linestyle='--')
    ax.set_xlabel('k')
    ax.set_ylabel('Min cluster size')
    ax.set_title('Cluster balance')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return df
