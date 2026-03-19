"""
Visualize SRF dimensions from consensus analysis.

Shows the actual embedding structure, not just the clustering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.colors import CYCLE, GRAY_LIGHT, TEAL, setup_style
from src.utils.figure_theme import create_figure, despine


def plot_embedding_heatmap(
    embedding: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
    title: str = "Embedding",
) -> None:
    """
    Plot embedding as heatmap, samples reordered by dendrogram.

    Parameters
    ----------
    embedding : ndarray of shape (n_samples, rank)
    linkage_matrix : from consensus hierarchical clustering
    output_path : Path
    """
    setup_style()

    # Reorder by consensus dendrogram
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    embedding_ordered = embedding[order]

    n_samples, rank = embedding.shape

    fig, ax = plt.subplots(figsize=(max(3, rank * 0.5), 8))

    im = ax.imshow(embedding_ordered, cmap='viridis', aspect='auto')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Samples (reordered)')
    ax.set_xticks(range(rank))
    ax.set_xticklabels([f'{i+1}' for i in range(rank)])
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label('Loading')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_dimension_profiles(
    embedding: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot each dimension as a bar chart, samples reordered.
    """
    setup_style()

    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    embedding_ordered = embedding[order]

    n_samples, rank = embedding.shape

    fig, axes = plt.subplots(rank, 1, figsize=(10, rank * 1.5), sharex=True)
    if rank == 1:
        axes = [axes]

    x = np.arange(n_samples)

    for d in range(rank):
        ax = axes[d]
        values = embedding_ordered[:, d]
        colors = [CYCLE[d % len(CYCLE)] if v > 0.1 else GRAY_LIGHT for v in values]
        ax.bar(x, values, width=1.0, color=colors, edgecolor='none')
        ax.set_ylabel(f'Dim {d+1}')
        ax.set_ylim(0, values.max() * 1.1)
        despine(ax)

    axes[-1].set_xlabel('Samples (reordered)')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_dimension_consistency(
    embeddings: list[np.ndarray],
    linkage_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Show consistency of each dimension across runs.

    For each dimension, plot mean ± std loading across runs.
    """
    setup_style()

    # Need to align embeddings first (by dominant assignment)
    # Simple approach: just show variance in dominant dimension assignment
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']

    n_samples = embeddings[0].shape[0]
    rank = embeddings[0].shape[1]

    # Stack and reorder
    stacked = np.stack([e[order] for e in embeddings], axis=0)  # (n_runs, n_samples, rank)

    # For each sample, compute how consistently it's assigned to each dimension
    # Assignment = argmax
    assignments = np.argmax(stacked, axis=2)  # (n_runs, n_samples)

    # For each sample, count how often it's assigned to each dimension
    assignment_counts = np.zeros((n_samples, rank))
    for i in range(n_samples):
        for d in range(rank):
            assignment_counts[i, d] = np.mean(assignments[:, i] == d)

    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(n_samples)
    bottom = np.zeros(n_samples)

    for d in range(rank):
        ax.bar(x, assignment_counts[:, d], bottom=bottom, width=1.0,
               color=CYCLE[d % len(CYCLE)], label=f'Dim {d+1}')
        bottom += assignment_counts[:, d]

    ax.set_xlabel('Samples (reordered)')
    ax.set_ylabel('Assignment frequency')
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc='upper right', ncol=min(rank, 4))
    ax.set_title('Which dimension dominates each sample across runs')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_top_samples_per_dimension(
    embedding: np.ndarray,
    output_path: Path,
    labels: list[str] | None = None,
    top_k: int = 10,
) -> None:
    """
    For each dimension, show top-k samples with highest loading.
    """
    setup_style()

    n_samples, rank = embedding.shape

    if labels is None:
        labels = [str(i) for i in range(n_samples)]

    fig, axes = plt.subplots(1, rank, figsize=(rank * 3, 4))
    if rank == 1:
        axes = [axes]

    for d in range(rank):
        ax = axes[d]
        loadings = embedding[:, d]
        top_idx = np.argsort(loadings)[-top_k:][::-1]

        y_pos = np.arange(top_k)
        ax.barh(y_pos, loadings[top_idx], color=CYCLE[d % len(CYCLE)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in top_idx], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Loading')
        ax.set_title(f'Dimension {d+1}')
        despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_embedding_scatter(
    embedding: np.ndarray,
    output_path: Path,
    labels: list[str] | None = None,
    dim1: int = 0,
    dim2: int = 1,
) -> None:
    """
    Scatter plot of samples in 2D embedding space.
    """
    setup_style()

    if embedding.shape[1] < 2:
        print("  Skipping scatter: need at least 2 dimensions")
        return

    fig, ax = create_figure("square")

    x = embedding[:, dim1]
    y = embedding[:, dim2]

    ax.scatter(x, y, c=TEAL, s=30, alpha=0.7, edgecolor='white', linewidth=0.5)

    if labels is not None:
        for i, label in enumerate(labels):
            if x[i] > np.percentile(x, 90) or y[i] > np.percentile(y, 90):
                ax.annotate(label, (x[i], y[i]), fontsize=6, alpha=0.7)

    ax.set_xlabel(f'Dimension {dim1 + 1}')
    ax.set_ylabel(f'Dimension {dim2 + 1}')
    ax.set_title('Embedding space')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def visualize_dimensions(
    embedding: np.ndarray,
    embeddings: list[np.ndarray],
    linkage_matrix: np.ndarray,
    output_dir: Path,
    labels: list[str] | None = None,
) -> None:
    """
    Generate all dimension visualizations.

    Parameters
    ----------
    embedding : ndarray
        Selected/consensus embedding
    embeddings : list
        All embeddings from runs (for consistency analysis)
    linkage_matrix : ndarray
        From consensus hierarchical clustering
    output_dir : Path
    labels : list of str, optional
        Sample labels
    """
    print("\n=== Dimension Visualizations ===")

    print("  Plotting embedding heatmap...")
    plot_embedding_heatmap(embedding, linkage_matrix, output_dir / "embedding_heatmap.png")

    print("  Plotting dimension profiles...")
    plot_dimension_profiles(embedding, linkage_matrix, output_dir / "dimension_profiles.png")

    print("  Plotting dimension consistency...")
    plot_dimension_consistency(embeddings, linkage_matrix, output_dir / "dimension_consistency.png")

    print("  Plotting top samples per dimension...")
    plot_top_samples_per_dimension(embedding, output_dir / "top_samples.png", labels=labels)

    if embedding.shape[1] >= 2:
        print("  Plotting embedding scatter...")
        plot_embedding_scatter(embedding, output_dir / "embedding_scatter.png", labels=labels)
