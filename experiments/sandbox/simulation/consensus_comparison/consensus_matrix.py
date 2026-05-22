"""
Consensus Matrix with Hierarchical Clustering (Brunet et al., 2004).

This implements the classic consensus clustering approach:
1. Run SRF multiple times with different seeds
2. Compute consensus matrix C where C_ij = fraction of runs where i,j clustered together
3. Hierarchical clustering on C to visualize stability
4. Cophenetic correlation coefficient to measure stability

References:
- Brunet et al. (2004) PNAS - introduced consensus matrix for NMF
- Monti et al. (2003) Machine Learning - general consensus clustering framework
- Kuang et al. (2012) SDM - symmetric NMF for graph clustering

Usage:
    ./scripts/submit sandbox/simulation/consensus_comparison/consensus_matrix.py --bg
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from pysrf.consensus import EnsembleEmbedding
from similarity import build_similarity
from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, setup_style, CYCLE
from src.utils.figure_theme import create_figure, despine
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def compute_connectivity_matrix(embedding: np.ndarray) -> np.ndarray:
    """
    Compute connectivity matrix from embedding.

    For each pair of samples, C_ij = 1 if they have the same dominant dimension.

    Parameters
    ----------
    embedding : ndarray of shape (n_samples, rank)
        Factor matrix W

    Returns
    -------
    connectivity : ndarray of shape (n_samples, n_samples)
        Binary connectivity matrix
    """
    # Assign each sample to its dominant dimension
    assignments = np.argmax(embedding, axis=1)

    # C_ij = 1 if samples i and j have the same dominant dimension
    n = len(assignments)
    connectivity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            connectivity[i, j] = 1.0 if assignments[i] == assignments[j] else 0.0

    return connectivity


def compute_consensus_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Compute consensus matrix by averaging connectivity matrices.

    Parameters
    ----------
    embeddings : list of ndarrays
        List of embeddings from multiple runs

    Returns
    -------
    consensus : ndarray of shape (n_samples, n_samples)
        Consensus matrix with values in [0, 1]
    """
    connectivity_matrices = [compute_connectivity_matrix(e) for e in embeddings]
    consensus = np.mean(connectivity_matrices, axis=0)
    return consensus


def compute_cophenetic_correlation(consensus: np.ndarray, linkage_matrix: np.ndarray) -> float:
    """
    Compute cophenetic correlation coefficient.

    Measures how well the hierarchical clustering preserves the pairwise
    distances in the consensus matrix. CCC close to 1 = stable clustering.

    Parameters
    ----------
    consensus : ndarray
        Consensus matrix
    linkage_matrix : ndarray
        Linkage matrix from hierarchical clustering

    Returns
    -------
    ccc : float
        Cophenetic correlation coefficient
    """
    # Distance matrix from consensus (1 - consensus since consensus is similarity)
    distance = 1 - consensus

    # Condensed distance matrix (upper triangle)
    condensed_dist = squareform(distance, checks=False)

    # Cophenetic distances from linkage
    coph_dist, _ = cophenet(linkage_matrix, condensed_dist)

    return coph_dist


def compute_dispersion_coefficient(consensus: np.ndarray) -> float:
    """
    Compute dispersion coefficient (Brunet et al., 2004).

    Measures how close consensus entries are to 0 or 1.
    High dispersion = unstable (many entries near 0.5).

    Parameters
    ----------
    consensus : ndarray
        Consensus matrix

    Returns
    -------
    dispersion : float
        Dispersion coefficient in [0, 1], lower = more stable
    """
    # Get upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(consensus, k=1)
    entries = consensus[triu_idx]

    # Dispersion = mean of 4 * p * (1 - p), which is 0 at p=0,1 and 1 at p=0.5
    dispersion = np.mean(4 * entries * (1 - entries))

    return dispersion


def plot_consensus_heatmap(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
    title: str = "Consensus Matrix",
) -> None:
    """Plot consensus matrix as heatmap with dendrogram."""
    setup_style()

    # Reorder by hierarchical clustering
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    consensus_ordered = consensus[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(consensus_ordered, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')

    ax.set_xlabel('Samples (reordered)')
    ax.set_ylabel('Samples (reordered)')
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Co-clustering frequency')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_consensus_histogram(consensus: np.ndarray, output_path: Path) -> None:
    """Plot histogram of consensus matrix entries."""
    setup_style()

    # Get upper triangle entries
    triu_idx = np.triu_indices_from(consensus, k=1)
    entries = consensus[triu_idx]

    fig, ax = create_figure("single")

    ax.hist(entries, bins=50, color=TEAL, edgecolor='white', linewidth=0.5)
    ax.axvline(0.5, color=ROSE, linestyle='--', linewidth=1.5, label='Unstable threshold')

    ax.set_xlabel('Consensus value')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    labels: list[str] | None = None,
    k: int | None = None,
    title: str = "Dendrogram",
) -> dict:
    """
    Plot dendrogram from hierarchical clustering.

    Parameters
    ----------
    linkage_matrix : ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage
    output_path : Path
        Where to save the plot
    labels : list of str, optional
        Labels for leaf nodes (sample names)
    k : int, optional
        If provided, draw horizontal line at cut point for k clusters
    title : str
        Plot title

    Returns
    -------
    dendro : dict
        Dendrogram data including 'leaves' (ordering) and 'color_list'
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute cut threshold if k is provided
    color_threshold = 0
    if k is not None and k > 1:
        # Find the distance threshold that gives k clusters
        # This is the (n-k)th merge distance
        n = linkage_matrix.shape[0] + 1
        if k <= n:
            color_threshold = linkage_matrix[-(k), 2]

    dendro = dendrogram(
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
    print(f"  Saved: {output_path.name}")

    return dendro


def extract_cluster_assignments(
    linkage_matrix: np.ndarray,
    k: int,
    labels: list[str] | None = None,
) -> dict:
    """
    Extract cluster assignments by cutting dendrogram at k clusters.

    Parameters
    ----------
    linkage_matrix : ndarray
        Linkage matrix from hierarchical clustering
    k : int
        Number of clusters
    labels : list of str, optional
        Sample labels

    Returns
    -------
    result : dict
        'assignments': array of cluster IDs (1 to k)
        'clusters': dict mapping cluster ID to list of sample indices/labels
    """
    # Cut tree to get k clusters
    assignments = fcluster(linkage_matrix, k, criterion='maxclust')

    n = len(assignments)
    if labels is None:
        labels = [str(i) for i in range(n)]

    # Group samples by cluster
    clusters = {}
    for cluster_id in range(1, k + 1):
        mask = assignments == cluster_id
        clusters[cluster_id] = {
            'indices': np.where(mask)[0].tolist(),
            'labels': [labels[i] for i in np.where(mask)[0]],
            'size': int(np.sum(mask)),
        }

    return {
        'assignments': assignments,
        'clusters': clusters,
    }


def compute_soft_memberships(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute soft cluster memberships from consensus matrix.

    For each sample, compute average consensus with other samples
    in each cluster. This gives a probabilistic membership.

    Parameters
    ----------
    consensus : ndarray of shape (n, n)
        Consensus matrix
    linkage_matrix : ndarray
        Linkage matrix for cluster assignments
    k : int
        Number of clusters

    Returns
    -------
    memberships : ndarray of shape (n, k)
        Soft membership probabilities (rows sum to 1)
    """
    assignments = fcluster(linkage_matrix, k, criterion='maxclust')
    n = consensus.shape[0]

    memberships = np.zeros((n, k))

    for i in range(n):
        for cluster_id in range(1, k + 1):
            # Average consensus with members of this cluster (excluding self)
            cluster_members = np.where(assignments == cluster_id)[0]
            cluster_members = cluster_members[cluster_members != i]

            if len(cluster_members) > 0:
                memberships[i, cluster_id - 1] = np.mean(consensus[i, cluster_members])

    # Normalize to sum to 1
    row_sums = memberships.sum(axis=1, keepdims=True)
    memberships = memberships / np.maximum(row_sums, 1e-10)

    return memberships


def plot_soft_memberships(
    memberships: np.ndarray,
    linkage_matrix: np.ndarray,
    output_path: Path,
    labels: list[str] | None = None,
) -> None:
    """Plot soft membership matrix as stacked bar chart."""
    setup_style()

    # Reorder by dendrogram
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    memberships_ordered = memberships[order]

    n, k = memberships.shape

    fig, ax = plt.subplots(figsize=(12, 4))

    # Stacked bar chart
    x = np.arange(n)
    bottom = np.zeros(n)

    colors = [CYCLE[i % len(CYCLE)] for i in range(k)]

    for cluster_id in range(k):
        ax.bar(x, memberships_ordered[:, cluster_id], bottom=bottom,
               width=1.0, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')
        bottom += memberships_ordered[:, cluster_id]

    ax.set_xlabel('Samples (reordered by dendrogram)')
    ax.set_ylabel('Membership probability')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc='upper right')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def analyze_hierarchical_granularity(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    cv_rank: int,
    output_dir: Path,
    max_k: int = 10,
) -> pd.DataFrame:
    """
    Analyze sub-structure in consensus matrix beyond CV-optimal rank.

    Even though CV says k=2 is optimal, the consensus matrix may reveal
    finer structure - "variant dimensions" that some runs find.

    Parameters
    ----------
    consensus : ndarray
        Consensus matrix
    linkage_matrix : ndarray
        Linkage matrix from hierarchical clustering
    cv_rank : int
        CV-selected optimal rank
    output_dir : Path
        Where to save outputs
    max_k : int
        Maximum k to analyze

    Returns
    -------
    granularity_df : DataFrame
        Metrics at each k level
    """
    setup_style()
    n = consensus.shape[0]

    results = []
    all_assignments = {}

    for k in range(2, min(max_k + 1, n)):
        assignments = fcluster(linkage_matrix, k, criterion='maxclust')
        all_assignments[k] = assignments

        # Compute within-cluster consensus (higher = more stable)
        within_consensus = []
        for cluster_id in range(1, k + 1):
            members = np.where(assignments == cluster_id)[0]
            if len(members) > 1:
                # Average pairwise consensus within cluster
                cluster_consensus = consensus[np.ix_(members, members)]
                triu = cluster_consensus[np.triu_indices(len(members), k=1)]
                within_consensus.append(np.mean(triu))

        # Compute between-cluster consensus (lower = better separation)
        between_consensus = []
        for c1 in range(1, k + 1):
            for c2 in range(c1 + 1, k + 1):
                members1 = np.where(assignments == c1)[0]
                members2 = np.where(assignments == c2)[0]
                if len(members1) > 0 and len(members2) > 0:
                    cross = consensus[np.ix_(members1, members2)]
                    between_consensus.append(np.mean(cross))

        # Cluster sizes
        sizes = [np.sum(assignments == c) for c in range(1, k + 1)]

        results.append({
            'k': k,
            'mean_within_consensus': np.mean(within_consensus) if within_consensus else 0,
            'mean_between_consensus': np.mean(between_consensus) if between_consensus else 0,
            'separation': (np.mean(within_consensus) - np.mean(between_consensus))
                         if within_consensus and between_consensus else 0,
            'min_cluster_size': min(sizes),
            'cluster_sizes': sizes,
        })

    df = pd.DataFrame(results)

    # Plot granularity analysis
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Within vs between consensus
    ax = axes[0]
    ax.plot(df['k'], df['mean_within_consensus'], 'o-', color=TEAL,
            label='Within-cluster', markersize=6)
    ax.plot(df['k'], df['mean_between_consensus'], 'o-', color=ROSE,
            label='Between-cluster', markersize=6)
    ax.axvline(cv_rank, color=GRAY, linestyle='--', label=f'CV rank={cv_rank}')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Mean consensus')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Cluster cohesion vs separation')
    despine(ax)

    # Separation score
    ax = axes[1]
    ax.plot(df['k'], df['separation'], 'o-', color=CYAN, markersize=6)
    ax.axvline(cv_rank, color=GRAY, linestyle='--')
    ax.axhline(0, color=GRAY_LIGHT, linestyle='-', linewidth=0.5)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Separation (within - between)')
    ax.set_title('Higher = better defined clusters')
    despine(ax)

    # Minimum cluster size
    ax = axes[2]
    ax.plot(df['k'], df['min_cluster_size'], 'o-', color=SAND, markersize=6)
    ax.axvline(cv_rank, color=GRAY, linestyle='--')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Smallest cluster size')
    ax.set_title('Cluster balance')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / 'granularity_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: granularity_analysis.png")

    # Plot alluvial-style diagram showing how clusters split
    plot_cluster_splits(all_assignments, cv_rank, output_dir)

    return df


def plot_cluster_splits(
    all_assignments: dict[int, np.ndarray],
    cv_rank: int,
    output_dir: Path,
) -> None:
    """
    Show how clusters split as k increases.

    This reveals which sub-groups emerge beyond the CV-optimal rank.
    """
    setup_style()

    # Show k from cv_rank to cv_rank + 4
    k_range = [k for k in range(cv_rank, cv_rank + 5) if k in all_assignments]
    if len(k_range) < 2:
        return

    n = len(all_assignments[k_range[0]])

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each k, plot samples as horizontal bars colored by cluster
    y_positions = np.arange(n)

    # Reorder samples by their assignment trajectory
    # Sort by (assignment at k=cv_rank, then k=cv_rank+1, etc.)
    sort_keys = np.column_stack([all_assignments[k] for k in k_range])
    order = np.lexsort(sort_keys.T[::-1])

    bar_width = 0.8 / len(k_range)

    for i, k in enumerate(k_range):
        assignments = all_assignments[k][order]
        x_offset = i * bar_width

        for cluster_id in range(1, k + 1):
            mask = assignments == cluster_id
            color = CYCLE[(cluster_id - 1) % len(CYCLE)]
            ax.barh(y_positions[mask], bar_width, left=x_offset,
                    color=color, edgecolor='white', linewidth=0.2)

    ax.set_yticks([])
    ax.set_xticks([i * bar_width + bar_width/2 for i in range(len(k_range))])
    ax.set_xticklabels([f'k={k}' for k in k_range])
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(f'Samples (n={n}, reordered)')
    ax.set_title(f'Cluster splits beyond CV-optimal k={cv_rank}')

    # Highlight CV-optimal
    ax.axvline(bar_width/2, color='black', linestyle='-', linewidth=2, alpha=0.3)

    despine(ax)
    plt.tight_layout()
    fig.savefig(output_dir / 'cluster_splits.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: cluster_splits.png")


def plot_stability_vs_rank(
    results: list[dict],
    output_path: Path,
) -> None:
    """Plot stability metrics vs rank."""
    setup_style()

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Cophenetic correlation
    ax = axes[0]
    ax.plot(df['rank'], df['cophenetic_corr'], 'o-', color=TEAL, markersize=6)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cophenetic correlation')
    ax.set_title('Cluster stability')
    ax.set_ylim(0, 1)
    despine(ax)

    # Dispersion
    ax = axes[1]
    ax.plot(df['rank'], df['dispersion'], 'o-', color=ROSE, markersize=6)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Dispersion')
    ax.set_title('Lower = more stable')
    ax.set_ylim(0, 1)
    despine(ax)

    # Reconstruction error
    ax = axes[2]
    ax.plot(df['rank'], df['mean_recon_error'], 'o-', color=CYAN, markersize=6)
    ax.fill_between(
        df['rank'],
        df['mean_recon_error'] - df['std_recon_error'],
        df['mean_recon_error'] + df['std_recon_error'],
        color=CYAN, alpha=0.2
    )
    ax.set_xlabel('Rank')
    ax.set_ylabel('Reconstruction error')
    ax.set_title('Mean ± std across runs')
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def run_consensus_analysis(
    similarity: np.ndarray,
    rank: int,
    n_runs: int = 50,
    n_jobs: int = -1,
) -> dict:
    """
    Run consensus analysis for a single rank.

    Parameters
    ----------
    similarity : ndarray
        Similarity matrix
    rank : int
        Number of dimensions
    n_runs : int
        Number of SRF runs
    n_jobs : int
        Parallel jobs

    Returns
    -------
    result : dict
        Dictionary with consensus matrix and metrics
    """
    print(f"  Running {n_runs} SRF fits with rank={rank}...")

    # Run ensemble
    ensemble = EnsembleEmbedding(
        base_estimator=SRF(rank=rank, max_outer=100, max_inner=30),
        n_runs=n_runs,
        n_jobs=n_jobs,
    )
    stacked = ensemble.fit_transform(similarity)

    # Extract individual embeddings
    n_samples = similarity.shape[0]
    embeddings = [
        stacked[:, i*rank:(i+1)*rank]
        for i in range(n_runs)
    ]

    # Compute reconstruction errors
    recon_errors = []
    for emb in embeddings:
        recon = emb @ emb.T
        error = np.linalg.norm(similarity - recon, 'fro') / np.linalg.norm(similarity, 'fro')
        recon_errors.append(error)

    # Compute consensus matrix
    consensus = compute_consensus_matrix(embeddings)

    # Hierarchical clustering on consensus
    distance = 1 - consensus
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method='average')

    # Compute metrics
    ccc = compute_cophenetic_correlation(consensus, linkage_matrix)
    dispersion = compute_dispersion_coefficient(consensus)

    return {
        'rank': rank,
        'consensus': consensus,
        'linkage': linkage_matrix,
        'embeddings': embeddings,
        'cophenetic_corr': ccc,
        'dispersion': dispersion,
        'mean_recon_error': np.mean(recon_errors),
        'std_recon_error': np.std(recon_errors),
        'recon_errors': recon_errors,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    # Load dataset - use Mur92 (small, fast)
    from omegaconf import OmegaConf

    dataset_cfg = OmegaConf.create({
        'name': 'mur92',
        'type': 'neural_rsm',
        'path': '/SSD/datasets/similarity_datasets/mur92',
        'bounds_task': 'mur92',
        'rank_range': [1, 20, 1],
    })

    print("Loading Mur92 similarity matrix...")
    similarity = build_similarity(dataset_cfg)
    print(f"Shape: {similarity.shape}")

    n_samples = similarity.shape[0]
    n_runs = 50
    # Fixed rank from CV rank selection (Mur92 optimal_rank=2)
    ranks = [2]

    print(f"\nRunning consensus stability analysis at FIXED rank:")
    print(f"  n_samples: {n_samples}")
    print(f"  n_runs: {n_runs}")
    print(f"  rank: {ranks[0]} (from CV rank selection)")
    print()

    results = []

    for rank in ranks:
        print(f"\n=== Rank {rank} ===")
        result = run_consensus_analysis(similarity, rank, n_runs=n_runs, n_jobs=40)
        results.append(result)

        # Plot consensus heatmap for this rank
        plot_consensus_heatmap(
            result['consensus'],
            result['linkage'],
            OUTPUT_DIR / f"consensus_heatmap_k{rank}.png",
            title=f"Consensus Matrix (k={rank}, CCC={result['cophenetic_corr']:.3f})"
        )

        # Plot histogram
        plot_consensus_histogram(
            result['consensus'],
            OUTPUT_DIR / f"consensus_histogram_k{rank}.png"
        )

        print(f"  Cophenetic correlation: {result['cophenetic_corr']:.3f}")
        print(f"  Dispersion: {result['dispersion']:.3f}")
        print(f"  Reconstruction error: {result['mean_recon_error']:.4f} ± {result['std_recon_error']:.4f}")

    # Skip stability_vs_rank plot when only one rank
    if len(ranks) > 1:
        plot_stability_vs_rank(
            [{'rank': r['rank'],
              'cophenetic_corr': r['cophenetic_corr'],
              'dispersion': r['dispersion'],
              'mean_recon_error': r['mean_recon_error'],
              'std_recon_error': r['std_recon_error']}
             for r in results],
            OUTPUT_DIR / "stability_vs_rank.png"
        )

    # Save summary
    summary_df = pd.DataFrame([
        {
            'rank': r['rank'],
            'cophenetic_corr': r['cophenetic_corr'],
            'dispersion': r['dispersion'],
            'mean_recon_error': r['mean_recon_error'],
            'std_recon_error': r['std_recon_error'],
        }
        for r in results
    ])
    summary_df.to_csv(OUTPUT_DIR / "consensus_summary.csv", index=False)
    print(f"\nSaved summary to consensus_summary.csv")

    print("\n=== Stability Assessment ===")
    print(summary_df.to_string(index=False))

    # Interpret stability
    cv_rank = ranks[0]
    ccc = results[0]['cophenetic_corr']
    disp = results[0]['dispersion']
    if ccc > 0.9:
        stability = "STABLE - runs converge to same solution"
    elif ccc > 0.7:
        stability = "MODERATE - some variability across runs"
    else:
        stability = "UNSTABLE - runs find different solutions"
    print(f"\nStability: {stability}")

    # === NEW: Hierarchical granularity analysis ===
    print("\n=== Hierarchical Granularity Analysis ===")
    print(f"Looking for sub-structure beyond CV-optimal k={cv_rank}...")

    consensus = results[0]['consensus']
    linkage_matrix = results[0]['linkage']

    # Plot full dendrogram
    print("\n  Plotting dendrogram...")
    plot_dendrogram(
        linkage_matrix,
        OUTPUT_DIR / "dendrogram.png",
        labels=None,  # Could add object names if available
        k=cv_rank,
        title=f"Hierarchical Clustering (cut at k={cv_rank})"
    )

    # Analyze granularity at multiple k
    granularity_df = analyze_hierarchical_granularity(
        consensus, linkage_matrix, cv_rank, OUTPUT_DIR, max_k=8
    )
    granularity_df.to_csv(OUTPUT_DIR / "granularity_metrics.csv", index=False)
    print(f"\n  Granularity metrics saved to granularity_metrics.csv")

    # Extract and save cluster assignments at CV rank and beyond
    print(f"\n=== Cluster Assignments ===")
    for k in range(cv_rank, min(cv_rank + 4, n_samples)):
        cluster_result = extract_cluster_assignments(linkage_matrix, k)
        print(f"\n  k={k}:")
        for cid, info in cluster_result['clusters'].items():
            print(f"    Cluster {cid}: {info['size']} samples (indices {info['indices'][:5]}{'...' if info['size'] > 5 else ''})")

        # Save assignments
        assign_df = pd.DataFrame({
            'sample_idx': range(n_samples),
            f'cluster_k{k}': cluster_result['assignments']
        })
        assign_df.to_csv(OUTPUT_DIR / f"assignments_k{k}.csv", index=False)

    # Compute and plot soft memberships
    print("\n  Computing soft memberships...")
    soft_memberships = compute_soft_memberships(consensus, linkage_matrix, cv_rank)
    plot_soft_memberships(soft_memberships, linkage_matrix,
                          OUTPUT_DIR / "soft_memberships.png")

    # Save soft memberships
    membership_df = pd.DataFrame(
        soft_memberships,
        columns=[f'cluster_{i+1}_prob' for i in range(cv_rank)]
    )
    membership_df['sample_idx'] = range(n_samples)
    membership_df['entropy'] = -np.sum(
        soft_memberships * np.log(soft_memberships + 1e-10), axis=1
    )
    membership_df = membership_df[['sample_idx', 'entropy'] +
                                   [f'cluster_{i+1}_prob' for i in range(cv_rank)]]
    membership_df.to_csv(OUTPUT_DIR / "soft_memberships.csv", index=False)

    # Identify boundary samples (high entropy)
    high_entropy = membership_df.nlargest(10, 'entropy')
    print(f"\n  Top 10 'boundary' samples (highest membership entropy):")
    print(high_entropy[['sample_idx', 'entropy']].to_string(index=False))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
