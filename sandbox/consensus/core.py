"""
Core consensus matrix computation (Brunet et al., 2004).

Functions for computing connectivity matrices, consensus matrices,
and stability metrics from multiple SRF runs.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform

from pysrf import SRF
from pysrf.consensus import EnsembleEmbedding


def compute_connectivity_matrix(embedding: np.ndarray) -> np.ndarray:
    """
    Compute connectivity matrix from embedding.

    C_ij = 1 if samples i and j have the same dominant dimension.

    Parameters
    ----------
    embedding : ndarray of shape (n_samples, rank)

    Returns
    -------
    connectivity : ndarray of shape (n_samples, n_samples)
    """
    assignments = np.argmax(embedding, axis=1)
    n = len(assignments)
    connectivity = (assignments[:, None] == assignments[None, :]).astype(float)
    return connectivity


def compute_consensus_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Compute consensus matrix by averaging connectivity matrices.

    Parameters
    ----------
    embeddings : list of ndarrays
        Embeddings from multiple runs

    Returns
    -------
    consensus : ndarray of shape (n_samples, n_samples)
        Values in [0, 1] indicating co-clustering frequency
    """
    connectivity_matrices = [compute_connectivity_matrix(e) for e in embeddings]
    return np.mean(connectivity_matrices, axis=0)


def compute_cophenetic_correlation(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
) -> float:
    """
    Compute cophenetic correlation coefficient.

    Measures how well hierarchical clustering preserves pairwise distances.
    CCC close to 1 = stable clustering.
    """
    distance = 1 - consensus
    np.fill_diagonal(distance, 0)
    distance_condensed = squareform(distance, checks=False)
    ccc, _ = cophenet(linkage_matrix, distance_condensed)
    return ccc


def compute_dispersion(consensus: np.ndarray) -> float:
    """
    Compute dispersion coefficient.

    Dispersion = mean(4 * p * (1-p)), which is:
    - 0 when all entries are 0 or 1 (stable)
    - 1 when all entries are 0.5 (unstable)
    """
    triu_idx = np.triu_indices_from(consensus, k=1)
    entries = consensus[triu_idx]
    return np.mean(4 * entries * (1 - entries))


def run_ensemble(
    similarity: np.ndarray,
    rank: int,
    n_runs: int = 50,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """
    Run SRF ensemble and return individual embeddings.

    Parameters
    ----------
    similarity : ndarray
        Similarity matrix
    rank : int
        Number of dimensions
    n_runs : int
        Number of runs with different seeds
    n_jobs : int
        Parallel jobs

    Returns
    -------
    embeddings : list of ndarrays
        List of (n_samples, rank) embeddings
    """
    ensemble = EnsembleEmbedding(
        base_estimator=SRF(rank=rank, max_outer=100, max_inner=30),
        n_runs=n_runs,
        n_jobs=n_jobs,
    )
    stacked = ensemble.fit_transform(similarity)

    n_samples = similarity.shape[0]
    embeddings = [stacked[:, i * rank:(i + 1) * rank] for i in range(n_runs)]
    return embeddings


def run_consensus_analysis(
    similarity: np.ndarray,
    rank: int,
    n_runs: int = 50,
    n_jobs: int = -1,
) -> dict:
    """
    Run full consensus analysis for a given rank.

    Returns
    -------
    result : dict with keys:
        - consensus: consensus matrix
        - linkage: hierarchical clustering linkage
        - embeddings: list of embeddings from each run
        - cophenetic_corr: CCC metric
        - dispersion: dispersion coefficient
        - recon_errors: reconstruction error per run
    """
    print(f"  Running {n_runs} SRF fits with rank={rank}...")

    embeddings = run_ensemble(similarity, rank, n_runs, n_jobs)

    # Reconstruction errors
    recon_errors = []
    for emb in embeddings:
        recon = emb @ emb.T
        error = np.linalg.norm(similarity - recon, 'fro') / np.linalg.norm(similarity, 'fro')
        recon_errors.append(error)

    # Consensus matrix
    consensus = compute_consensus_matrix(embeddings)

    # Hierarchical clustering
    distance = 1 - consensus
    np.fill_diagonal(distance, 0)
    distance_condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(distance_condensed, method='average')

    # Metrics
    ccc = compute_cophenetic_correlation(consensus, linkage_matrix)
    dispersion = compute_dispersion(consensus)

    return {
        'rank': rank,
        'consensus': consensus,
        'linkage': linkage_matrix,
        'embeddings': embeddings,
        'cophenetic_corr': ccc,
        'dispersion': dispersion,
        'mean_recon_error': np.mean(recon_errors),
        'std_recon_error': np.std(recon_errors),
    }


def extract_cluster_assignments(
    linkage_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """Cut dendrogram at k clusters, return assignments (1 to k)."""
    return fcluster(linkage_matrix, k, criterion='maxclust')


def compute_soft_memberships(
    consensus: np.ndarray,
    linkage_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute soft cluster memberships from consensus matrix.

    For each sample, average consensus with members of each cluster.

    Returns
    -------
    memberships : ndarray of shape (n_samples, k)
        Rows sum to 1
    """
    assignments = fcluster(linkage_matrix, k, criterion='maxclust')
    n = consensus.shape[0]

    memberships = np.zeros((n, k))
    for i in range(n):
        for cluster_id in range(1, k + 1):
            members = np.where(assignments == cluster_id)[0]
            members = members[members != i]
            if len(members) > 0:
                memberships[i, cluster_id - 1] = np.mean(consensus[i, members])

    # Normalize
    row_sums = memberships.sum(axis=1, keepdims=True)
    memberships = memberships / np.maximum(row_sums, 1e-10)
    return memberships
