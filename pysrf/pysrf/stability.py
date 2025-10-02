"""Stability-based ensemble methods for SRF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .model import SRF

ndarray = np.ndarray


def fit_stable(
    similarity_matrix: ndarray,
    rank: int,
    n_runs: int = 50,
    random_state: int = 0,
    n_jobs: int = -1,
    stack: bool = True,
    verbose: int = 0,
    **srf_kwargs,
) -> ndarray | list[ndarray]:
    """
    Fit multiple SRF models with different initializations for stable embeddings.

    This function fits multiple SRF models with different random seeds to obtain
    stable embeddings. The embeddings can be stacked horizontally for downstream
    clustering or returned as a list.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix to factorize
    rank : int
        Number of latent dimensions
    n_runs : int, default=50
        Number of independent runs with different initializations
    random_state : int, default=0
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    stack : bool, default=True
        If True, stack embeddings horizontally (n_samples, rank * n_runs).
        If False, return list of (n_samples, rank) arrays.
    srf_kwargs : dict, optional
        Additional keyword arguments for SRF.

    Returns
    -------
    embeddings : ndarray or list[ndarray]
        If stack=True: (n_samples, rank * n_runs) stacked embeddings
        If stack=False: list of n_runs arrays of shape (n_samples, rank)

    Examples
    --------
    >>> from pysrf.stability import fit_stable
    >>> embeddings = fit_stable(similarity_matrix, rank=20, n_runs=50)
    >>> embeddings.shape
    (1000, 1000)  # 20 * 50
    """

    def _fit(seed: int) -> ndarray:
        model = SRF(
            rank=rank,
            random_state=seed,
            **srf_kwargs,
        )
        return model.fit_transform(similarity_matrix)

    seeds = [random_state + i for i in range(n_runs)]
    embeddings = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit)(seed) for seed in seeds
    )

    if stack:
        return np.hstack(embeddings)
    return embeddings


def _norm_columns(v: ndarray) -> ndarray:
    """Normalize columns to unit length."""
    n = np.linalg.norm(v, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _merge_clusters_by_median(x: ndarray, labels: ndarray, k: int) -> ndarray:
    """Merge clusters by taking median and sorting by cluster size."""
    med = np.array([np.median(x[:, labels == i], axis=1) for i in range(k)]).T
    sums = np.array([np.sum(x[:, labels == i]) for i in range(k)])
    order = np.argsort(-sums)
    return med[:, order]


def cluster_stable(
    stacked_embeddings: ndarray,
    min_clusters: int = 2,
    max_clusters: int = 100,
    step: int = 2,
    random_state: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
    **cluster_kwargs,
) -> tuple[ndarray, int, pd.DataFrame]:
    """
    Cluster stacked embeddings using KMeans with silhouette-based selection.

    This function normalizes the stacked embeddings, searches over cluster counts,
    selects the best number of clusters using silhouette score, and returns
    merged cluster representatives.

    Parameters
    ----------
    stacked_embeddings : ndarray of shape (n_samples, rank * n_runs)
        Stacked embeddings from multiple SRF runs
    min_clusters : int, default=2
        Minimum number of clusters to try
    max_clusters : int, default=100
        Maximum number of clusters to try
    step : int, default=2
        Step size for cluster search
    random_state : int, default=0
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=0
        Verbosity level
    **cluster_kwargs : dict
        Additional arguments for KMeans (e.g., n_init, max_iter)

    Returns
    -------
    final_embedding : ndarray of shape (n_samples, best_k)
        Final embedding with best_k cluster representatives
    best_k : int
        Optimal number of clusters selected
    cluster_results : DataFrame
        Results of cluster search with columns: n_clusters, silhouette_score, best_k

    Examples
    --------
    >>> from pysrf.stability import fit_stable, cluster_stable
    >>> stacked = fit_stable(similarity_matrix, rank=20, n_runs=50)
    >>> final, k, results = cluster_stable(stacked, min_clusters=10, max_clusters=50)
    >>> final.shape
    (1000, 30)  # selected 30 clusters
    """
    x = _norm_columns(stacked_embeddings).T
    n_samples = x.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    min_clusters = min(min_clusters, max_clusters)
    cluster_range = range(min_clusters, max_clusters + 1, step)

    kmeans_kwargs = {"random_state": random_state, "init": "k-means++", "n_init": 30}
    if cluster_kwargs:
        kmeans_kwargs.update(cluster_kwargs)

    def _score_k(k: int) -> dict:
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        km.fit(x)
        labels = km.labels_
        return {
            "n_clusters": k,
            "silhouette_score": float(silhouette_score(x, labels)),
        }

    records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_score_k)(k) for k in cluster_range
    )

    df = pd.DataFrame(records)
    best_k = int(df.iloc[np.argmax(df["silhouette_score"])]["n_clusters"])

    km = KMeans(n_clusters=best_k, **kmeans_kwargs)
    labels = km.fit(x).labels_
    final_embedding = _merge_clusters_by_median(stacked_embeddings, labels, best_k)

    df["best_k"] = best_k
    return final_embedding, best_k, df
