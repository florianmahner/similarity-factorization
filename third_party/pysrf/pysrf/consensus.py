"""Consensus methods for combining multiple SRF runs.

Provides ensemble fitting with different random initializations and
consensus procedures to select or aggregate the most stable embedding.

Reference
---------
Mahner, F.P., Lam, K.C. & Hebart, M.N. Interpretable dimensions
from sparse representational similarities. In preparation.
"""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def l2_norm_columns(v: np.ndarray) -> np.ndarray:
    """L2-normalize each column, leaving zero columns unchanged.

    Parameters
    ----------
    v : ndarray of shape (n, k)

    Returns
    -------
    v_normed : ndarray of shape (n, k)
    """
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def hungarian_match(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find the column permutation that best aligns target to ref.

    Uses cosine similarity (ref and target should be L2-normalized)
    and the Hungarian algorithm for optimal one-to-one assignment.

    Parameters
    ----------
    ref : ndarray of shape (n, k), L2-normalized columns
    target : ndarray of shape (n, k), L2-normalized columns

    Returns
    -------
    permutation : ndarray of shape (k,)
        Column indices such that target[:, permutation] ~ ref
    """
    sim = ref.T @ target
    _, col_ind = linear_sum_assignment(-sim)
    return col_ind


def align_embeddings(embeddings: np.ndarray, reference_idx: int = 0) -> np.ndarray:
    """Align a set of embeddings to a reference via Hungarian matching.

    Parameters
    ----------
    embeddings : ndarray of shape (n_runs, n_samples, rank)
    reference_idx : int
        Which run to use as the alignment target

    Returns
    -------
    aligned : ndarray of shape (n_runs, n_samples, rank)
    """
    n_runs = embeddings.shape[0]
    norms = np.array([l2_norm_columns(e) for e in embeddings])
    ref = norms[reference_idx]

    aligned = np.empty_like(embeddings)
    for i in range(n_runs):
        if i == reference_idx:
            aligned[i] = embeddings[i]
        else:
            perm = hungarian_match(ref, norms[i])
            aligned[i] = embeddings[i][:, perm]

    return aligned


def pairwise_agreement(aligned: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine similarity per run, averaged over dimensions.

    Parameters
    ----------
    aligned : ndarray of shape (n_runs, n_samples, rank)
        Aligned embeddings (e.g. from align_embeddings)

    Returns
    -------
    scores : ndarray of shape (n_runs,)
        Higher values indicate more stable runs
    """
    n_runs, _, rank = aligned.shape
    scores = np.zeros(n_runs)

    for j in range(rank):
        col = aligned[:, :, j]
        norms = np.linalg.norm(col, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        col_normed = col / norms
        sim = col_normed @ col_normed.T
        np.fill_diagonal(sim, 0.0)
        scores += sim.sum(axis=1) / (n_runs - 1)

    scores /= rank
    return scores


# ---------------------------------------------------------------------------
# Pipeline classes
# ---------------------------------------------------------------------------


class EnsembleFit(BaseEstimator, TransformerMixin):
    """Fit the base estimator multiple times and stack the embeddings.

    Each run uses a different random seed, producing embeddings that
    differ only in column permutation. The stacked output has shape
    (n_samples, rank * n_runs) and can be passed to a consensus method.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Estimator with fit/transform (e.g. SRF)
    n_runs : int, default=50
        Number of independent runs
    random_state : int, default=0
        Base random seed (run i uses random_state + i)
    n_jobs : int, default=-1
        Number of parallel jobs

    Attributes
    ----------
    embeddings_ : ndarray of shape (n_samples, rank * n_runs)
        Horizontally stacked embeddings from all runs
    estimators_ : list of BaseEstimator
        Fitted estimators from each run
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_runs: int = 50,
        random_state: int = 0,
        n_jobs: int = -1,
    ):
        self.base_estimator = base_estimator
        self.n_runs = n_runs
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> EnsembleFit:
        """Fit n_runs independent estimators in parallel.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored

        Returns
        -------
        self : EnsembleFit
        """

        def _fit_one(seed):
            est = clone(self.base_estimator)
            est.set_params(random_state=seed)
            return est, est.fit_transform(x)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(self.random_state + i) for i in range(self.n_runs)
        )

        estimators, embeddings = zip(*results)
        self.estimators_ = list(estimators)
        self.embeddings_ = np.hstack(embeddings)
        return self

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return the stacked embeddings (x is ignored).

        Parameters
        ----------
        x : Ignored
        y : Ignored

        Returns
        -------
        embeddings : ndarray of shape (n_samples, rank * n_runs)
        """
        check_is_fitted(self, "embeddings_")
        return self.embeddings_


class ClusterConsensus(BaseEstimator, TransformerMixin):
    """Consensus via column clustering.

    Scales columns, clusters them using KMeans, and returns cluster
    representatives. The number of clusters is selected via silhouette
    score over a candidate range.

    Parameters
    ----------
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
    n_init : int, default=30
        Number of KMeans initializations
    column_scaling : str, default="l2"
        Column scaling before clustering:
        "none", "l2", "center_l2", or "zscore"
    silhouette_metric : str, default="cosine"
        Distance metric for silhouette score
    representative : str, default="mean"
        How to compute cluster representatives: "mean" or "median"
    renorm_output : bool, default=True
        Whether to L2-normalize output columns

    Attributes
    ----------
    best_k_ : int
        Optimal number of clusters
    labels_ : ndarray of shape (n_columns,)
        Cluster label for each input column
    cluster_results_ : DataFrame
        Silhouette scores for each candidate k
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 100,
        step: int = 2,
        random_state: int = 0,
        n_jobs: int = -1,
        n_init: int = 30,
        column_scaling: str = "l2",
        silhouette_metric: str = "cosine",
        representative: str = "mean",
        renorm_output: bool = True,
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.step = step
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_init = n_init
        self.column_scaling = column_scaling
        self.silhouette_metric = silhouette_metric
        self.representative = representative
        self.renorm_output = renorm_output

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> ClusterConsensus:
        """Select best k via silhouette score and cluster columns.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_columns)
            Stacked embeddings (e.g. from EnsembleFit)
        y : Ignored

        Returns
        -------
        self : ClusterConsensus
        """
        x = _validate_2d(x)
        x_t = self._scale_columns(x).T

        max_k = min(self.max_clusters, x_t.shape[0] - 1)
        if self.min_clusters > max_k:
            raise ValueError(
                f"min_clusters={self.min_clusters} exceeds feasible "
                f"maximum {max_k} for {x.shape[1]} columns"
            )

        def _score(k):
            km = KMeans(
                n_clusters=k, random_state=self.random_state, n_init=self.n_init
            )
            labels = km.fit_predict(x_t)
            sc = silhouette_score(x_t, labels, metric=self.silhouette_metric)
            return {"n_clusters": k, "silhouette_score": float(sc), "labels": labels}

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_score)(k) for k in range(self.min_clusters, max_k + 1, self.step)
        )
        self.cluster_results_ = pd.DataFrame(
            [
                {
                    "n_clusters": r["n_clusters"],
                    "silhouette_score": r["silhouette_score"],
                }
                for r in results
            ]
        )
        best_idx = self.cluster_results_["silhouette_score"].idxmax()
        self.best_k_ = int(self.cluster_results_.loc[best_idx, "n_clusters"])
        self.labels_ = results[best_idx]["labels"]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Merge columns within each cluster into representatives.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_columns)
        y : Ignored

        Returns
        -------
        merged : ndarray of shape (n_samples, best_k_)
            Sorted by cluster size (largest first)
        """
        check_is_fitted(self, "labels_")
        x = _validate_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {x.shape[1]}"
            )
        x_scaled = self._scale_columns(x)
        agg = np.median if self.representative == "median" else np.mean

        reps = []
        counts = []
        for k in range(self.best_k_):
            cols = x_scaled[:, self.labels_ == k]
            reps.append(
                agg(cols, axis=1) if cols.shape[1] > 0 else np.zeros(x.shape[0])
            )
            counts.append(cols.shape[1])

        merged = np.column_stack(reps)
        merged = merged[:, np.argsort(counts)[::-1]]

        if self.renorm_output:
            merged = l2_norm_columns(merged)
        return merged

    def _scale_columns(self, v: np.ndarray) -> np.ndarray:
        if self.column_scaling == "none":
            return v
        if self.column_scaling == "l2":
            return l2_norm_columns(v)
        if self.column_scaling == "center_l2":
            return l2_norm_columns(v - v.mean(axis=0, keepdims=True))
        if self.column_scaling == "zscore":
            std = v.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (v - v.mean(axis=0, keepdims=True)) / std
        raise ValueError(f"Unknown column_scaling='{self.column_scaling}'")


class AlignedConsensus(BaseEstimator, TransformerMixin):
    """Consensus via Hungarian alignment for symmetric NMF.

    Symmetric NMF produces the same dimensions across runs but in
    different order (permutation ambiguity, no rotation). This class
    aligns dimensions via the Hungarian algorithm and returns the
    most central run — the one closest to the element-wise median
    across all aligned runs.

    Parameters
    ----------
    rank : int
        Number of dimensions in each embedding

    Attributes
    ----------
    aligned_embeddings_ : ndarray of shape (n_runs, n_samples, rank)
        Embeddings after Hungarian alignment
    consensus_median_ : ndarray of shape (n_samples, rank)
        Element-wise median across aligned runs
    centrality_scores_ : ndarray of shape (n_runs,)
        Frobenius distance of each run to the consensus median
    agreement_scores_ : ndarray of shape (n_runs,)
        Mean pairwise cosine similarity per run (higher = more stable)
    selected_run_idx_ : int
        Index of the most central run

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from pysrf import SRF
    >>> from pysrf.consensus import EnsembleFit, AlignedConsensus
    >>>
    >>> pipeline = Pipeline([
    ...     ('ensemble', EnsembleFit(SRF(rank=10), n_runs=50)),
    ...     ('consensus', AlignedConsensus(rank=10))
    ... ])
    >>> embedding = pipeline.fit_transform(similarity_matrix)
    """

    def __init__(self, rank: int):
        self.rank = rank

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> AlignedConsensus:
        """Align embeddings and select the most central run.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_runs * rank)
            Stacked embeddings from EnsembleFit
        y : Ignored

        Returns
        -------
        self : AlignedConsensus
        """
        x = np.asarray(x)
        n_samples, n_total = x.shape
        n_runs = n_total // self.rank

        if n_total % self.rank != 0:
            raise ValueError(
                f"Number of columns ({n_total}) not divisible by rank ({self.rank})"
            )

        # (n_samples, n_runs, rank) -> (n_runs, n_samples, rank)
        embeddings = x.reshape(n_samples, n_runs, self.rank).transpose(1, 0, 2)

        self.aligned_embeddings_ = align_embeddings(embeddings, reference_idx=0)
        self.consensus_median_ = np.median(self.aligned_embeddings_, axis=0)

        diffs = self.aligned_embeddings_ - self.consensus_median_[np.newaxis]
        self.centrality_scores_ = np.sqrt(np.einsum("ijk,ijk->i", diffs, diffs))

        self.agreement_scores_ = pairwise_agreement(self.aligned_embeddings_)
        self.selected_run_idx_ = int(np.argmin(self.centrality_scores_))

        return self

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return the most central run.

        Parameters
        ----------
        x : Ignored
        y : Ignored

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
        """
        check_is_fitted(self, "aligned_embeddings_")
        return self.aligned_embeddings_[self.selected_run_idx_]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_2d(x: np.ndarray) -> np.ndarray:
    """Validate that x is a finite 2-d array."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Input must be 2-d")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or Inf")
    return x
