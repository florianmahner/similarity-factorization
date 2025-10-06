"""Consensus methods for combining multiple SRF runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted

ndarray = np.ndarray


class EnsembleEmbedding(BaseEstimator, TransformerMixin):
    """
    Ensemble embedding from multiple runs with different initializations.

    Fits the base estimator multiple times with different random seeds and
    stacks the resulting embeddings horizontally.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Estimator to run multiple times (should have .fit() and .transform())
    n_runs : int, default=50
        Number of independent runs
    random_state : int, default=0
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs

    Attributes
    ----------
    embeddings_ : ndarray of shape (n_samples, n_features * n_runs)
        Stacked embeddings from all runs
    estimators_ : list of estimators
        Fitted estimators from each run

    Examples
    --------
    >>> from pysrf import SRF
    >>> from pysrf.consensus import EnsembleEmbedding
    >>> model = SRF(rank=10)
    >>> ensemble = EnsembleEmbedding(model, n_runs=50)
    >>> embeddings = ensemble.fit_transform(similarity_matrix)
    >>> embeddings.shape
    (1000, 500)  # 10 * 50
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

    def fit(self, x: ndarray, y: ndarray | None = None) -> EnsembleEmbedding:
        """
        Fit multiple instances of the base estimator.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : None
            Ignored, present for sklearn compatibility

        Returns
        -------
        self : EnsembleEmbedding
            Fitted estimator
        """

        def _fit_one(seed: int) -> tuple[BaseEstimator, ndarray]:
            est = clone(self.base_estimator)
            if hasattr(est, "random_state"):
                est.set_params(random_state=seed)
            est.fit(x)
            embedding = est.transform(x)
            return est, embedding

        seeds = [self.random_state + i for i in range(self.n_runs)]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(seed) for seed in seeds
        )

        self.estimators_, embeddings = zip(*results)
        self.embeddings_ = np.hstack(embeddings)

        return self

    def transform(self, x: ndarray, y: ndarray | None = None) -> ndarray:
        """
        Return stacked consensus embeddings.

        Parameters
        ----------
        x : ndarray
            Input data (ignored, returns stored embeddings)
        y : None
            Ignored

        Returns
        -------
        embeddings : ndarray of shape (n_samples, n_features * n_runs)
            Stacked embeddings
        """
        check_is_fitted(self, "embeddings_")
        return self.embeddings_


class ClusterEmbedding(BaseEstimator, TransformerMixin):
    """
    Reduce embedding dimensionality via clustering.

    Takes stacked embeddings, normalizes columns, clusters them, and returns
    merged cluster representatives. The optimal number of clusters is selected
    via silhouette score.

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

    Attributes
    ----------
    best_k_ : int
        Optimal number of clusters selected
    labels_ : ndarray
        Cluster labels for each column
    cluster_results_ : DataFrame
        Silhouette scores for each k tried

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from pysrf import SRF
    >>> from pysrf.consensus import EnsembleEmbedding, ClusterEmbedding
    >>>
    >>> pipeline = Pipeline([
    ...     ('ensemble', EnsembleEmbedding(SRF(rank=10), n_runs=50)),
    ...     ('cluster', ClusterEmbedding(min_clusters=10, max_clusters=30))
    ... ])
    >>> final_embedding = pipeline.fit_transform(similarity_matrix)
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 100,
        step: int = 2,
        random_state: int = 0,
        n_jobs: int = -1,
        n_init: int = 30,
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.step = step
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_init = n_init

    def fit(self, x: ndarray, y: ndarray | None = None) -> ClusterEmbedding:
        """
        Find optimal number of clusters and fit KMeans.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Stacked embeddings to cluster
        y : None
            Ignored

        Returns
        -------
        self : ClusterEmbedding
            Fitted estimator
        """
        x_norm = self._norm_columns(x)
        x_t = x_norm.T

        max_k = min(self.max_clusters, x_t.shape[0] - 1)
        cluster_range = range(self.min_clusters, max_k + 1, self.step)

        def _score_k(k: int) -> dict:
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                init="k-means++",
                n_init=self.n_init,
            )
            labels = km.fit_predict(x_t)
            score = silhouette_score(x_t, labels)
            return {"n_clusters": k, "silhouette_score": float(score)}

        records = Parallel(n_jobs=self.n_jobs)(
            delayed(_score_k)(k) for k in cluster_range
        )

        self.cluster_results_ = pd.DataFrame(records)
        self.best_k_ = int(
            self.cluster_results_.loc[
                self.cluster_results_["silhouette_score"].idxmax(), "n_clusters"
            ]
        )

        km = KMeans(
            n_clusters=self.best_k_,
            random_state=self.random_state,
            init="k-means++",
            n_init=self.n_init,
        )
        self.labels_ = km.fit_predict(x_t)
        self.x_input_ = x

        return self

    def transform(self, x: ndarray, y: ndarray | None = None) -> ndarray:
        """
        Return merged cluster representatives.

        Parameters
        ----------
        x : ndarray
            Input (ignored, uses stored data from fit)
        y : None
            Ignored

        Returns
        -------
        merged : ndarray of shape (n_samples, best_k)
            Cluster representatives sorted by cluster size
        """
        check_is_fitted(self, "best_k_")

        merged = []
        sizes = []
        for i in range(self.best_k_):
            cluster_cols = self.x_input_[:, self.labels_ == i]
            merged.append(np.median(cluster_cols, axis=1))
            sizes.append(cluster_cols.sum())

        merged = np.column_stack(merged)
        order = np.argsort(sizes)[::-1]
        return merged[:, order]

    @staticmethod
    def _norm_columns(v: ndarray) -> ndarray:
        """Normalize columns to unit length."""
        n = np.linalg.norm(v, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return v / n
