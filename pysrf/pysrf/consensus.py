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

    Takes stacked embeddings, scales columns, clusters them, and returns
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
    column_scaling : str, default="l2"
        Column scaling method. Options: "none", "l2", "center_l2", "zscore"
        - "none": no scaling
        - "l2": normalize to unit length
        - "center_l2": center then normalize
        - "zscore": standardize (mean=0, std=1)
    silhouette_metric : str, default="cosine"
        Distance metric for silhouette score. Better aligned with l2-normalized
        columns. See sklearn.metrics.silhouette_score for options
    representative : str, default="mean"
        Method to compute cluster representative. Options: "mean", "median"
    renorm_output : bool, default=True
        Whether to renormalize output columns to unit length

    Attributes
    ----------
    best_k_ : int
        Optimal number of clusters selected
    labels_ : ndarray
        Cluster labels for each column
    cluster_results_ : DataFrame
        Silhouette scores for each k tried
    n_features_in_ : int
        Number of features seen during fit

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

    def fit(self, x: ndarray, y: ndarray | None = None) -> ClusterEmbedding:
        x = self._validate_x(x)
        x_scaled = self._scale_columns(x)
        x_t = x_scaled.T

        max_k = max(2, min(self.max_clusters, x_t.shape[0] - 1))
        if self.min_clusters > max_k:
            raise ValueError(
                f"min_clusters={self.min_clusters} exceeds feasible maximum {max_k} "
                f"for n_columns={x.shape[1]} (silhouette needs at least 2 and at most n-1)."
            )
        cluster_range = range(self.min_clusters, max_k + 1, self.step)

        def score_k(k: int) -> dict:
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                init="k-means++",
                n_init=self.n_init,
            )
            labels = km.fit_predict(x_t)
            sc = silhouette_score(x_t, labels, metric=self.silhouette_metric)
            return {"n_clusters": k, "silhouette_score": float(sc)}

        records = Parallel(n_jobs=self.n_jobs)(
            delayed(score_k)(k) for k in cluster_range
        )
        self.cluster_results_ = pd.DataFrame(records)
        best_idx = self.cluster_results_["silhouette_score"].idxmax()
        self.best_k_ = int(self.cluster_results_.loc[best_idx, "n_clusters"])

        km = KMeans(
            n_clusters=self.best_k_,
            random_state=self.random_state,
            init="k-means++",
            n_init=self.n_init,
        )
        self.labels_ = km.fit_predict(x_t)
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: ndarray, y: ndarray | None = None) -> ndarray:
        check_is_fitted(self, ("best_k_", "labels_"))
        x = self._validate_x(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} columns but model was fit with {self.n_features_in_}."
            )
        x_scaled = self._scale_columns(x)

        agg_func = np.median if self.representative == "median" else np.mean
        reps = []
        counts = []

        for i in range(self.best_k_):
            mask = self.labels_ == i
            cols = x_scaled[:, mask]
            if cols.shape[1] == 0:
                reps.append(np.zeros(x.shape[0], dtype=x.dtype))
                counts.append(0)
            else:
                reps.append(agg_func(cols, axis=1))
                counts.append(cols.shape[1])

        merged = np.column_stack(reps)
        order = np.argsort(counts)[::-1]
        merged = merged[:, order]

        if self.renorm_output:
            merged = self._l2_norm_columns(merged)
        return merged

    # helpers

    def _validate_x(self, x: ndarray) -> ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must be a 2d array (n_samples, n_features).")
        if not np.isfinite(x).all():
            raise ValueError("x contains nan or inf.")
        return x

    def _scale_columns(self, v: ndarray) -> ndarray:
        if self.column_scaling == "none":
            return v

        if self.column_scaling == "l2":
            return self._l2_norm_columns(v)

        if self.column_scaling == "center_l2":
            centered = v - v.mean(axis=0, keepdims=True)
            return self._l2_norm_columns(centered)

        if self.column_scaling == "zscore":
            mean = v.mean(axis=0, keepdims=True)
            std = v.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (v - mean) / std

        raise ValueError(f"unknown column_scaling='{self.column_scaling}'")

    @staticmethod
    def _l2_norm_columns(v: ndarray) -> ndarray:
        n = np.linalg.norm(v, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return v / n
