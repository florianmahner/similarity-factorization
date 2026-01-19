"""Consensus methods for combining multiple SRF runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
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


class AlignedConsensus(BaseEstimator, TransformerMixin):
    """
    Consensus embedding using Hungarian alignment for symmetric NMF.

    For symmetric NMF (like SRF), different runs produce the same dimensions
    but in different order (permutation ambiguity only, no rotation). This class:
    1. Aligns dimensions across runs using Hungarian matching on cosine similarity
    2. Computes the median consensus to identify the "central" solution
    3. Either returns the most central run or aggregates via median/mean

    IMPORTANT: For symmetric NMF, use aggregation="select" (default) which returns
    the most central run. This preserves valid factorization (WW^T ≈ S) while
    being most representative. Median/mean aggregation breaks the factorization
    structure and should only be used for visualization or stability analysis.

    Why "select" is best for symmetric NMF:
    - All runs have nearly identical reconstruction error
    - Most central run has best top-k agreement with consensus
    - Preserves valid factorization structure (unlike median/mean)
    - Reduces variance while maintaining interpretability

    Note: K-means clustering (used in cNMF for standard NMF) is NOT appropriate
    for symmetric NMF - it's 2-4x worse because it ignores permutation structure.

    Parameters
    ----------
    rank : int
        Number of dimensions (k) in each embedding
    aggregation : str, default="refine"
        How to produce final embedding:
        - "refine": median consensus + NNLS refinement (RECOMMENDED - best of both)
        - "select": return the most central run (valid factorization)
        - "median": element-wise median (breaks factorization, use for viz only)
        - "mean": element-wise mean (breaks factorization)
    outlier_threshold : float or None, default=None
        MAD threshold for outlier detection. Set to None to disable.
        Estimates with deviation > threshold * MAD are excluded.
    reference : str, default="first"
        How to select reference run for alignment:
        - "first": use first run
        - "median_recon": use run with median reconstruction error
    random_state : int, default=0
        Random seed for reproducibility

    Attributes
    ----------
    aligned_embeddings_ : ndarray of shape (n_runs, n_samples, rank)
        Embeddings after Hungarian alignment
    consensus_median_ : ndarray of shape (n_samples, rank)
        Median consensus (for reference, even when using "select")
    centrality_scores_ : ndarray of shape (n_runs,)
        Distance of each run to consensus (lower = more central)
    agreement_scores_ : ndarray of shape (n_runs,)
        Mean cosine similarity of each run with all others (higher = more central)
    selected_run_idx_ : int
        Index of the most central run
    outlier_mask_ : ndarray of shape (n_runs, rank)
        Boolean mask where True indicates outlier estimate
    n_outliers_per_dim_ : ndarray of shape (rank,)
        Number of outliers detected per dimension

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from pysrf import SRF
    >>> from pysrf.consensus import EnsembleEmbedding, AlignedConsensus
    >>>
    >>> # Recommended: select most central run
    >>> pipeline = Pipeline([
    ...     ('ensemble', EnsembleEmbedding(SRF(rank=10), n_runs=50)),
    ...     ('consensus', AlignedConsensus(rank=10))  # default: aggregation="select"
    ... ])
    >>> embedding = pipeline.fit_transform(similarity_matrix)
    >>>
    >>> # Access stability metrics
    >>> consensus = pipeline.named_steps['consensus']
    >>> print(f"Selected run: {consensus.selected_run_idx_}")
    >>> print(f"Centrality scores: {consensus.centrality_scores_}")
    """

    def __init__(
        self,
        rank: int,
        aggregation: str = "refine",
        outlier_threshold: float | None = None,
        reference: str = "first",
        random_state: int = 0,
        refine_iterations: int = 20,
    ):
        self.rank = rank
        self.aggregation = aggregation
        self.outlier_threshold = outlier_threshold
        self.reference = reference
        self.random_state = random_state
        self.refine_iterations = refine_iterations

    def fit(self, x: ndarray, y: ndarray | None = None) -> AlignedConsensus:
        """
        Fit the consensus model.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_runs * rank)
            Stacked embeddings from EnsembleEmbedding
        y : None
            Ignored

        Returns
        -------
        self : AlignedConsensus
        """
        x = np.asarray(x)
        n_samples, n_total = x.shape
        n_runs = n_total // self.rank

        if n_total % self.rank != 0:
            raise ValueError(
                f"Number of columns ({n_total}) must be divisible by rank ({self.rank})"
            )

        # Reshape to (n_runs, n_samples, rank)
        embeddings = x.reshape(n_samples, n_runs, self.rank).transpose(1, 0, 2)

        # L2-normalize columns of each embedding
        embeddings_norm = np.array([self._l2_norm_columns(e) for e in embeddings])

        # Select reference
        ref_idx = self._select_reference(embeddings_norm)
        reference = embeddings_norm[ref_idx]

        # Align all embeddings to reference using Hungarian matching
        self.aligned_embeddings_ = np.zeros_like(embeddings)
        for i, emb in enumerate(embeddings):
            if i == ref_idx:
                self.aligned_embeddings_[i] = emb
            else:
                perm = self._hungarian_match(reference, embeddings_norm[i])
                self.aligned_embeddings_[i] = emb[:, perm]

        # Detect outliers per dimension using MAD
        if self.outlier_threshold is not None:
            self.outlier_mask_ = self._detect_outliers()
            self.n_outliers_per_dim_ = self.outlier_mask_.sum(axis=0)
        else:
            n_runs = self.aligned_embeddings_.shape[0]
            self.outlier_mask_ = np.zeros((n_runs, self.rank), dtype=bool)
            self.n_outliers_per_dim_ = np.zeros(self.rank, dtype=int)

        # Compute consensus median (always computed for reference)
        self.consensus_median_ = np.median(self.aligned_embeddings_, axis=0)

        # Compute centrality (distance to consensus)
        self.centrality_scores_ = np.array([
            np.linalg.norm(e - self.consensus_median_, 'fro')
            for e in self.aligned_embeddings_
        ])

        # Compute agreement scores (mean cosine similarity with other runs)
        self.agreement_scores_ = self._compute_agreement_scores()

        # Identify the most central run (always computed)
        self.selected_run_idx_ = int(np.argmin(self.centrality_scores_))

        # For refinement mode, compute average reconstruction as target
        if self.aggregation == "refine":
            self._target_similarity_ = np.mean([
                e @ e.T for e in self.aligned_embeddings_
            ], axis=0)

        return self

    def transform(self, x: ndarray, y: ndarray | None = None) -> ndarray:
        """
        Compute consensus embedding.

        Parameters
        ----------
        x : ndarray
            Input (ignored, uses stored aligned embeddings)
        y : None
            Ignored

        Returns
        -------
        consensus : ndarray of shape (n_samples, rank)
            Consensus embedding based on aggregation method
        """
        check_is_fitted(self, "aligned_embeddings_")

        # Selection mode: return the most agreed run directly
        if self.aggregation == "select":
            return self.aligned_embeddings_[self.selected_run_idx_]

        # Refine mode: median consensus + NNLS refinement
        if self.aggregation == "refine":
            return self._nnls_refine(
                self.consensus_median_,
                self._target_similarity_,
                n_iter=self.refine_iterations,
            )

        n_runs, n_samples, rank = self.aligned_embeddings_.shape
        consensus = np.zeros((n_samples, rank))

        agg_func = np.median if self.aggregation == "median" else np.mean

        for j in range(rank):
            # Get all estimates for dimension j
            dim_estimates = self.aligned_embeddings_[:, :, j]  # (n_runs, n_samples)

            # Remove outliers
            good_mask = ~self.outlier_mask_[:, j]
            good_estimates = dim_estimates[good_mask]

            if len(good_estimates) == 0:
                # Fallback: use all if all are outliers
                consensus[:, j] = agg_func(dim_estimates, axis=0)
            else:
                consensus[:, j] = agg_func(good_estimates, axis=0)

        return consensus

    def _select_reference(self, embeddings: ndarray) -> int:
        """Select reference embedding for alignment."""
        if self.reference == "first":
            return 0
        elif self.reference == "median_recon":
            # Select run with median reconstruction error
            recons = [e @ e.T for e in embeddings]
            mean_recon = np.mean(recons, axis=0)
            errors = [np.linalg.norm(r - mean_recon, 'fro') for r in recons]
            return int(np.argsort(errors)[len(errors) // 2])
        else:
            raise ValueError(f"Unknown reference method: {self.reference}")

    def _hungarian_match(self, ref: ndarray, target: ndarray) -> ndarray:
        """
        Find optimal permutation to align target to reference.

        Uses Hungarian algorithm on cosine similarity matrix.

        Parameters
        ----------
        ref : ndarray of shape (n_samples, rank)
            Reference embedding (L2-normalized)
        target : ndarray of shape (n_samples, rank)
            Target embedding to align (L2-normalized)

        Returns
        -------
        permutation : ndarray of shape (rank,)
            Optimal column permutation for target
        """
        # Cosine similarity between all pairs of dimensions
        # ref.T @ target gives (rank, rank) similarity matrix
        sim = ref.T @ target  # Already L2-normalized, so this is cosine sim

        # Hungarian algorithm minimizes cost, so negate similarity
        row_ind, col_ind = linear_sum_assignment(-sim)

        return col_ind

    def _detect_outliers(self) -> ndarray:
        """
        Detect outlier estimates per dimension using MAD.

        Returns
        -------
        outlier_mask : ndarray of shape (n_runs, rank)
            True where estimate is an outlier
        """
        n_runs, n_samples, rank = self.aligned_embeddings_.shape
        outlier_mask = np.zeros((n_runs, rank), dtype=bool)

        for j in range(rank):
            # Get all estimates for dimension j: (n_runs, n_samples)
            dim_estimates = self.aligned_embeddings_[:, :, j]

            # Compute median estimate
            median_est = np.median(dim_estimates, axis=0)

            # Compute distance of each estimate from median (cosine distance)
            # Normalize for cosine comparison
            dim_norm = dim_estimates / (
                np.linalg.norm(dim_estimates, axis=1, keepdims=True) + 1e-10
            )
            median_norm = median_est / (np.linalg.norm(median_est) + 1e-10)

            # Cosine similarity to median
            cos_sim = dim_norm @ median_norm

            # Convert to distance (1 - similarity)
            distances = 1 - cos_sim

            # MAD-based outlier detection
            med_dist = np.median(distances)
            mad = median_abs_deviation(distances)

            if mad > 0:
                # Deviation from median in MAD units
                deviation = np.abs(distances - med_dist) / mad
                outlier_mask[:, j] = deviation > self.outlier_threshold
            # If MAD is 0, all estimates are identical, no outliers

        return outlier_mask

    def _compute_agreement_scores(self) -> ndarray:
        """
        Compute agreement score for each run.

        Agreement is the mean cosine similarity of each run's dimensions
        with all other runs' corresponding dimensions (after alignment).

        Returns
        -------
        scores : ndarray of shape (n_runs,)
            Agreement score for each run (higher = more central/agreed upon)
        """
        n_runs, n_samples, rank = self.aligned_embeddings_.shape
        scores = np.zeros(n_runs)

        for i in range(n_runs):
            total_sim = 0.0
            count = 0
            for j in range(rank):
                emb_i = self.aligned_embeddings_[i, :, j]
                norm_i = np.linalg.norm(emb_i)
                if norm_i < 1e-10:
                    continue
                emb_i_normed = emb_i / norm_i

                for k in range(n_runs):
                    if k == i:
                        continue
                    emb_k = self.aligned_embeddings_[k, :, j]
                    norm_k = np.linalg.norm(emb_k)
                    if norm_k < 1e-10:
                        continue
                    emb_k_normed = emb_k / norm_k
                    total_sim += np.dot(emb_i_normed, emb_k_normed)
                    count += 1

            scores[i] = total_sim / count if count > 0 else 0.0

        return scores

    @staticmethod
    def _l2_norm_columns(v: ndarray) -> ndarray:
        """L2-normalize columns of a matrix."""
        n = np.linalg.norm(v, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    @staticmethod
    def _nnls_refine(W_init: ndarray, S: ndarray, n_iter: int = 20) -> ndarray:
        """
        Refine embedding using NNLS to match target similarity.

        Averaging embeddings breaks the factorization structure (WW^T ≈ S).
        This method projects the averaged embedding back to a valid factorization
        by iteratively solving NNLS problems row-wise.

        Parameters
        ----------
        W_init : ndarray of shape (n_samples, rank)
            Initial embedding (typically median consensus)
        S : ndarray of shape (n_samples, n_samples)
            Target similarity matrix (typically average of WW^T from all runs)
        n_iter : int, default=20
            Number of refinement iterations

        Returns
        -------
        W : ndarray of shape (n_samples, rank)
            Refined embedding where WW^T ≈ S
        """
        W = np.maximum(W_init.copy(), 0)
        n, k = W.shape

        for _ in range(n_iter):
            for i in range(n):
                w_new, _ = nnls(W, S[i, :])
                W[i, :] = w_new

        return W


