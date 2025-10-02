"""
Cross-validation utilities for matrix completion with symmetric matrices.

This module provides specialized cross-validation tools for symmetric non-negative
matrix factorization, including entry-wise masking strategies and grid search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Generator
from .model import SRF


def mask_missing_entries(
    x: np.ndarray,
    sampling_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    """
    Create a missing mask for symmetric matrix cross-validation.

    Subsample from valid upper triangular positions to keep as observed,
    maintaining symmetry. The diagonal is always observed to ensure proper
    scaling of the optimization.

    Parameters
    ----------
    x : ndarray
        Input symmetric matrix
    sampling_fraction : float
        Fraction of entries to keep as observed (0.0 to 1.0)
    rng : RandomState
        Random number generator
    missing_values : float or None
        Value that marks missing entries

    Returns
    -------
    missing_mask : ndarray of bool
        Boolean mask where True indicates missing entries
    """
    observed_mask = ~np.isnan(x) if missing_values is np.nan else x != missing_values

    triu_i, triu_j = np.triu_indices_from(x, k=1)
    triu_observed = observed_mask[triu_i, triu_j]
    valid_positions = np.where(triu_observed)[0]

    n_to_keep = int(sampling_fraction * len(valid_positions))
    if n_to_keep == 0:
        return np.ones_like(x, dtype=bool)

    # Subsample from valid upper triangular positions to keep as observed
    keep_positions = rng.choice(valid_positions, size=n_to_keep, replace=False)

    # Create missing mask directly - start with all missing
    missing_mask = np.ones_like(x, dtype=bool)

    # Set kept positions as observed (False = not missing)
    keep_i = triu_i[keep_positions]
    keep_j = triu_j[keep_positions]
    missing_mask[keep_i, keep_j] = False
    missing_mask[keep_j, keep_i] = False

    # Diagonal is always observed to ensure proper scaling of the optimization
    np.fill_diagonal(missing_mask, False)

    return missing_mask


def fit_and_score(
    estimator: BaseEstimator,
    x: np.ndarray,
    val_mask: np.ndarray,
    fit_params: dict,
    split_idx: int | None = None,
) -> dict:
    """
    Fit estimator with parameters and return validation score.

    Parameters
    ----------
    estimator : BaseEstimator
        Model instance to fit
    x : ndarray
        Full data matrix
    val_mask : ndarray
        Boolean mask indicating validation entries
    fit_params : dict
        Parameters to set on the estimator
    split_idx : int or None
        Index of the CV split

    Returns
    -------
    result : dict
        Dictionary with score, parameters, and fitted estimator
    """
    est = clone(estimator).set_params(**fit_params)
    est.set_params(missing_values=np.nan)

    # IMPORTANT If bounds not explicitly provided, compute from original data, not just
    # training portion
    if "bounds" not in fit_params or fit_params["bounds"] is None:
        original_bounds = (np.nanmin(x), np.nanmax(x))
        est.set_params(bounds=original_bounds)

    # we consider true nan entries as missing, so we need to mask them out in the
    # reconstruction entirely, not being able to treat them as artificial missing values.
    already_nan = np.isnan(x)

    x_copy = np.copy(x)
    x_copy[val_mask] = np.nan

    reconstruction = est.fit(x_copy).reconstruct()
    valid_mask = val_mask & ~already_nan
    mse = np.mean((x[valid_mask] - reconstruction[valid_mask]) ** 2)

    result = {
        "score": mse,
        "split": split_idx if split_idx is not None else 0,
        "estimator": est,
        "params": fit_params,
        "history": est.history_,
    }
    return result


class EntryMaskSplit(BaseCrossValidator):
    """
    Cross-validator for symmetric matrices using entry-wise splits.

    Generates multiple random train/validation splits by masking entries
    in a symmetric matrix while preserving symmetry.

    Parameters
    ----------
    n_repeats : int, default=5
        Number of random splits to generate
    sampling_fraction : float, default=0.8
        Fraction of entries to use for training
    random_state : int or None, default=None
        Random seed for reproducibility
    missing_values : float or None, default=np.nan
        Value that marks missing entries
    """

    def __init__(
        self,
        n_repeats: int = 5,
        sampling_fraction: float = 0.8,
        random_state: int | None = None,
        missing_values: float | None = np.nan,
    ):
        self.n_repeats = n_repeats
        self.sampling_fraction = sampling_fraction
        self.random_state = random_state
        self.missing_values = missing_values

    def get_n_splits(
        self, x: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        return self.n_repeats

    def split(
        self, x: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            yield mask_missing_entries(
                x, self.sampling_fraction, rng, self.missing_values
            )


class ADMMGridSearchCV:
    """
    Grid search cross-validation for matrix completion.

    Performs exhaustive grid search over specified parameter values with
    entry-wise cross-validation for symmetric matrices.

    Parameters
    ----------
    estimator : BaseEstimator
        Model instance to optimize
    param_grid : dict
        Dictionary with parameter names as keys and lists of values to try
    cv : EntryMaskSplit
        Cross-validation splitter
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    verbose : int, default=0
        Verbosity level
    fit_final_estimator : bool, default=False
        Whether to fit the model on full data with best parameters

    Attributes
    ----------
    best_params_ : dict
        Parameters that gave the best score
    best_score_ : float
        Best validation score achieved
    cv_results_ : DataFrame
        Detailed results for all parameter combinations
    best_estimator_ : estimator
        Fitted estimator with best parameters (if fit_final_estimator=True)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        cv: EntryMaskSplit,
        n_jobs: int = -1,
        verbose: int = 0,
        fit_final_estimator: bool = False,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fit_final_estimator = fit_final_estimator

    def fit(self, x: np.ndarray) -> "ADMMGridSearchCV":
        param_grid = ParameterGrid(self.param_grid)
        cv_splits = list(self.cv.split(x))

        all_jobs = [
            (params, val_mask, split_idx)
            for split_idx, val_mask in enumerate(cv_splits)
            for params in param_grid
        ]

        if self.verbose:
            print(
                f"Running {len(all_jobs)} jobs ({len(param_grid)} params x {len(cv_splits)} CV splits)"
            )

        all_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fit_and_score)(self.estimator, x, val_mask, params, split_idx)
            for params, val_mask, split_idx in all_jobs
        )

        df = pd.DataFrame([{**r["params"], "score": r["score"]} for r in all_results])
        param_cols = [col for col in df.columns if col != "score"]
        mean_scores = df.groupby(param_cols)["score"].mean()

        best_params_idx = mean_scores.idxmin()
        self.best_params_ = dict(
            zip(
                param_cols,
                best_params_idx if len(param_cols) > 1 else [best_params_idx],
            )
        )
        self.best_score_ = mean_scores.min()

        self.cv_results_ = pd.DataFrame(
            [
                {
                    **{"score": result["score"], "split": result["split"]},
                    **result["params"],
                }
                for result in all_results
            ]
        )

        if self.fit_final_estimator:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(x)

        return self


def cross_val_score(
    similarity_matrix: np.ndarray,
    param_grid: dict[str, list] | None = None,
    n_repeats: int = 5,
    sampling_fraction: float = 0.8,
    estimate_sampling_fraction: bool | dict = False,
    random_state: int = 0,
    verbose: int = 1,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    fit_final_estimator: bool = False,
    n_stable_runs: int = 1,
    cluster_stable: bool | dict = False,
) -> ADMMGridSearchCV:
    """
    Cross-validate SRF parameters for matrix completion.

    Convenience function that sets up grid search cross-validation for
    symmetric matrix factorization with optional automatic sampling fraction estimation.

    Parameters
    ----------
    similarity_matrix : ndarray
        Symmetric similarity matrix to cross-validate
    param_grid : dict or None, default=None
        Dictionary with parameter names (str) as keys and lists of values to try
        as values. If None, uses default {'rank': [5, 10, 15, 20]}
    n_repeats : int, default=5
        Number of times to repeat the cross-validation
    sampling_fraction : float, default=0.8
        Fraction of observed entries to use for training in each split (0.0 to 1.0).
        Ignored if estimate_sampling_fraction is True or dict
    estimate_sampling_fraction : bool or dict, default=False
        If True, automatically estimate optimal sampling fraction using sampling
        bound estimation from Random Matrix Theory. If dict, passed as kwargs to
        estimate_sampling_bounds_fast(). When enabled, overrides sampling_fraction
    random_state : int, default=0
        Random seed for reproducibility
    verbose : int, default=1
        Verbosity level
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 uses all processors)
    missing_values : float or None, default=np.nan
        Value to consider as missing
    fit_final_estimator : bool, default=False
        Whether to fit the final estimator on the best parameters
    n_stable_runs : int, default=1
        Number of independent runs with different initializations for stable embeddings.
        Only used if fit_final_estimator=True and n_stable_runs > 1.
    cluster_stable : bool or dict, default=False
        Whether to cluster the stable embeddings. Only used if fit_final_estimator=True
        and n_stable_runs > 1. If dict, passed as kwargs to cluster_stable().
        Common kwargs: min_clusters, max_clusters, step, n_init

    Returns
    -------
    grid : ADMMGridSearchCV
        Fitted ADMMGridSearchCV object with best parameters and scores.
        If n_stable_runs > 1, additional attributes:
            - stable_embeddings_ : stacked embeddings (n_samples, rank * n_runs)
            - clustered_embedding_ : clustered embedding (n_samples, best_k) if cluster_stable=True
            - cluster_results_ : DataFrame with clustering results if cluster_stable=True
            - best_k_ : optimal number of clusters if cluster_stable=True

    Examples
    --------
    >>> from pysrf import cross_val_score
    >>> param_grid = {'rank': [5, 10, 15], 'rho': [2.0, 3.0]}
    >>> result = cross_val_score(similarity_matrix, param_grid=param_grid)
    >>> print(f"Best params: {result.best_params_}")
    >>> print(f"Best score: {result.best_score_}")

    >>> # Automatic sampling fraction estimation
    >>> result = cross_val_score(
    ...     similarity_matrix,
    ...     param_grid={'rank': [5, 10, 15]},
    ...     estimate_sampling_fraction=True
    ... )

    >>> # Complete pipeline: CV + stable ensemble + clustering
    >>> result = cross_val_score(
    ...     similarity_matrix,
    ...     param_grid={'rank': list(range(5, 50))},
    ...     estimate_sampling_fraction=True,
    ...     fit_final_estimator=True,
    ...     n_stable_runs=50,
    ...     cluster_stable=True
    ... )
    >>> final_embedding = result.clustered_embedding_
    >>> print(f"Best rank: {result.best_params_['rank']}, Best k: {result.best_k_}")
    """
    if param_grid is None:
        param_grid = {"rank": [5, 10, 15, 20]}

    if estimate_sampling_fraction:
        from .bounds import estimate_sampling_bounds_fast

        kwargs = (
            estimate_sampling_fraction
            if isinstance(estimate_sampling_fraction, dict)
            else {}
        )
        if "random_state" not in kwargs:
            kwargs["random_state"] = random_state
        if "n_jobs" not in kwargs:
            kwargs["n_jobs"] = n_jobs
        if "verbose" not in kwargs:
            kwargs["verbose"] = bool(verbose)

        pmin, pmax, _ = estimate_sampling_bounds_fast(similarity_matrix, **kwargs)
        sampling_fraction = 0.5 * (pmin + pmax)

        if verbose:
            print(f"Estimated sampling bounds: pmin={pmin:.4f}, pmax={pmax:.4f}")
            print(f"Using sampling_fraction={sampling_fraction:.4f}")

    cv = EntryMaskSplit(
        n_repeats=n_repeats,
        sampling_fraction=sampling_fraction,
        random_state=random_state,
        missing_values=missing_values,
    )
    grid = ADMMGridSearchCV(
        estimator=SRF(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_final_estimator=fit_final_estimator,
    )
    grid.fit(similarity_matrix)

    if fit_final_estimator and n_stable_runs > 1:
        from .stability import fit_stable, cluster_stable

        if verbose:
            print(
                f"Fitting {n_stable_runs} stable runs with rank={grid.best_params_['rank']}"
            )

        srf_params = {k: v for k, v in grid.best_params_.items() if k != "rank"}

        grid.stable_embeddings_ = fit_stable(
            similarity_matrix,
            rank=grid.best_params_["rank"],
            n_runs=n_stable_runs,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **srf_params,
        )

        if cluster_stable:
            if verbose:
                print("Clustering stable embeddings")

            cluster_kwargs = cluster_stable if isinstance(cluster_stable, dict) else {}
            if "random_state" not in cluster_kwargs:
                cluster_kwargs["random_state"] = random_state
            if "n_jobs" not in cluster_kwargs:
                cluster_kwargs["n_jobs"] = n_jobs
            if "verbose" not in cluster_kwargs:
                cluster_kwargs["verbose"] = verbose

            (
                grid.clustered_embedding_,
                grid.best_k_,
                grid.cluster_results_,
            ) = cluster_stable(grid.stable_embeddings_, **cluster_kwargs)

    return grid
