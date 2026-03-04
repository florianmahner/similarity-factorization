"""
Cross-validation utilities for matrix completion with symmetric matrices.

This module provides specialized cross-validation tools for symmetric non-negative
matrix factorization, including entry-wise masking strategies and grid search.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from typing import Generator, Tuple
from .model import SRF


def create_train_val_split(
    x: np.ndarray,
    sampling_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/validation split for symmetric matrix cross-validation.

    Splits entries into three categories:
    1. Training: used to fit the model
    2. Validation: held out to evaluate the model
    3. Excluded: not used at all (e.g., constant diagonals)

    Parameters
    ----------
    x : ndarray
        Input symmetric matrix
    sampling_fraction : float
        Fraction of eligible entries kept for training; must be in (0, 1).
        The remaining (1 - sampling_fraction) becomes validation.
    rng : RandomState
        Random number generator
    missing_values : float or None
        Value that marks missing entries in the original data

    Returns
    -------
    train_mask : ndarray of bool
        True indicates training entry (used to fit model)
    validation_mask : ndarray of bool
        True indicates validation entry (held out for evaluation)

    Notes
    -----
    Entries where both masks are False are excluded from CV entirely.
    This happens for constant diagonal entries (e.g., all 1s in RBF kernels).
    """
    # Find entries that are observed in the original data
    originally_observed = (
        ~np.isnan(x) if missing_values is np.nan else x != missing_values
    )

    # Initialize both masks as False (excluded by default)
    train_mask = np.zeros_like(x, dtype=bool)
    validation_mask = np.zeros_like(x, dtype=bool)

    # Work with upper triangle (excluding diagonal for now)
    triu_i, triu_j = np.triu_indices_from(x, k=1)
    triu_observed = originally_observed[triu_i, triu_j]
    eligible_positions = np.where(triu_observed)[0]

    if len(eligible_positions) == 0:
        return train_mask, validation_mask

    # Determine how many to keep for training
    n_train = int(sampling_fraction * len(eligible_positions))
    n_train = max(1, n_train)  # At least one training sample

    # Randomly select which positions go to training vs validation
    train_positions = rng.choice(eligible_positions, size=n_train, replace=False)
    val_positions = np.setdiff1d(eligible_positions, train_positions)

    # Set training positions
    train_i = triu_i[train_positions]
    train_j = triu_j[train_positions]
    train_mask[train_i, train_j] = True
    train_mask[train_j, train_i] = True

    val_i = triu_i[val_positions]
    val_j = triu_j[val_positions]
    validation_mask[val_i, val_j] = True
    validation_mask[val_j, val_i] = True

    # Handle diagonal entries based on whether they're constant or variable
    diagonal_values = x.diagonal()

    if np.allclose(diagonal_values, diagonal_values[0]):
        # Constant diagonal (e.g., all 1s in RBF kernel)
        # EXCLUDE from CV entirely: neither train nor validation
        # Both masks remain False for diagonal
        pass
    else:
        # Variable diagonal (e.g., in linear kernel)
        # Include in CV sampling just like off-diagonal entries
        n_samples = x.shape[0]
        diag_indices = np.arange(n_samples)

        # Sample diagonal entries proportionally
        n_diag_train = int(sampling_fraction * n_samples)
        n_diag_train = max(1, n_diag_train)  # At least one training sample

        train_diag_indices = rng.choice(diag_indices, size=n_diag_train, replace=False)
        val_diag_indices = np.setdiff1d(diag_indices, train_diag_indices)

        # Set diagonal training entries
        train_mask[train_diag_indices, train_diag_indices] = True

        # Set diagonal validation entries
        validation_mask[val_diag_indices, val_diag_indices] = True

    return train_mask, validation_mask


def fit_and_score(
    estimator: BaseEstimator,
    x: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    fit_params: dict,
    split_idx: int | None = None,
) -> dict:
    """
    Fit estimator with parameters and return validation score.

    Parameters
    ----------
    estimator : BaseEstimator
        Model instance to fit (works with SRF or any estimator with .reconstruct())
    x : ndarray
        Full data matrix
    train_mask : ndarray of bool
        Boolean mask where True = training entry
    validation_mask : ndarray of bool
        Boolean mask where True = validation entry
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

    # Set SRF-specific params if estimator supports them
    if hasattr(est, "missing_values"):
        est.set_params(missing_values=np.nan)

    if hasattr(est, "bounds"):
        if "bounds" not in fit_params or fit_params["bounds"] is None:
            original_bounds = (np.nanmin(x), np.nanmax(x))
            est.set_params(bounds=original_bounds)

    # Track which entries were already NaN in the original data
    originally_nan = np.isnan(x)

    # Create training data: keep only training entries, mask everything else
    x_train = np.full_like(x, np.nan)
    x_train[train_mask] = x[train_mask]

    # Fit model on training data only
    est.fit(x_train)

    # Get reconstruction
    if hasattr(est, "reconstruct"):
        reconstruction = est.reconstruct()
    else:
        raise ValueError(
            f"Estimator {type(est).__name__} must have a .reconstruct() method "
            "for matrix completion cross-validation"
        )

    # Evaluate only on validation entries that were originally observed
    valid_eval_mask = validation_mask & ~originally_nan

    if not valid_eval_mask.any():
        raise ValueError("No valid validation entries to evaluate")

    mse = np.mean((x[valid_eval_mask] - reconstruction[valid_eval_mask]) ** 2)

    result = {
        "score": mse,
        "split": split_idx if split_idx is not None else 0,
        "estimator": est,
        "params": fit_params,
    }

    # Include history if available (optional)
    if hasattr(est, "history_"):
        result["history"] = est.history_

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
        Fraction of eligible entries kept for training; must be in (0, 1).
        Remaining (1 - sampling_fraction) becomes validation.
        Note: Constant diagonal entries are excluded from both.
    random_state : int or None, default=None
        Random seed for reproducibility
    missing_values : float or None, default=np.nan
        Value that marks missing entries in original data
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
        if not (0.0 < float(self.sampling_fraction) < 1.0):
            raise ValueError("sampling_fraction must be in (0, 1)")

    def get_n_splits(
        self, x: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        return self.n_repeats

    def split(
        self, x: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation splits.

        Yields
        ------
        train_mask : ndarray of bool
            Training entries (True = use for training)
        validation_mask : ndarray of bool
            Validation entries (True = use for evaluation)
        """
        rng = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            yield create_train_val_split(
                x, self.sampling_fraction, rng, self.missing_values
            )


class GridSearchCV:
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

    def fit(self, x: np.ndarray) -> "GridSearchCV":
        param_grid = ParameterGrid(self.param_grid)
        cv_splits = list(self.cv.split(x))

        all_jobs = [
            (params, train_mask, validation_mask, split_idx)
            for split_idx, (train_mask, validation_mask) in enumerate(cv_splits)
            for params in param_grid
        ]

        logger.info(
            "Running %d jobs (%d params x %d CV splits)",
            len(all_jobs),
            len(param_grid),
            len(cv_splits),
        )

        all_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fit_and_score)(
                self.estimator, x, train_mask, validation_mask, params, split_idx
            )
            for params, train_mask, validation_mask, split_idx in all_jobs
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


def _validate_sampling_fraction(sampling_fraction: float) -> float:
    if sampling_fraction is None:
        raise ValueError(
            "sampling_fraction must be provided when estimate_sampling_fraction is False"
        )
    try:
        sampling_fraction = float(sampling_fraction)
    except (TypeError, ValueError):
        raise TypeError("sampling_fraction must be a float in (0, 1)")
    if not (0.0 < sampling_fraction < 1.0):
        raise ValueError("sampling_fraction must be in (0, 1)")


def cross_val_score(
    similarity_matrix: np.ndarray,
    estimator: BaseEstimator | None = None,
    param_grid: dict[str, list] | None = None,
    n_repeats: int = 5,
    sampling_fraction: float = 0.8,
    estimate_sampling_fraction: bool | dict = False,
    sampling_selection: str = "mean",
    random_state: int = 0,
    verbose: int = 1,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    fit_final_estimator: bool = False,
) -> GridSearchCV:
    """
    Cross-validate any estimator for matrix completion.

    Generic cross-validation function that works with SRF or any sklearn-compatible
    estimator with a .reconstruct() method.

    Parameters
    ----------
    similarity_matrix : ndarray
        Symmetric similarity matrix to cross-validate
    estimator : BaseEstimator or None, default=None
        Estimator to cross-validate. If None, uses SRF(random_state=random_state).
        Can be a single estimator or a Pipeline. Must have a .reconstruct() method.
    param_grid : dict or None, default=None
        Dictionary with parameter names (str) as keys and lists of values to try
        as values. If None, uses default {'rank': [5, 10, 15, 20]} for SRF.
    n_repeats : int, default=5
        Number of times to repeat the cross-validation
    sampling_fraction : float, default=0.8
        Fraction of eligible entries to use for training in each split; must be in (0, 1).
        The remaining (1 - sampling_fraction) becomes validation.
        Note: Constant diagonal entries are excluded from both train and validation.
        Ignored when estimate_sampling_fraction is True or a dict; if both are provided,
        estimate_sampling_fraction takes precedence.
    estimate_sampling_fraction : bool or dict, default=False
        If True, automatically estimate optimal sampling fraction using sampling
        bound estimation from Random Matrix Theory. If dict, passed as kwargs to
        estimate_sampling_bounds_fast(). When enabled, overrides sampling_fraction.
    sampling_selection : str, default="mean"
        Selection method for the estimated sampling fraction; one of {"mean", "min", "max"}.
    random_state : int, default=0
        Random seed for reproducibility
    verbose : int, default=1
        Verbosity level
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 uses all processors)
    missing_values : float or None, default=np.nan
        Value to consider as missing in original data
    fit_final_estimator : bool, default=False
        Whether to fit the final estimator on the best parameters

    Returns
    -------
    grid : GridSearchCV
        Fitted GridSearchCV object with best parameters and scores

    Examples
    --------
    >>> from pysrf.cross_validation import cross_val_score
    >>> result = cross_val_score(similarity_matrix, param_grid={'rank': [5, 10, 15]})
    """
    if estimator is None:
        estimator = SRF(random_state=random_state)

    if param_grid is None:
        param_grid = {"rank": [5, 10, 15, 20]}

    valid_selections = {"mean", "min", "max"}
    if sampling_selection not in valid_selections:
        raise ValueError(
            f"sampling_selection must be one of {sorted(valid_selections)}"
        )

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
        kwargs.pop("verbose", None)

        pmin, pmax, s_noise = estimate_sampling_bounds_fast(similarity_matrix, **kwargs)
        sampling_fraction = {
            "mean": np.mean([pmin, pmax]),
            "min": pmin,
            "max": pmax,
        }[sampling_selection]

    else:
        _validate_sampling_fraction(sampling_fraction)

    cv = EntryMaskSplit(
        n_repeats=n_repeats,
        sampling_fraction=sampling_fraction,
        random_state=random_state,
        missing_values=missing_values,
    )
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_final_estimator=fit_final_estimator,
    )
    grid.fit(similarity_matrix)

    return grid
