"""Cross-validation utilities for symmetric matrix completion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.utils import check_random_state
from typing import Generator

from .model import SRF
from .coherence import RankEstimator


class EntryKFold(BaseCrossValidator):
    """K-fold partition cross-validator for symmetric similarity matrices.

    Pre-masks observed entries at an inflated outer probability so each
    of ``n_folds`` folds trains at exactly ``sampling_fraction``. The
    inflated probability is capped at ``max(0.95, 1 - 2000 / N_pairs)``.

    Parameters
    ----------
    n_folds : int, default=5
    sampling_fraction : float, default=0.8
    random_state : int or None, default=None
    missing_values : float or None, default=np.nan
    """

    def __init__(
        self,
        n_folds: int = 5,
        sampling_fraction: float = 0.8,
        random_state: int | None = None,
        missing_values: float | None = np.nan,
    ):
        ...

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
        ...

    def split(
        self, x: np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        ...


def fit_and_score(
    estimator: BaseEstimator,
    x: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    fit_params: dict,
    split_idx: int | None = None,
) -> dict:
    """Fit an estimator on training entries and score V-MSE on validation."""
    raise NotImplementedError


class GridSearchCV:
    """Grid-search cross-validation."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        cv: EntryKFold,
        n_jobs: int = -1,
        verbose: int = 0,
        fit_final_estimator: bool = False,
    ):
        ...

    def fit(self, x: np.ndarray) -> "GridSearchCV":
        ...


def cross_val_score(
    similarity_matrix: np.ndarray,
    estimator: BaseEstimator | None = None,
    param_grid: dict[str, list] | None = None,
    n_folds: int = 5,
    sampling_fraction: float | None = None,
    estimate_sampling_fraction: bool | dict = False,
    random_state: int = 0,
    verbose: int = 1,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    fit_final_estimator: bool = False,
) -> GridSearchCV:
    """K-fold cross-validation with optional automatic sampling fraction."""
    raise NotImplementedError
