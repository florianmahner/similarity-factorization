"""sklearn-style estimator that wraps the coherence pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class RankEstimator(BaseEstimator):
    """Bootstrap rank estimation by eigenspace stability under masking.

    Parameters
    ----------
    recovery_tolerance : float, default=0.10
    max_rank : int or None, default=None
    sampling_grid : array-like or None, default=None
    n_bootstrap : int, default=20
    high_band_quantile : float, default=0.85
    random_state : int or None, default=0
    n_jobs : int or None, default=None

    Attributes
    ----------
    rank_ : int
    sampling_fraction_ : float
    eigenvalues_ : ndarray
    leakage_ : ndarray
    sampling_grid_ : ndarray
    recovery_raw_ : ndarray
    recovery_monotone_ : ndarray
    detectability_floor_ : float
    n_features_in_ : int
    """

    def __init__(
        self,
        recovery_tolerance: float = 0.10,
        max_rank: int | None = None,
        sampling_grid: np.ndarray | None = None,
        n_bootstrap: int = 20,
        high_band_quantile: float = 0.85,
        random_state: int | None = 0,
        n_jobs: int | None = None,
    ) -> None:
        ...

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RankEstimator":
        """Estimate rank and calibrated sampling fraction."""
        raise NotImplementedError

    def cv_sampling_fraction(self, n_folds: int) -> float:
        """Outer-mask probability for ``n_folds``-fold CV (inflated and capped)."""
        raise NotImplementedError
