"""``RankEstimator`` — the public sklearn-style estimator class.

Orchestrates the three internal pieces: a masked bootstrap of the
eigenspace, a leakage profile + changepoint to pick the rank, and a
recovery-curve inversion + detectability floor to pick the calibrated
sampling fraction.

The fitted estimator exposes:

- ``rank_`` and ``sampling_fraction_`` — the two essentials;
- a method ``cv_sampling_fraction(n_folds)`` for k-fold CV;
- diagnostic arrays for plotting and inspection.
"""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Integral, Interval, Real
from sklearn.utils.validation import check_is_fitted

from ._bootstrap import (
    bootstrap_coherence,
    observation_mask,
    reference_eigenpairs,
    symmetrize,
)
from ._rank_selection import changepoint, leakage_profile
from ._sampling_fraction import detectability_floor, invert_recovery, recovery_curve


logger = logging.getLogger(__name__)


_CV_CAP_FLOOR = 0.95
"""Lower bound on the outer-mask cap, regardless of ``n``."""

_CV_HOLDOUT_BUDGET = 2000
"""Minimum number of off-diagonal pairs kept for validation. The cap
adapts to ``n`` as ``max(_CV_CAP_FLOOR, 1 - _CV_HOLDOUT_BUDGET / N_pairs)``.
Matches ``recipe_K``'s ``M_min=2000, p_max_floor=0.95`` defaults."""


class RankEstimator(BaseEstimator):
    """Bootstrap rank estimation by eigenspace stability under masking.

    Estimates the number of signal dimensions in a symmetric similarity
    matrix and the minimum sampling fraction needed to recover them.

    The method works by:

    1. Bootstrapping eigenvectors of randomly masked copies of the
       matrix.
    2. Measuring how stable each eigenspace dimension is under masking,
       summarized as a per-dimension *leakage* score. Signal dimensions
       have small leakage; noise dimensions leak heavily.
    3. Picking the rank at the F-statistic changepoint of the leakage
       profile (``rank_``).
    4. Inverting the *recovery curve* — the fraction of top-``rank_``
       spectral mass captured at each sampling rate — at the requested
       ``recovery_tolerance`` to get the calibrated sampling fraction
       (``sampling_fraction_``).
    5. Flooring that sampling fraction by the random-matrix
       detectability threshold to guard against optimistic curves on
       borderline data.

    Parameters
    ----------
    recovery_tolerance : float, default=0.10
        Maximum fraction of top-``rank_`` spectral mass allowed to be
        missing at the calibrated sampling fraction. Smaller values
        demand more observed entries.
    max_rank : int or None, default=None
        Largest candidate rank to test. Defaults to ``min(n // 4, 100)``.
    sampling_grid : array-like of shape (P,) or None, default=None
        Strictly-increasing sampling probabilities in (0, 1] at which
        to evaluate eigenspace stability. Defaults to
        ``linspace(0.05, 0.95, 20)``.
    n_bootstrap : int, default=20
        Bootstrap replicates per sampling probability.
    high_band_quantile : float, default=0.85
        Quantile of ``sampling_grid`` defining the high-p band used to
        aggregate the per-rank leakage score.
    random_state : int or None, default=0
        Seed for reproducibility.
    n_jobs : int or None, default=None
        Number of parallel workers. ``None`` uses ``cpu_count - 1``.

    Attributes
    ----------
    rank_ : int
        Estimated number of signal dimensions.
    sampling_fraction_ : float
        Minimum per-fold training density needed to recover the signal
        eigenspace within ``recovery_tolerance``. Suitable as the
        ``sampling_fraction`` argument of :func:`pysrf.cross_val_score`
        for repeated-holdout verification.
    eigenvalues_ : ndarray of shape (max_rank,)
        Top-``max_rank`` reference eigenvalues, descending.
    leakage_ : ndarray of shape (max_rank,)
        Per-dimension instability score. Signal dimensions have small
        leakage; noise dimensions have large leakage.
    sampling_grid_ : ndarray of shape (P,)
        Sampling probabilities at which the bootstrap was evaluated
        (sorted).
    recovery_raw_ : ndarray of shape (P,)
        Raw deficit (fraction of top-``rank_`` spectral mass missing)
        at each ``sampling_grid_`` point.
    recovery_monotone_ : ndarray of shape (P,)
        Non-increasing projection of ``recovery_raw_``. The estimator
        inverts this curve at ``recovery_tolerance`` to get
        ``sampling_fraction_``.
    detectability_floor_ : float
        Random-matrix lower bound on ``sampling_fraction_``. If
        ``sampling_fraction_`` equals this value, the signal is at the
        detection limit and rank estimates may be unstable.
    n_features_in_ : int
        Size of the input matrix.

    Examples
    --------
    >>> from pysrf import RankEstimator, cross_val_score
    >>> est = RankEstimator(recovery_tolerance=0.10).fit(s)
    >>> est.rank_
    10
    >>> est.sampling_fraction_
    0.68

    Verify with repeated-holdout cross-validation at the calibrated
    sampling fraction:

    >>> grid = cross_val_score(
    ...     s,
    ...     param_grid={"rank": [est.rank_ - 2, est.rank_, est.rank_ + 2]},
    ...     sampling_fraction=est.sampling_fraction_,
    ...     n_repeats=5,
    ... )

    For k-fold cross-validation (each fold trains at ``sampling_fraction_``):

    >>> from pysrf import EntryKFold
    >>> cv = EntryKFold(n_folds=5, sampling_fraction=est.sampling_fraction_)
    >>> grid = cross_val_score(s, cv=cv, param_grid={"rank": [...]})
    """

    _parameter_constraints = {
        "recovery_tolerance": [Interval(Real, 0.0, 1.0, closed="neither")],
        "max_rank": [None, Interval(Integral, 1, None, closed="left")],
        "sampling_grid": [None, "array-like"],
        "n_bootstrap": [Interval(Integral, 1, None, closed="left")],
        "high_band_quantile": [Interval(Real, 0.0, 1.0, closed="right")],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
    }

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
        self.recovery_tolerance = recovery_tolerance
        self.max_rank = max_rank
        self.sampling_grid = sampling_grid
        self.n_bootstrap = n_bootstrap
        self.high_band_quantile = high_band_quantile
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> RankEstimator:
        """Estimate rank and calibrated sampling fraction.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing entries may be marked
            with NaN.
        y : Ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : RankEstimator
            Fitted estimator.
        """
        self._validate_params()

        s_sym = symmetrize(np.asarray(x, dtype=np.float64))
        n = s_sym.shape[0]
        self.n_features_in_ = n

        k_max = self._resolve_max_rank(n)
        sampling_grid = self._resolve_sampling_grid()
        n_jobs = self._resolve_n_jobs()
        random_state = 0 if self.random_state is None else int(self.random_state)

        s_filled, mask, observed_rate = observation_mask(s_sym)
        eigenvalues, u_ref = reference_eigenpairs(
            s_filled, mask, observed_rate, k_max, random_state
        )

        overlap, projected = bootstrap_coherence(
            s_filled, mask, u_ref, k_max, sampling_grid,
            n_bootstrap=self.n_bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        overlap_median = np.median(overlap, axis=2)
        projected_median = np.median(projected, axis=2)

        leakage = leakage_profile(overlap_median, sampling_grid, self.high_band_quantile)
        rank = changepoint(leakage)

        p_sorted, recovery_raw, recovery_monotone = recovery_curve(
            sampling_grid, eigenvalues, projected_median, rank
        )
        raw_fraction = invert_recovery(p_sorted, recovery_monotone, self.recovery_tolerance)
        floor = detectability_floor(eigenvalues, rank)
        sampling_fraction = float(max(raw_fraction, floor))

        self.rank_ = int(rank)
        self.sampling_fraction_ = sampling_fraction
        self.eigenvalues_ = eigenvalues
        self.leakage_ = leakage
        self.sampling_grid_ = p_sorted
        self.recovery_raw_ = recovery_raw
        self.recovery_monotone_ = recovery_monotone
        self.detectability_floor_ = float(floor)
        return self

    def cv_sampling_fraction(self, n_folds: int) -> float:
        """Outer mask probability for n-fold cross-validation.

        In k-fold CV, the outer mask keeps a fraction ``p_outer`` of
        entries observed and partitions them into ``k`` folds. Each
        fold trains on ``(k - 1) / k`` of the kept entries. To make
        every fold train at the calibrated ``sampling_fraction_``, the
        outer mask must be inflated:

            p_outer = sampling_fraction_ * k / (k - 1)

        The return value is capped at
        ``max(0.95, 1 - 2000 / N_pairs)`` so at least ~2000 off-diagonal
        pairs are kept for validation regardless of ``n``. If the cap
        binds, a warning is emitted and each fold trains below the
        calibrated density.

        Parameters
        ----------
        n_folds : int
            Number of inner folds. Must be at least 2.

        Returns
        -------
        p_outer : float
            Probability to use as ``sampling_fraction`` on an
            :class:`EntryKFold` cross-validator with ``n_folds`` folds.
        """
        check_is_fitted(self)
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        cap = self._cv_cap()
        unclipped = self.sampling_fraction_ * n_folds / (n_folds - 1)
        if unclipped > cap:
            effective = cap * (n_folds - 1) / n_folds
            warnings.warn(
                f"Inflated outer mask {unclipped:.3f} for n_folds={n_folds} "
                f"exceeds the cap {cap:.3f}. Each fold will train at "
                f"{effective:.3f} instead of the calibrated "
                f"{self.sampling_fraction_:.3f}; rank-selection accuracy may "
                f"degrade.",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(min(unclipped, cap))

    def _cv_cap(self) -> float:
        """Adaptive cap on the outer-mask probability for CV."""
        n = int(self.n_features_in_)
        n_pairs = n * (n - 1) / 2
        if n_pairs <= 0:
            return _CV_CAP_FLOOR
        return float(max(_CV_CAP_FLOOR, 1.0 - _CV_HOLDOUT_BUDGET / n_pairs))

    def _resolve_max_rank(self, n: int) -> int:
        if self.max_rank is None:
            return max(min(n // 4, 100), 2)
        return int(self.max_rank)

    def _resolve_sampling_grid(self) -> np.ndarray:
        if self.sampling_grid is None:
            return np.linspace(0.05, 0.95, 20)
        return np.asarray(self.sampling_grid, dtype=np.float64)

    def _resolve_n_jobs(self) -> int:
        cpu = os.cpu_count() or 2
        if self.n_jobs is None:
            return max(1, cpu - 1)
        return max(1, min(int(self.n_jobs), cpu - 1))
