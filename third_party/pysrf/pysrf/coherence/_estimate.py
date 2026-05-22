from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .._common import (
    RandomStateLike,
    as_seed_sequence,
    observation_mask,
    symmetrize_observations,
)
from ._bootstrap import _bootstrap_subspace_stability, _top_eigenpairs
from ._rank_selection import _select_rank
from ._sampling_fraction import _calibrate_sampling_fraction


@dataclass(frozen=True)
class RankEstimate:
    """Output of estimate_rank.

    Attributes
    ----------
    rank : the estimated number of signal dimensions in S.
    sampling_fraction : per-fold training density to pass to
        pysrf.cross_val_score.
    eigenvalues : the top eigenvalues of S, sorted descending.
    leakage : per-dimension instability score; small for signal,
        large for noise.
    sampling_grid : sampling probabilities used by the bootstrap,
        sorted ascending.
    recovery_loss_raw : per sampling probability, the fraction of
        top-rank spectral mass that the bootstrap fails to recover.
    recovery_loss_monotone : the same curve forced to be non-increasing
        in p; this is the curve we invert at recovery_tolerance.
    detectability_floor : a random-matrix lower bound on sampling_fraction.
    n_features_in : size of the input matrix.
    """

    rank: int
    sampling_fraction: float
    eigenvalues: np.ndarray
    leakage: np.ndarray
    sampling_grid: np.ndarray
    recovery_loss_raw: np.ndarray
    recovery_loss_monotone: np.ndarray
    detectability_floor: float
    n_features_in: int


def estimate_rank(
    s: np.ndarray,
    recovery_tolerance: float = 0.10,
    max_rank: int | None = None,
    sampling_grid: np.ndarray | None = None,
    n_bootstrap: int = 50,
    high_band_quantile: float = 0.85,
    random_state: RandomStateLike = 0,
    n_jobs: int | None = None,
) -> RankEstimate:
    """Estimate (rank, sampling_fraction) for a symmetric similarity matrix.

    Bootstraps the top eigenspace under random off-diagonal subsampling.
    Picks the rank from the F-statistic changepoint of the per-dimension
    leakage profile. Inverts the recovery curve at recovery_tolerance
    (floored by a random-matrix detectability bound) to pick the
    calibrated sampling fraction.

    Parameters
    ----------
    s : array of shape (n, n)
        Symmetric similarity matrix; NaN marks missing entries.
    recovery_tolerance : default=0.10
        How much of the top-rank spectral mass we tolerate losing at
        the calibrated sampling fraction.
    max_rank : default None
        Largest candidate rank. Defaults to max(min(n // 4, 100), 2).
    sampling_grid : default None
        Sampling probabilities tried by the bootstrap. Defaults to
        np.linspace(0.05, 0.95, 20).
    n_bootstrap : default=20
    high_band_quantile : default=0.85
        Quantile of sampling_grid defining the high-p band used to
        aggregate the per-rank leakage score.
    random_state : default=0
        Anything as_seed_sequence accepts: int, None, SeedSequence,
        Generator, or RandomState.
    n_jobs : default None
        Parallel workers. None means cpu_count - 1.

    Returns
    -------
    RankEstimate
    """
    s, observed_mask = _prepare_input(s)
    n = s.shape[0]

    if max_rank is None:
        max_rank = max(min(n // 4, 100), 2)
    if sampling_grid is None:
        sampling_grid = np.linspace(0.05, 0.95, 20)
    else:
        sampling_grid = np.asarray(sampling_grid, dtype=np.float64)
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = max_workers if n_jobs is None else max(1, min(int(n_jobs), max_workers))

    top_eigenvalues, top_eigenvectors = _top_eigenpairs(s, max_rank)

    coherence, recovered_spectral_mass = _bootstrap_subspace_stability(
        s, observed_mask, top_eigenvectors, sampling_grid,
        n_bootstrap=n_bootstrap, random_state=random_state, n_jobs=n_jobs,
    )

    rank, leakage = _select_rank(coherence, sampling_grid, high_band_quantile)
    sampling_fraction, loss_curve = _calibrate_sampling_fraction(
        recovered_spectral_mass, top_eigenvalues, sampling_grid, rank, recovery_tolerance,
    )

    return RankEstimate(
        rank=int(rank),
        sampling_fraction=float(sampling_fraction),
        eigenvalues=top_eigenvalues,
        leakage=leakage,
        sampling_grid=loss_curve.sampling_grid,
        recovery_loss_raw=loss_curve.raw,
        recovery_loss_monotone=loss_curve.monotone,
        detectability_floor=loss_curve.floor,
        n_features_in=int(n),
    )


def _prepare_input(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = symmetrize_observations(s)
    observed_mask = observation_mask(s)
    np.fill_diagonal(observed_mask, True)
    return np.nan_to_num(s, nan=0.0), observed_mask
