"""Sampling bounds estimation for matrices with missing values.

When a similarity matrix contains missing entries (NaN), standard bounds estimation
fails. This module provides bounds estimation via imputation: factorize the incomplete
matrix, reconstruct a complete approximation, then estimate bounds on that.

The key insight is that the estimated bounds are stable regardless of the imputation
rank used, so we can use any reasonable rank for imputation.

Usage:
    from experiments.bounds_missing import estimate_bounds_with_missing

    pmin, pmax = estimate_bounds_with_missing(similarity_matrix)
    sampling_fraction = (pmin + pmax) / 2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra


def impute_similarity_matrix(
    S: NDArray[np.floating],
    rank: int,
    random_state: int = 42,
    max_outer: int = 100,
    tol: float = 1e-6,
) -> NDArray[np.floating]:
    """Impute missing values via SRF factorization and reconstruction.

    Parameters
    ----------
    S : ndarray
        Symmetric similarity matrix with NaN for missing entries.
    rank : int
        Rank for factorization.
    random_state : int
        Random seed for SRF initialization.
    max_outer : int
        Maximum outer iterations for SRF.
    tol : float
        Convergence tolerance for SRF.

    Returns
    -------
    S_imputed : ndarray
        Complete symmetric matrix with missing values filled in.
    """
    model = SRF(
        rank=rank,
        missing_values=np.nan,
        random_state=random_state,
        max_outer=max_outer,
        tol=tol,
    )
    model.fit(S)
    S_imputed = model.reconstruct()

    obs_min, obs_max = np.nanmin(S), np.nanmax(S)
    return np.clip(S_imputed, obs_min, obs_max)


def estimate_bounds_with_missing(
    S: NDArray[np.floating],
    imputation_rank: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[float, float]:
    """Estimate sampling bounds for a matrix with missing values.

    Uses the coherence approach: impute missing values via SRF, then estimate
    bounds on the imputed (complete) matrix.

    Parameters
    ----------
    S : ndarray
        Symmetric similarity matrix with NaN for missing entries.
    imputation_rank : int or None
        Rank used for imputation. If None, uses heuristic based on matrix size.
    random_state : int
        Random state for imputation and bounds estimation.
    verbose : bool
        Print progress information.

    Returns
    -------
    pmin : float
        Lower bound on sampling fraction.
    pmax : float
        Upper bound on sampling fraction.
    """
    n = S.shape[0]
    n_missing = np.isnan(S).sum()

    if n_missing == 0:
        pmin, pmax, _ = estimate_sampling_bounds_ultra(
            S, random_state=random_state, verbose=verbose
        )
        return float(pmin), float(pmax)

    if imputation_rank is None:
        imputation_rank = max(5, min(n // 10, 20))

    if verbose:
        missing_frac = n_missing / (n * n)
        print(f"Matrix has {missing_frac:.1%} missing values")
        print(f"Imputing with rank={imputation_rank}")

    S_imputed = impute_similarity_matrix(
        S, rank=imputation_rank, random_state=random_state
    )

    pmin, pmax, _ = estimate_sampling_bounds_ultra(
        S_imputed, random_state=random_state, verbose=verbose
    )

    return float(pmin), float(pmax)


def get_sampling_fraction(
    S: NDArray[np.floating],
    selection: str = "mean",
    **kwargs,
) -> float:
    """Get sampling fraction for cross-validation, handling missing values.

    Parameters
    ----------
    S : ndarray
        Symmetric similarity matrix, may contain NaN.
    selection : str
        How to combine bounds: 'mean', 'min', or 'max'.
    **kwargs
        Passed to estimate_bounds_with_missing.

    Returns
    -------
    sampling_fraction : float
        Recommended sampling fraction for CV.
    """
    pmin, pmax = estimate_bounds_with_missing(S, **kwargs)

    if selection == "mean":
        return (pmin + pmax) / 2
    elif selection == "min":
        return pmin
    elif selection == "max":
        return pmax
    else:
        raise ValueError(f"selection must be 'mean', 'min', or 'max', got {selection}")
