"""Recovery curve, monotone projection, and the sampling-fraction calibration.

Once the rank has been chosen, the *recovery curve* is the fraction of
top-rank spectral mass missing under random masking, as a function of
the sampling probability ``p``. Larger ``p`` means less masking and a
smaller deficit; we enforce that monotonicity, invert at the chosen
``recovery_tolerance``, and floor the result by a random-matrix
detectability threshold to guard against optimistic inversions on
borderline data.
"""

from __future__ import annotations

import numpy as np


def _monotone_decreasing(y: np.ndarray) -> np.ndarray:
    """Closest non-increasing sequence to ``y`` in the least-squares sense.

    Implementation uses the pool-adjacent-violators algorithm: walk
    left to right, merging adjacent blocks whenever the running mean
    would increase, until every adjacent pair is non-increasing.
    """
    y = np.asarray(y, dtype=np.float64)
    block_sums: list[float] = []
    block_sizes: list[int] = []
    for value in y:
        s, n = float(value), 1
        while block_sums and block_sums[-1] / block_sizes[-1] < s / n:
            s += block_sums.pop()
            n += block_sizes.pop()
        block_sums.append(s)
        block_sizes.append(n)
    out = np.empty(len(y), dtype=np.float64)
    pos = 0
    for s, n in zip(block_sums, block_sizes):
        out[pos : pos + n] = s / n
        pos += n
    return out


def recovery_curve(
    sampling_grid: np.ndarray,
    eigenvalues: np.ndarray,
    projected_median: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fraction of top-``rank`` spectral mass missed under masking.

    ``projected_median[rank-1, p]`` is the median trace of ``S``
    projected onto the top-``rank`` bootstrap subspace at sampling
    probability ``p``. The raw deficit is

        deficit(p) = 1 - projected_median[rank-1, p] / sum(eigenvalues[:rank])

    Larger ``p`` (less masking) → smaller deficit. We sort by ``p``
    and apply :func:`_monotone_decreasing` to enforce that monotonicity
    against bootstrap noise.

    Returns
    -------
    p_sorted : ndarray of shape (P,)
    raw : ndarray of shape (P,)
        Empirical deficit, not necessarily monotone.
    monotone : ndarray of shape (P,)
        Non-increasing projection of ``raw``.
    """
    reference_mass = float(eigenvalues[:rank].sum())
    raw = 1.0 - projected_median[rank - 1, :] / max(reference_mass, 1e-12)
    order = np.argsort(sampling_grid)
    p_sorted = sampling_grid[order]
    raw_sorted = raw[order]
    return p_sorted, raw_sorted, _monotone_decreasing(raw_sorted)


def invert_recovery(
    p_sorted: np.ndarray, monotone: np.ndarray, tolerance: float
) -> float:
    """Smallest ``p`` where the recovery deficit is at or below ``tolerance``.

    Linear interpolation between adjacent grid points. Falls back to
    the boundary of the grid if ``tolerance`` lies outside the observed
    deficit range.
    """
    if monotone[-1] >= tolerance:
        return float(p_sorted[-1])
    if monotone[0] <= tolerance:
        return float(p_sorted[0])

    index = int(np.searchsorted(-monotone, -tolerance, side="left"))
    index = max(min(index, len(monotone) - 1), 1)
    lower, upper = monotone[index - 1], monotone[index]
    p_lo, p_hi = p_sorted[index - 1], p_sorted[index]
    if lower == upper:
        return float(p_lo)
    fraction = (lower - tolerance) / (lower - upper)
    return float(p_lo + fraction * (p_hi - p_lo))


def detectability_floor(eigenvalues: np.ndarray, rank: int) -> float:
    """Random-matrix lower bound on the calibrated sampling fraction.

    Below this sampling probability the rank-th signal eigenvalue would
    fall below the bulk noise threshold and could not be reliably
    detected, regardless of how clean the recovery curve looks. The
    floor is set by the first spectral gap at the signal/noise
    boundary:

        gap_squared = (eigenvalues[rank] / eigenvalues[rank - 1]) ** 2
        floor       = gap_squared / (1 + gap_squared)

    A small first gap (signal barely above bulk) gives a floor near
    0.5; a large gap gives a floor near 0.
    """
    lam_k = float(eigenvalues[rank - 1]) if rank >= 1 else float(eigenvalues[0])
    lam_kp1 = float(eigenvalues[min(rank, len(eigenvalues) - 1)])
    if lam_k <= 1e-12:
        return 1.0
    gap_squared = (lam_kp1 / lam_k) ** 2
    return float(np.clip(gap_squared / (1.0 + gap_squared), 0.0, 1.0))
