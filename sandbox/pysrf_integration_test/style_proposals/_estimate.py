"""Public function and dataclass for rank estimation."""

# RATIONALE:
# Cut:
#   - Author/License header block (matches stats.py, which omits them).
#   - The three _resolve_* helpers. Each contained 1-3 lines of trivial
#     default-handling that read more clearly inlined at the single
#     call site in estimate_rank than as separately named functions.
#     The names didn't earn their place: "_resolve_max_rank" is no more
#     informative than "max(min(n // 4, 100), 2)" right where it's used.
#   - Verbose per-field prose in CoherenceProfile. The type annotations
#     already carry shape and dtype; the docstring now gives a one-line
#     intent per field instead of repeating "ndarray of shape (P,)".
#   - The intermediate `s_sym`, `n`, `k_max`, `grid`, `jobs`, `seed`,
#     `s_filled`, `mask`, `observed_rate`, `raw_fraction`, `floor`
#     bindings were unpacked into the construction call where they only
#     appeared once. Kept names that get reused or that document a
#     conceptual step (eigenvalues, leakage, rank, recovery_*).
#
# Kept:
#   - Full Parameters section on estimate_rank: this is the user-facing
#     API surface and the task brief explicitly required it.
#   - CoherenceProfile field order and types (load-bearing for pickling
#     and for the audit script's attribute access).
#   - Exact computation order and arguments — the audit re-runs and
#     requires byte-identical numerics.

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ._bootstrap import (
    _bootstrap_coherence,
    _observation_mask,
    _reference_eigenpairs,
    _symmetrize,
)
from ._rank_selection import _changepoint, _leakage_profile
from ._sampling_fraction import _detectability_floor, _invert_recovery, _recovery_curve


@dataclass(frozen=True)
class CoherenceProfile:
    """Output of :func:`estimate_rank`.

    Attributes
    ----------
    rank : estimated number of signal dimensions.
    sampling_fraction : per-fold training density for CV; pass to
        :func:`pysrf.cross_val_score`.
    eigenvalues : top reference eigenvalues, descending.
    leakage : per-dimension instability score (small for signal, large for noise).
    sampling_grid : sampling probabilities used by the bootstrap.
    recovery_raw : empirical recovery deficit at each grid point.
    recovery_monotone : non-increasing projection of ``recovery_raw``;
        inverted at ``recovery_tolerance`` to set ``sampling_fraction``.
    detectability_floor : random-matrix lower bound on ``sampling_fraction``.
    n_features_in : size of the input matrix.
    """

    rank: int
    sampling_fraction: float
    eigenvalues: np.ndarray
    leakage: np.ndarray
    sampling_grid: np.ndarray
    recovery_raw: np.ndarray
    recovery_monotone: np.ndarray
    detectability_floor: float
    n_features_in: int


def estimate_rank(
    s: np.ndarray,
    recovery_tolerance: float = 0.10,
    max_rank: int | None = None,
    sampling_grid: np.ndarray | None = None,
    n_bootstrap: int = 20,
    high_band_quantile: float = 0.85,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> CoherenceProfile:
    """Estimate the number of signal dimensions of a symmetric similarity matrix.

    Bootstraps the top eigenspace under random off-diagonal masking,
    picks the rank from the F-statistic changepoint of the per-dimension
    leakage profile, and inverts the recovery curve at
    ``recovery_tolerance`` (floored by a random-matrix detectability
    bound) to pick the calibrated sampling fraction.

    Parameters
    ----------
    s : array-like of shape (n, n)
        Symmetric similarity matrix; NaN marks missing entries.
    recovery_tolerance : float, default=0.10
        Max fraction of top-rank spectral mass allowed to be missing
        at the calibrated sampling fraction.
    max_rank : int or None, default=None
        Largest candidate rank. Defaults to ``min(n // 4, 100)``.
    sampling_grid : array-like or None, default=None
        Strictly-increasing sampling probabilities in (0, 1]. Defaults
        to ``np.linspace(0.05, 0.95, 20)``.
    n_bootstrap : int, default=20
    high_band_quantile : float, default=0.85
        Quantile of ``sampling_grid`` defining the high-p band used to
        aggregate the per-rank leakage score.
    random_state : int, default=0
    n_jobs : int or None, default=None
        Parallel workers. ``None`` uses ``cpu_count - 1``.

    Returns
    -------
    estimate : CoherenceProfile
    """
    s_sym = _symmetrize(np.asarray(s, dtype=np.float64))
    n = s_sym.shape[0]
    k_max = max(min(n // 4, 100), 2) if max_rank is None else int(max_rank)
    grid = (
        np.linspace(0.05, 0.95, 20)
        if sampling_grid is None
        else np.asarray(sampling_grid, dtype=np.float64)
    )
    cpu = os.cpu_count() or 2
    jobs = max(1, cpu - 1) if n_jobs is None else max(1, min(int(n_jobs), cpu - 1))
    seed = int(random_state)

    s_filled, mask, observed_rate = _observation_mask(s_sym)
    eigenvalues, u_ref = _reference_eigenpairs(s_filled, mask, observed_rate, k_max, seed)

    overlap, projected = _bootstrap_coherence(
        s_filled, mask, u_ref, k_max, grid,
        n_bootstrap=n_bootstrap, random_state=seed, n_jobs=jobs,
    )
    leakage = _leakage_profile(np.median(overlap, axis=2), grid, high_band_quantile)
    rank = _changepoint(leakage)

    p_sorted, recovery_raw, recovery_monotone = _recovery_curve(
        grid, eigenvalues, np.median(projected, axis=2), rank
    )
    floor = _detectability_floor(eigenvalues, rank)
    sampling_fraction = max(
        _invert_recovery(p_sorted, recovery_monotone, recovery_tolerance), floor
    )

    return CoherenceProfile(
        rank=int(rank),
        sampling_fraction=float(sampling_fraction),
        eigenvalues=eigenvalues,
        leakage=leakage,
        sampling_grid=p_sorted,
        recovery_raw=recovery_raw,
        recovery_monotone=recovery_monotone,
        detectability_floor=float(floor),
        n_features_in=int(n),
    )
