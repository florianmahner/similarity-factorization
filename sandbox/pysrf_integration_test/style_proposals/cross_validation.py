"""K-fold cross-validation of SRF rank by reconstruction MSE."""

# RATIONALE:
# Refactor mirrors the style of src/tools/stats.py: one-line module
# docstring, one-line summaries for internal helpers, no Author/License
# banner, type hints throughout. The algorithm is byte-identical to the
# original -- only the surface changed.
#
# Concrete changes:
#   * Module docstring collapsed from an 11-line manifesto to a single line.
#     The "ported from update_pysrf" provenance is a commit-message concern,
#     not something every reader needs at the top of the file.
#   * Argument validation extracted into _validate, keyed off a module-level
#     _RESERVED_SRF_KWARGS frozenset. cross_val_score now reads as a
#     six-line setup followed by the actual fit/score pipeline.
#   * _outer_mask_probability kept as a helper rather than inlined: it owns
#     two distinct concerns (cap and warning) and naming it preserves them.
#   * Job construction lifted into _build_jobs, a generator yielding a
#     frozen @dataclass _Job. This kills the magic-tuple positions used by
#     the original `(rep, fold_idx, rank, tm, vm, seed)` indexing. Iteration
#     order and the fit_seed formula are unchanged. Explicit nested loops
#     are kept over itertools.product per the repo's "no complex one-liners"
#     rule.
#   * DataFrame now built via a dict-of-columns so the column order is
#     declared, not implicit in the dict-iteration order of list-of-dicts.
#   * _fit_score / _mask_missing_entries / _split_observed_into_folds kept
#     algorithmically identical; docstrings shrunk to one line each, and
#     a redundant `observed = ~m_outer[...]` intermediate dropped in
#     _split_observed_into_folds.
#
# Net: ~155 lines vs 227, no behavioural change. The reproduction at
# sandbox/pysrf_integration_test/spectrum_vs_cv/run.py and the existing
# test_cross_validation.py suite remain the contract for "still correct".

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from .coherence._sampling_fraction import _adaptive_cap
from .model import SRF

_RESERVED_SRF_KWARGS = frozenset({"rank", "bounds", "missing_values", "random_state"})


@dataclass(frozen=True)
class _Job:
    rep: int
    fold: int
    rank: int
    train_mask: np.ndarray
    val_mask: np.ndarray
    seed: int


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: list[int],
    sampling_fraction: float,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: int = 0,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    srf_kwargs: dict | None = None,
) -> pd.DataFrame:
    """K-fold cross-validation of SRF rank.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n, n)
    ranks : list of int
    sampling_fraction : float
        Per-fold training density (e.g. ``estimate_rank(s).sampling_fraction``).
    n_folds : int, default=5
    n_repeats : int, default=1
    random_state : int, default=0
    n_jobs : int, default=-1
    missing_values : float or None, default=np.nan
    srf_kwargs : dict or None
        Extra kwargs for :class:`SRF`. Must not contain ``rank``, ``bounds``,
        ``missing_values``, or ``random_state``.

    Returns
    -------
    curve : DataFrame with columns ``rep``, ``fold``, ``rank``, ``val_mse``.
    """
    srf_kwargs = _validate(sampling_fraction, n_folds, n_repeats, srf_kwargs)

    s = np.asarray(similarity_matrix)
    n = s.shape[0]
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))
    off_diag = ~np.eye(n, dtype=bool)
    p_outer = _outer_mask_probability(sampling_fraction, n_folds, n)
    seeds = np.random.SeedSequence(random_state).generate_state(n_repeats)

    jobs = list(_build_jobs(s, seeds, p_outer, n_folds, ranks, off_diag, missing_values))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(s, j.train_mask, j.val_mask, j.rank, bounds, j.seed, srf_kwargs)
        for j in jobs
    )
    return pd.DataFrame({
        "rep": [j.rep for j in jobs],
        "fold": [j.fold for j in jobs],
        "rank": [j.rank for j in jobs],
        "val_mse": scores,
    })


def _validate(
    sampling_fraction: float,
    n_folds: int,
    n_repeats: int,
    srf_kwargs: dict | None,
) -> dict:
    """Validate cross_val_score arguments and return a defensive srf_kwargs copy."""
    if not (0.0 < sampling_fraction < 1.0):
        raise ValueError("sampling_fraction must be in (0, 1)")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    srf_kwargs = dict(srf_kwargs) if srf_kwargs else {}
    reserved = _RESERVED_SRF_KWARGS & srf_kwargs.keys()
    if reserved:
        raise ValueError(
            f"srf_kwargs must not contain {sorted(reserved)}; these are set per fit"
        )
    return srf_kwargs


def _build_jobs(
    s: np.ndarray,
    seeds: np.ndarray,
    p_outer: float,
    n_folds: int,
    ranks: list[int],
    off_diag: np.ndarray,
    missing_values: float | None,
) -> Iterator[_Job]:
    """Yield one ``_Job`` per (rep, fold, rank) with masks and fit seed."""
    for rep, rep_seed in enumerate(seeds):
        rng = check_random_state(int(rep_seed))
        m_outer = _mask_missing_entries(s, p_outer, rng, missing_values)
        val_folds = _split_observed_into_folds(m_outer, n_folds, rng)
        if val_folds is None:
            continue
        for fold_idx, v in enumerate(val_folds):
            train_mask = (~(v | m_outer)) & off_diag
            val_mask = v & off_diag
            for rank in ranks:
                seed = (7919 * (rep + 1) + 101 * fold_idx + 13 * int(rank)) & 0xFFFFFFFF
                yield _Job(rep, fold_idx, int(rank), train_mask, val_mask, seed)


def _outer_mask_probability(sampling_fraction: float, n_folds: int, n: int) -> float:
    """Inflated, capped outer-mask probability; warns if cap binds."""
    cap = _adaptive_cap(n)
    unclipped = sampling_fraction * n_folds / (n_folds - 1)
    if unclipped > cap:
        effective = cap * (n_folds - 1) / n_folds
        warnings.warn(
            f"Inflated outer mask {unclipped:.3f} for n_folds={n_folds} "
            f"exceeds the cap {cap:.3f}. Each fold will train at "
            f"{effective:.3f} instead of the requested {sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(unclipped, cap)


def _mask_missing_entries(
    x: np.ndarray,
    observed_fraction: float,
    rng: np.random.RandomState,
    missing_values: float | None,
) -> np.ndarray:
    """Symmetric outer mask (True = missing); diagonal always observed."""
    if missing_values is None or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    ):
        observed = np.isfinite(x)
    else:
        observed = np.not_equal(x, missing_values)
    triu_i, triu_j = np.triu_indices_from(x, k=1)
    valid = np.where(observed[triu_i, triu_j])[0]

    n_keep = int(observed_fraction * len(valid))
    if n_keep == 0:
        return np.ones_like(x, dtype=bool)

    keep = rng.choice(valid, size=n_keep, replace=False)
    missing = np.ones_like(x, dtype=bool)
    missing[triu_i[keep], triu_j[keep]] = False
    missing[triu_j[keep], triu_i[keep]] = False
    np.fill_diagonal(missing, False)
    return missing


def _split_observed_into_folds(
    m_outer: np.ndarray,
    n_folds: int,
    rng: np.random.RandomState,
) -> list[np.ndarray] | None:
    """Partition observed off-diagonal pairs into ``n_folds`` symmetric masks."""
    n = m_outer.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    valid = np.where(~m_outer[triu_i, triu_j])[0]
    if len(valid) < n_folds:
        return None
    perm = rng.permutation(len(valid))
    folds = []
    for group in np.array_split(perm, n_folds):
        mask = np.zeros((n, n), dtype=bool)
        ii, jj = triu_i[valid[group]], triu_j[valid[group]]
        mask[ii, jj] = True
        mask[jj, ii] = True
        folds.append(mask)
    return folds


def _fit_score(
    s: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    seed: int,
    srf_kwargs: dict,
) -> float:
    """Fit SRF on training entries; return upper-triangle V-MSE."""
    x_train = np.full_like(s, np.nan)
    x_train[train_mask] = s[train_mask]
    est = SRF(
        rank=rank, bounds=bounds, missing_values=np.nan,
        random_state=seed, **srf_kwargs,
    )
    est.fit(x_train)
    s_hat = est.reconstruct()

    iu = np.triu_indices(s.shape[0], k=1)
    m = val_mask[iu] & np.isfinite(s[iu]) & np.isfinite(s_hat[iu])
    if not m.any():
        return float("nan")
    return float(np.mean((s[iu][m] - s_hat[iu][m]) ** 2))
