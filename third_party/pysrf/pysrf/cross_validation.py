from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from ._common import RandomStateLike, as_seed_sequence, observation_mask, spawn_ints
from .coherence._sampling_fraction import _adaptive_cap
from .model import SRF

_RESERVED_SRF_KWARGS = frozenset({"rank", "bounds", "missing_values", "random_state"})


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: Sequence[int],
    sampling_fraction: float,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: RandomStateLike = 0,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
    srf_kwargs: dict | None = None,
) -> pd.DataFrame:
    """K-fold confirmation curve for SRF rank estimates."""
    ranks, srf_kwargs = _validate_args(ranks, sampling_fraction, n_folds, n_repeats, srf_kwargs)
    s = np.asarray(similarity_matrix, dtype=np.float64)
    observed_mask = observation_mask(s, missing_values)
    bounds = _observed_bounds(s, observed_mask)

    pool_fraction = _cv_pool_fraction(sampling_fraction, n_folds, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(
        random_state, n_repeats, n_folds, len(ranks),
    )
    splits = _entry_splits(observed_mask, pool_fraction, n_folds, split_seeds)
    jobs = list(_fit_jobs(splits, ranks, fit_seeds))

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(s, train_mask, validation_mask, rank, bounds, seed, srf_kwargs)
        for _, _, rank, train_mask, validation_mask, seed in jobs
    )
    return pd.DataFrame(
        {"rep": rep, "fold": fold, "rank": rank, "val_mse": score}
        for (rep, fold, rank, *_), score in zip(jobs, scores)
    )


def _validate_args(
    ranks: Sequence[int],
    sampling_fraction: float,
    n_folds: int,
    n_repeats: int,
    srf_kwargs: dict | None,
) -> tuple[list[int], dict]:
    ranks = [int(rank) for rank in ranks]
    if not ranks:
        raise ValueError("ranks must contain at least one value")
    if any(rank < 1 for rank in ranks):
        raise ValueError("ranks must be positive")
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
    return ranks, srf_kwargs


def _observed_bounds(s: np.ndarray, observed_mask: np.ndarray) -> tuple[float, float]:
    values = s[observed_mask & np.isfinite(s)]
    if values.size == 0:
        raise ValueError("similarity_matrix has no finite observed entries")
    return float(values.min()), float(values.max())


def _split_fit_seeds(
    random_state: RandomStateLike,
    n_repeats: int,
    n_folds: int,
    n_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_sequence = as_seed_sequence(random_state)
    split_seeds = spawn_ints(seed_sequence, n_repeats)
    fit_seeds = spawn_ints(seed_sequence, n_repeats * n_folds * n_ranks)
    return split_seeds, fit_seeds.reshape(n_repeats, n_folds, n_ranks)


def _entry_splits(
    observed_mask: np.ndarray,
    pool_fraction: float,
    n_folds: int,
    split_seeds: np.ndarray,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for rep, seed in enumerate(split_seeds):
        rng = check_random_state(int(seed))
        cv_pool_mask = _sample_cv_pool(observed_mask, pool_fraction, rng)
        for fold, validation_mask in enumerate(_validation_masks(cv_pool_mask, n_folds, rng)):
            train_mask = cv_pool_mask & ~validation_mask
            diag = np.diag_indices_from(train_mask)
            train_mask[diag] = observed_mask[diag]
            splits.append((rep, fold, train_mask, validation_mask))
    return splits


def _sample_cv_pool(
    observed_mask: np.ndarray,
    pool_fraction: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    n = observed_mask.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    observed_pairs = np.flatnonzero(observed_mask[triu_i, triu_j])
    if observed_pairs.size == 0:
        raise ValueError("cross-validation needs observed off-diagonal entries")

    n_keep = max(1, int(pool_fraction * observed_pairs.size))
    kept_pairs = rng.choice(observed_pairs, size=n_keep, replace=False)
    pool_mask = np.zeros_like(observed_mask, dtype=bool)
    pool_mask[triu_i[kept_pairs], triu_j[kept_pairs]] = True
    pool_mask[triu_j[kept_pairs], triu_i[kept_pairs]] = True
    return pool_mask


def _validation_masks(
    cv_pool_mask: np.ndarray,
    n_folds: int,
    rng: np.random.RandomState,
) -> list[np.ndarray]:
    n = cv_pool_mask.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    pool_pairs = np.flatnonzero(cv_pool_mask[triu_i, triu_j])
    if pool_pairs.size < n_folds:
        raise ValueError(
            f"cross-validation needs at least {n_folds} observed off-diagonal entries"
        )

    masks = []
    for fold_pairs in np.array_split(rng.permutation(pool_pairs), n_folds):
        mask = np.zeros((n, n), dtype=bool)
        mask[triu_i[fold_pairs], triu_j[fold_pairs]] = True
        mask[triu_j[fold_pairs], triu_i[fold_pairs]] = True
        masks.append(mask)
    return masks


def _fit_jobs(
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    ranks: list[int],
    fit_seeds: np.ndarray,
):
    for rep, fold, train_mask, validation_mask in splits:
        for rank_idx, rank in enumerate(ranks):
            yield (
                rep, fold, int(rank),
                train_mask, validation_mask,
                int(fit_seeds[rep, fold, rank_idx]),
            )


def _cv_pool_fraction(sampling_fraction: float, n_folds: int, n: int) -> float:
    cap = _adaptive_cap(n)
    requested = sampling_fraction * n_folds / (n_folds - 1)
    if requested > cap:
        effective = cap * (n_folds - 1) / n_folds
        warnings.warn(
            f"Requested CV pool fraction {requested:.3f} exceeds the cap {cap:.3f}. "
            f"Each fold will train at {effective:.3f} instead of "
            f"{sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(requested, cap)


def _fit_score(
    s: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    seed: int,
    srf_kwargs: dict,
) -> float:
    with threadpool_limits(limits=1):
        train_matrix = np.full(s.shape, np.nan, dtype=np.float64)
        train_matrix[train_mask] = s[train_mask]
        est = SRF(
            rank=rank, bounds=bounds, missing_values=np.nan,
            random_state=seed, **srf_kwargs,
        )
        est.fit(train_matrix)
        s_hat = est.reconstruct()

        triu = np.triu_indices(s.shape[0], k=1)
        validation_entries = (
            validation_mask[triu] & np.isfinite(s[triu]) & np.isfinite(s_hat[triu])
        )
        if not validation_entries.any():
            return float("nan")
        residual = s[triu][validation_entries] - s_hat[triu][validation_entries]
        return float(np.mean(residual * residual))
