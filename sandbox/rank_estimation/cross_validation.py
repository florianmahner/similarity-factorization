"""K-fold cross-validation of SRF rank at a calibrated sampling fraction.

Two pieces:

- :class:`EntryKFold`: splitter that pre-masks at the inflated outer
  probability and partitions the kept entries into ``n_folds`` mutually
  exclusive symmetric folds. Each fold trains at exactly
  ``sampling_fraction`` (after inflation), validates on the rest.

- :func:`cross_val_score`: sweep a list of ranks at a given sampling
  fraction. Returns a long-format DataFrame so the user can aggregate
  however they like (``.groupby("rank")["score"].mean()`` is the usual
  case).

The unit of cross-validation is a matrix *entry*, not a row. That's why
this module doesn't try to plug into ``sklearn.model_selection`` — the
row-based CV abstractions there don't apply to matrix completion.
"""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import warnings
from typing import Generator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from pysrf import SRF


_CV_CAP_FLOOR = 0.95
_CV_HOLDOUT_BUDGET = 2000


def _adaptive_cap(n: int) -> float:
    """Adaptive cap on the outer-mask probability.

    Matches ``recipe_K`` defaults: ``max(0.95, 1 - 2000 / N_pairs)``,
    so at least ~2000 off-diagonal pairs are kept for validation
    regardless of matrix size.
    """
    n_pairs = n * (n - 1) / 2
    if n_pairs <= 0:
        return _CV_CAP_FLOOR
    return float(max(_CV_CAP_FLOOR, 1.0 - _CV_HOLDOUT_BUDGET / n_pairs))


def _is_constant(values: np.ndarray, atol: float = 1e-10) -> bool:
    return bool(np.allclose(values, values[0], atol=atol))


def _eligible_pair_positions(
    x: np.ndarray, missing_values: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle indices of originally-observed entries."""
    if missing_values is None or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    ):
        observed = np.isfinite(x)
    else:
        observed = np.not_equal(x, missing_values)
    rows, cols = np.triu_indices_from(x, k=1)
    eligible = np.where(observed[rows, cols])[0]
    return rows, cols, eligible


def _inflate_outer_mask(
    sampling_fraction: float, n_folds: int, cap: float
) -> float:
    """Outer mask probability so each of n_folds folds trains at sampling_fraction."""
    unclipped = sampling_fraction * n_folds / (n_folds - 1)
    if unclipped > cap:
        effective = cap * (n_folds - 1) / n_folds
        warnings.warn(
            f"Inflated outer mask {unclipped:.3f} for n_folds={n_folds} "
            f"exceeds the cap {cap:.3f}. Each fold will train at "
            f"{effective:.3f} instead of the requested "
            f"{sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(unclipped, cap)


def _bernoulli_keep(
    positions: np.ndarray, p: float, rng: np.random.RandomState
) -> np.ndarray:
    if positions.size == 0:
        return positions
    return positions[rng.uniform(size=positions.size) < p]


def _partition(
    positions: np.ndarray, n_folds: int, rng: np.random.RandomState
) -> list[np.ndarray]:
    if positions.size == 0:
        return [np.array([], dtype=int) for _ in range(n_folds)]
    shuffled = rng.permutation(positions)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled, n_folds)]


def _set_symmetric(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
    mask[rows, cols] = True
    mask[cols, rows] = True


class EntryKFold(BaseCrossValidator):
    """K-fold partition cross-validator for symmetric similarity matrices.

    Pre-masks the observed entries at an inflated outer probability,
    then partitions the kept entries into ``n_folds`` mutually exclusive
    symmetric folds. For each fold the validation set is that fold's
    partition and the training set is the union of the other folds.
    The outer mask is set so every fold trains at ``sampling_fraction``::

        outer_mask = sampling_fraction * n_folds / (n_folds - 1)

    Capped at ``max(0.95, 1 - 2000 / N_pairs)`` so at least ~2000
    off-diagonal pairs are kept for validation regardless of ``n``
    (matches ``recipe_K``'s ``M_min=2000, p_max_floor=0.95``). When
    the cap binds, a warning is emitted and each fold trains slightly
    below ``sampling_fraction``.

    Constant diagonals (e.g. all-ones in RBF kernels) are excluded.
    Variable diagonals are partitioned across folds.

    Parameters
    ----------
    n_folds : int, default=5
    sampling_fraction : float, default=0.8
        Per-fold training density.
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
        self.n_folds = n_folds
        self.sampling_fraction = sampling_fraction
        self.random_state = random_state
        self.missing_values = missing_values
        if int(self.n_folds) < 2:
            raise ValueError(f"n_folds must be at least 2, got {self.n_folds}")
        if not (0.0 < float(self.sampling_fraction) < 1.0):
            raise ValueError("sampling_fraction must be in (0, 1)")

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
        return int(self.n_folds)

    def split(
        self, x: np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield ``(train_mask, val_mask)`` for each of the ``n_folds`` folds.

        Diagonal entries are always observed (in ``train_mask``) and never
        validated (never in ``val_mask``). Only off-diagonal entries
        participate in the outer mask + inner partition. This matches the
        standard symmetric-matrix completion convention: validation MSE
        is computed on off-diagonal entries only.
        """
        rng = check_random_state(self.random_state)
        p_outer = _inflate_outer_mask(
            float(self.sampling_fraction),
            int(self.n_folds),
            _adaptive_cap(x.shape[0]),
        )

        rows, cols, eligible = _eligible_pair_positions(x, self.missing_values)
        pool = _bernoulli_keep(eligible, p_outer, rng)
        folds = _partition(pool, int(self.n_folds), rng)
        diag_in_training = self._observed_diagonal_indices(x)

        for fold_idx in range(len(folds)):
            train_mask = np.zeros(x.shape, dtype=bool)
            val_mask = np.zeros(x.shape, dtype=bool)

            val_positions = folds[fold_idx]
            train_positions = np.setdiff1d(pool, val_positions, assume_unique=False)
            _set_symmetric(val_mask, rows[val_positions], cols[val_positions])
            _set_symmetric(train_mask, rows[train_positions], cols[train_positions])

            train_mask[diag_in_training, diag_in_training] = True

            yield train_mask, val_mask

    def _observed_diagonal_indices(self, x: np.ndarray) -> np.ndarray:
        """Indices of originally-observed diagonal entries (always in training)."""
        diag = np.diag(x)
        if self.missing_values is None or (
            isinstance(self.missing_values, float) and np.isnan(self.missing_values)
        ):
            observed = np.isfinite(diag)
        else:
            observed = np.not_equal(diag, self.missing_values)
        return np.where(observed)[0]


def _fit_score_one(
    s: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rank: int,
    bounds: tuple[float, float],
    random_state: int,
) -> float:
    """Fit SRF at ``rank`` on training entries, score V-MSE on validation."""
    x_train = np.full_like(s, np.nan)
    x_train[train_mask] = s[train_mask]

    est = clone(SRF(rank=rank, bounds=bounds, missing_values=np.nan,
                    random_state=random_state))
    est.fit(x_train)
    reconstruction = est.reconstruct()

    if not val_mask.any():
        return float("nan")
    return float(np.mean((s[val_mask] - reconstruction[val_mask]) ** 2))


def cross_val_score(
    s: np.ndarray,
    ranks: list[int],
    sampling_fraction: float,
    n_folds: int = 5,
    random_state: int = 0,
    n_jobs: int = -1,
    missing_values: float | None = np.nan,
) -> pd.DataFrame:
    """Sweep SRF ranks via k-fold CV at the given sampling fraction.

    Each of the ``n_folds`` folds trains at exactly ``sampling_fraction``
    (after the outer-mask inflation handled by :class:`EntryKFold`) and
    validates on its held-out partition.

    Parameters
    ----------
    s : ndarray of shape (n, n)
        Symmetric similarity matrix. Missing entries marked according
        to ``missing_values``.
    ranks : list of int
        Candidate ranks to score.
    sampling_fraction : float
        Per-fold training density. Typically
        ``RankEstimator(...).fit(s).sampling_fraction_``.
    n_folds : int, default=5
    random_state : int, default=0
    n_jobs : int, default=-1
    missing_values : float or None, default=np.nan

    Returns
    -------
    curve : DataFrame
        Long-format with columns ``rank``, ``fold``, ``score``. Aggregate
        with ``curve.groupby("rank")["score"].mean()`` for the CV curve.
    """
    cv = EntryKFold(
        n_folds=n_folds,
        sampling_fraction=sampling_fraction,
        random_state=random_state,
        missing_values=missing_values,
    )
    bounds = (float(np.nanmin(s)), float(np.nanmax(s)))
    splits = list(cv.split(s))

    jobs = [
        (rank_idx, fold_idx, rank, train_mask, val_mask)
        for fold_idx, (train_mask, val_mask) in enumerate(splits)
        for rank_idx, rank in enumerate(ranks)
    ]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score_one)(
            s, tm, vm, int(rank), bounds,
            random_state + 1000 * fold_idx + rank_idx,
        )
        for (rank_idx, fold_idx, rank, tm, vm) in jobs
    )
    return pd.DataFrame([
        {"rank": int(rank), "fold": fold_idx, "score": score}
        for (_, fold_idx, rank, *_), score in zip(jobs, scores)
    ])
