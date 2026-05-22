# RATIONALE: Stripped the file from 75 lines of mostly prose to a compact core
# while preserving byte-identical arithmetic. Dropped the per-function Author/
# License headers (they belong at the package level, not per file) to match
# stats.py. Collapsed the module docstring to one line and function docstrings
# to summary sentences -- the long explanations duplicated content already in
# _estimate.py's CoherenceProfile docstring. Killed `_segment_sse`: a one-line
# "sum of squared deviations" helper carries no abstraction value, and inlining
# both calls into `_f_statistic` makes the formula readable in place. Replaced
# the `valid`-mask gymnastics in `_changepoint` (build full array, post-filter
# with a `>= min_rank` boolean mask, fall back if empty) with a single lower
# bound `lo` and a sliced argmax -- equivalent under every `min_rank` regime I
# traced, including the original's `valid.size == 0` fallback when `min_rank`
# exceeds the array. Deliberately did NOT vectorise the F-statistic sweep with
# cumsum: that would change the arithmetic from `sum((x-mean)**2)` to
# `sum(x**2) - sum(x)**2/n`, which is mathematically equivalent but not
# byte-identical, and the loop is O(n) anyway -- not a hot path.

"""Per-dimension leakage and F-statistic changepoint."""

from __future__ import annotations

import numpy as np


# ------- Leakage profile ------- #


def _leakage_profile(
    overlap_median: np.ndarray,
    sampling_grid: np.ndarray,
    high_band_quantile: float,
) -> np.ndarray:
    """Per-dimension leakage aggregated over the high-p band of the sampling grid."""
    threshold = np.quantile(sampling_grid, high_band_quantile)
    band = np.where(sampling_grid >= threshold)[0]
    if band.size == 0:
        band = np.array([len(sampling_grid) - 1])
    p_band = sampling_grid[band]
    scale = p_band / np.maximum(1.0 - p_band, 1e-12)
    deviation = (1.0 - overlap_median[:, band]) * scale[np.newaxis, :]
    return np.median(deviation, axis=1).astype(np.float64)


# ------- F-statistic changepoint ------- #


def _changepoint(leakage: np.ndarray, min_rank: int = 2, min_segment: int = 2) -> int:
    """Rank at the two-segment F-statistic maximum of the leakage profile."""
    n = len(leakage)
    if n < 2 * min_segment:
        return n
    scores = np.full(n - 1, -np.inf)
    for split in range(min_segment - 1, n - min_segment):
        scores[split] = _f_statistic(leakage[: split + 1], leakage[split + 1 :])
    lo = max(min_rank - 1, 0) if min_rank <= n - 1 else 0
    return int(lo + np.nanargmax(scores[lo:])) + 1


def _f_statistic(left: np.ndarray, right: np.ndarray) -> float:
    """F-statistic for a two-mean comparison between two segments."""
    n_left, n_right = len(left), len(right)
    sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    pooled_var = sse / max(n_left + n_right - 2, 1)
    if pooled_var <= 1e-12:
        return np.inf
    harmonic = n_left * n_right / (n_left + n_right)
    return harmonic * (right.mean() - left.mean()) ** 2 / pooled_var
