from __future__ import annotations

import numpy as np


def _select_rank(
    coherence: np.ndarray,
    sampling_grid: np.ndarray,
    high_band_quantile: float,
    min_rank: int = 2,
    min_segment: int = 2,
) -> tuple[int, np.ndarray]:
    coherence_median = np.median(coherence, axis=2)
    leakage = _leakage_profile(coherence_median, sampling_grid, high_band_quantile)
    rank = _changepoint(leakage, min_rank=min_rank, min_segment=min_segment)
    return rank, leakage


def _leakage_profile(
    coherence_median: np.ndarray,
    sampling_grid: np.ndarray,
    high_band_quantile: float,
) -> np.ndarray:
    threshold = np.quantile(sampling_grid, high_band_quantile)
    high_band = np.where(sampling_grid >= threshold)[0]
    if high_band.size == 0:
        high_band = np.array([len(sampling_grid) - 1])
    p_band = sampling_grid[high_band]
    p_over_one_minus_p = p_band / np.maximum(1.0 - p_band, 1e-12)
    deviation = (1.0 - coherence_median[:, high_band]) * p_over_one_minus_p[np.newaxis, :]
    return np.median(deviation, axis=1).astype(np.float64)


def _changepoint(leakage: np.ndarray, min_rank: int, min_segment: int) -> int:
    n = len(leakage)
    if n < 2 * min_segment:
        return n
    scores = np.full(n - 1, -np.inf)
    for split in range(min_segment - 1, n - min_segment):
        scores[split] = _f_statistic(leakage[: split + 1], leakage[split + 1 :])
    lower = max(min_rank - 1, 0) if min_rank <= n - 1 else 0
    return int(lower + np.nanargmax(scores[lower:])) + 1


def _f_statistic(left: np.ndarray, right: np.ndarray) -> float:
    n_left, n_right = len(left), len(right)
    within_segment_sse = (
        ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    )
    pooled_variance = within_segment_sse / max(n_left + n_right - 2, 1)
    if pooled_variance <= 1e-12:
        return np.inf
    harmonic_size = n_left * n_right / (n_left + n_right)
    mean_gap_squared = (right.mean() - left.mean()) ** 2
    return harmonic_size * mean_gap_squared / pooled_variance
