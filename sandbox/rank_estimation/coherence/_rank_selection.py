"""Per-dimension leakage profile and rank changepoint detection.

Given the bootstrap overlap statistics, computes a per-dimension
*leakage* score: how much each eigenspace dimension fails to overlap
the reference under masking, scaled by the perturbation magnitude.
Signal dimensions have a small, roughly constant leakage; noise
dimensions have a leakage that blows up. The boundary between the two
regimes is found by a two-segment F-statistic changepoint detector.
"""

from __future__ import annotations

import numpy as np


def leakage_profile(
    overlap_median: np.ndarray,
    sampling_grid: np.ndarray,
    high_band_quantile: float,
) -> np.ndarray:
    """Per-dimension leakage aggregated over the high-p band.

    For each rank ``r`` and sampling probability ``p``, the scaled
    deviation from a perfectly recovered subspace is

        deviation[r, p] = (1 - overlap_median[r, p]) * p / (1 - p)

    Signal ranks have a bounded constant deviation; noise ranks have a
    deviation that grows with ``p``. The leakage profile takes the
    median deviation across the top ``high_band_quantile`` of
    ``sampling_grid``, where the signal/noise contrast is sharpest.
    """
    threshold = np.quantile(sampling_grid, high_band_quantile)
    band = np.where(sampling_grid >= threshold)[0]
    if band.size == 0:
        band = np.array([len(sampling_grid) - 1])
    p_band = sampling_grid[band]
    scale = p_band / np.maximum(1.0 - p_band, 1e-12)
    deviation = (1.0 - overlap_median[:, band]) * scale[np.newaxis, :]
    return np.median(deviation, axis=1).astype(np.float64)


def _segment_sse(values: np.ndarray) -> float:
    """Sum of squared deviations from the segment mean."""
    return float(((values - values.mean()) ** 2).sum())


def _f_statistic(left: np.ndarray, right: np.ndarray) -> float:
    """F-statistic for a two-mean comparison between two segments."""
    n_left, n_right = len(left), len(right)
    pooled_var = (_segment_sse(left) + _segment_sse(right)) / max(n_left + n_right - 2, 1)
    if pooled_var <= 1e-12:
        return np.inf
    harmonic = n_left * n_right / (n_left + n_right)
    return harmonic * (right.mean() - left.mean()) ** 2 / pooled_var


def changepoint(
    leakage: np.ndarray, min_rank: int = 2, min_segment: int = 2
) -> int:
    """Rank at the two-segment split that maximizes the F-statistic.

    Sweeps every valid split of the leakage profile into a low-flat
    "signal" segment followed by a high "noise" segment, and returns
    the rank at which the F-statistic of the two-mean comparison is
    largest. ``min_rank`` rules out splits that would put the boundary
    at trivially small ranks; ``min_segment`` enforces at least that
    many points per segment.
    """
    n = len(leakage)
    if n < 2 * min_segment:
        return n
    scores = np.full(n - 1, -np.inf)
    for split in range(min_segment - 1, n - min_segment):
        scores[split] = _f_statistic(leakage[: split + 1], leakage[split + 1 :])
    valid = np.arange(n - 1)
    valid = valid[valid + 1 >= min_rank]
    if valid.size == 0:
        valid = np.arange(n - 1)
    best = int(valid[np.nanargmax(scores[valid])])
    return best + 1
