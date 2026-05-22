"""Per-dimension leakage and F-statistic changepoint."""

from __future__ import annotations

import numpy as np


def leakage_profile(
    overlap_median: np.ndarray,
    sampling_grid: np.ndarray,
    high_band_quantile: float,
) -> np.ndarray:
    """Per-dimension scaled leakage, median over the high-p band."""
    raise NotImplementedError


def changepoint(leakage: np.ndarray, min_rank: int = 2, min_segment: int = 2) -> int:
    """Two-segment F-statistic changepoint."""
    raise NotImplementedError
