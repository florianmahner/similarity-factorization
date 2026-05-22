"""Recovery curve and sampling-fraction calibration."""

from __future__ import annotations

import numpy as np


def recovery_curve(
    sampling_grid: np.ndarray,
    eigenvalues: np.ndarray,
    projected_median: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_sorted, raw_deficit, monotone_deficit)."""
    raise NotImplementedError


def invert_recovery(
    p_sorted: np.ndarray, monotone: np.ndarray, tolerance: float
) -> float:
    """Smallest ``p`` at which deficit ≤ ``tolerance``."""
    raise NotImplementedError


def detectability_floor(eigenvalues: np.ndarray, rank: int) -> float:
    """BBP-style floor ``x/(1+x)`` with ``x = (λ_{r+1}/λ_r)²``."""
    raise NotImplementedError
