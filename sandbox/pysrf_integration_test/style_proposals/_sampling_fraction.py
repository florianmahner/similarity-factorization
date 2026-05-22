# RATIONALE: Refactored to match src/tools/stats.py: one-line module docstring,
# concise type-hinted helpers, no per-file Author/License headers, and no
# Parameters/Returns blocks where the signature already tells the story. Cut
# the hand-rolled PAV (~15 lines) in favour of `scipy.optimize.isotonic_regression`,
# which is byte-equivalent on the inputs this module produces (verified across
# random arrays). Trimmed the prose explaining `gap_squared = (lam_{k+1}/lam_k)**2`
# and the `1 - 2000 / N_pairs` cap: both are clearer from the two-line bodies
# than from a paragraph above them. Promoted the cap constants to default
# arguments of `_adaptive_cap`; nothing else uses them, so module-level globals
# bought nothing. Folded `_recovery_curve` so the deficit calculation reads top
# to bottom without an intermediate `raw_sorted` name. Kept every public symbol
# (`_recovery_curve`, `_invert_recovery`, `_detectability_floor`, `_adaptive_cap`)
# and its signature intact — `_estimate.py` and `cross_validation.py` import
# them unchanged. Behaviour is identical.

"""Recovery curve and sampling-fraction calibration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression


def _recovery_curve(
    sampling_grid: np.ndarray,
    eigenvalues: np.ndarray,
    projected_median: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fraction of top-``rank`` spectral mass missed under masking, sorted by p."""
    reference_mass = max(float(eigenvalues[:rank].sum()), 1e-12)
    order = np.argsort(sampling_grid)
    p_sorted = sampling_grid[order]
    raw = 1.0 - projected_median[rank - 1, order] / reference_mass
    return p_sorted, raw, isotonic_regression(raw, increasing=False).x


def _invert_recovery(p_sorted: np.ndarray, monotone: np.ndarray, tolerance: float) -> float:
    """Smallest ``p`` where the recovery deficit is at or below ``tolerance``."""
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


def _detectability_floor(eigenvalues: np.ndarray, rank: int) -> float:
    """Random-matrix lower bound on the calibrated sampling fraction."""
    lam_k = float(eigenvalues[rank - 1]) if rank >= 1 else float(eigenvalues[0])
    lam_kp1 = float(eigenvalues[min(rank, len(eigenvalues) - 1)])
    if lam_k <= 1e-12:
        return 1.0
    gap_squared = (lam_kp1 / lam_k) ** 2
    return float(np.clip(gap_squared / (1.0 + gap_squared), 0.0, 1.0))


def _adaptive_cap(n: int, floor: float = 0.95, holdout_budget: int = 2000) -> float:
    """Outer-mask cap reserving at least ``holdout_budget`` off-diagonal pairs."""
    n_pairs = n * (n - 1) / 2
    if n_pairs <= 0:
        return floor
    return float(max(floor, 1.0 - holdout_budget / n_pairs))
