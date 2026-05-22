from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import isotonic_regression


@dataclass(frozen=True)
class LossCurve:
    sampling_grid: np.ndarray  # ascending
    raw: np.ndarray            # 1 - recovered_mass / reference_mass, at each p
    monotone: np.ndarray       # isotonic projection of raw (non-increasing in p)
    floor: float               # random-matrix lower bound on the calibrated p


def _calibrate_sampling_fraction(
    recovered_spectral_mass: np.ndarray,
    top_eigenvalues: np.ndarray,
    sampling_grid: np.ndarray,
    rank: int,
    recovery_tolerance: float,
) -> tuple[float, LossCurve]:
    recovered_median = np.median(recovered_spectral_mass, axis=2)
    loss_curve = _recovery_loss_curve(sampling_grid, top_eigenvalues, recovered_median, rank)
    sampling_fraction = max(
        _smallest_p_below_tolerance(
            loss_curve.sampling_grid, loss_curve.monotone, recovery_tolerance,
        ),
        loss_curve.floor,
    )
    return float(sampling_fraction), loss_curve


def _recovery_loss_curve(
    sampling_grid: np.ndarray,
    top_eigenvalues: np.ndarray,
    recovered_median: np.ndarray,
    rank: int,
) -> LossCurve:
    reference_mass = max(float(top_eigenvalues[:rank].sum()), 1e-12)
    order = np.argsort(sampling_grid)
    sorted_grid = sampling_grid[order]
    raw_loss = 1.0 - recovered_median[rank - 1, order] / reference_mass
    monotone_loss = isotonic_regression(raw_loss, increasing=False).x
    return LossCurve(
        sampling_grid=sorted_grid,
        raw=raw_loss,
        monotone=monotone_loss,
        floor=float(_detectability_floor(top_eigenvalues, rank)),
    )


def _smallest_p_below_tolerance(
    sampling_grid: np.ndarray, monotone_loss: np.ndarray, tolerance: float,
) -> float:
    if monotone_loss[-1] >= tolerance:
        return float(sampling_grid[-1])
    if monotone_loss[0] <= tolerance:
        return float(sampling_grid[0])
    index = int(np.searchsorted(-monotone_loss, -tolerance, side="left"))
    index = max(min(index, len(monotone_loss) - 1), 1)
    lower, upper = monotone_loss[index - 1], monotone_loss[index]
    p_lo, p_hi = sampling_grid[index - 1], sampling_grid[index]
    if lower == upper:
        return float(p_lo)
    fraction = (lower - tolerance) / (lower - upper)
    return float(p_lo + fraction * (p_hi - p_lo))


def _detectability_floor(top_eigenvalues: np.ndarray, rank: int) -> float:
    lam_k = float(top_eigenvalues[rank - 1]) if rank >= 1 else float(top_eigenvalues[0])
    lam_kp1 = float(top_eigenvalues[min(rank, len(top_eigenvalues) - 1)])
    if lam_k <= 1e-12:
        return 1.0
    gap_squared = (lam_kp1 / lam_k) ** 2
    return float(np.clip(gap_squared / (1.0 + gap_squared), 0.0, 1.0))


def _adaptive_cap(n: int, floor: float = 0.95, holdout_budget: int = 2000) -> float:
    n_pairs = n * (n - 1) / 2
    if n_pairs <= 0:
        return floor
    return float(max(floor, 1.0 - holdout_budget / n_pairs))
