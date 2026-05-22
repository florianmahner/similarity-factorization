"""Re-invert the recovery curve at different recovery_tolerance values.

Uses the cached recovery_loss_monotone from estimate_rank — no recompute.
Shows what p* would be for tolerance in {0.20, 0.10, 0.05, 0.02} and
how that compares to the detectability_floor (which is a lower bound).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_OUT = (
    PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def smallest_p_below(grid: np.ndarray, monotone: np.ndarray, tol: float) -> float:
    """Mirror of pysrf._sampling_fraction._smallest_p_below_tolerance."""
    if monotone[-1] >= tol:
        return float(grid[-1])
    if monotone[0] <= tol:
        return float(grid[0])
    idx = int(np.searchsorted(-monotone, -tol, side="left"))
    idx = max(min(idx, len(monotone) - 1), 1)
    lower, upper = monotone[idx - 1], monotone[idx]
    p_lo, p_hi = grid[idx - 1], grid[idx]
    if lower == upper:
        return float(p_lo)
    frac = (lower - tol) / (lower - upper)
    return float(p_lo + frac * (p_hi - p_lo))


def main() -> None:
    TOLS = [0.20, 0.10, 0.05, 0.02, 0.01]

    for name in ["peterson_animals", "vgg16"]:
        payload = json.loads((EXP_OUT / name / "result.json").read_text())
        est = payload["estimate"]
        grid = np.asarray(est["sampling_grid"])
        raw = np.asarray(est["recovery_loss_raw"])
        monotone = np.asarray(est["recovery_loss_monotone"])
        floor = float(est["detectability_floor"])
        rank = int(est["rank"])

        log(f"\n=== {name}  (n={payload['n']}  k_cut={rank}) ===")
        log(f"  detectability_floor = {floor:.4f}")
        log(f"  recovery_loss curve (monotone):")
        for g, m, r in zip(grid, monotone, raw):
            log(f"    p={g:.3f}  raw={r:.4f}  monotone={m:.4f}")

        log("")
        log(f"  {'tolerance':>9}  {'p_raw':>7}  {'p_with_floor':>12}")
        for tol in TOLS:
            p_raw = smallest_p_below(grid, monotone, tol)
            p_final = max(p_raw, floor)
            log(f"  {tol:>9.2f}  {p_raw:>7.4f}  {p_final:>12.4f}")


if __name__ == "__main__":
    main()
