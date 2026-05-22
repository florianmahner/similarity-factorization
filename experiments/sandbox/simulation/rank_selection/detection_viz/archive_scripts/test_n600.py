"""Quick test: does n=600 improve rank detection for k=30?

Usage:
    poetry run python sandbox/rank_detection_viz/test_n600.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf.cross_validation import cross_val_score
from utils.simulation import simulation_dirichlet


def test_rank_detection(n: int, k: int, alpha: float, n_reps: int = 5) -> dict:
    """Test rank detection for given parameters."""
    errors = []
    selected_ranks = []

    for seed in range(n_reps):
        rng = np.random.default_rng(seed)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
        S = w @ w.T

        ranks = list(range(max(2, k - 8), k + 10, 2))
        if k not in ranks:
            ranks.append(k)
        ranks = sorted(ranks)

        result = cross_val_score(
            S,
            param_grid={"rank": ranks},
            n_repeats=3,
            estimate_sampling_fraction=True,
            verbose=0,
            n_jobs=-1,
        )
        selected = result.best_params_["rank"]
        errors.append(abs(selected - k))
        selected_ranks.append(selected)

    return {
        "mae": np.mean(errors),
        "std": np.std(errors),
        "selected_ranks": selected_ranks,
    }


def main():
    k = 30
    n_reps = 5

    print(f"Testing rank detection for k={k}")
    print("=" * 50)

    for n in [300, 600]:
        total_obs = n * (n - 1) // 2
        dof = n * k
        ratio = total_obs / dof

        print(f"\nn={n}: total obs/dof = {ratio:.1f}")
        print("-" * 40)

        for alpha in [1.0, 5.0]:
            print(f"  alpha={alpha}...", end=" ", flush=True)
            result = test_rank_detection(n, k, alpha, n_reps)
            print(f"MAE={result['mae']:.1f} ± {result['std']:.1f}, selected={result['selected_ranks']}")


if __name__ == "__main__":
    main()
