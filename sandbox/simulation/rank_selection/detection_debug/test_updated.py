"""Quick test of updated rank_detection with simulation_dirichlet."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from utils.simulation import simulation_dirichlet


def make_similarity(n: int, k: int, alpha: float, seed: int) -> np.ndarray:
    """Generate similarity using simulation_dirichlet with linear kernel."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=alpha)
    return w @ w.T


def evaluate(n: int, true_rank: int, alpha: float, seed: int) -> dict:
    """Evaluate rank detection for a single condition."""
    similarity = make_similarity(n, true_rank, alpha, seed)

    lower = max(2, true_rank - 10)
    upper = true_rank + 10
    candidate_ranks = list(range(lower, upper + 1, 2))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    cv_results = cross_val_score(
        similarity,
        estimator=SRF(init="random_sqrt", random_state=seed, max_outer=50, max_inner=30),
        param_grid={"rank": candidate_ranks},
        n_repeats=5,
        sampling_fraction=0.8,
        random_state=seed,
        verbose=0,
        n_jobs=1,
        fit_final_estimator=False,
    )

    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()
    best_rank = int(mean_scores.idxmin())

    return {
        "true_rank": true_rank,
        "alpha": alpha,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - true_rank),
        "is_correct": best_rank == true_rank,
    }


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Test parameters
    n = 200
    true_ranks = [10, 20, 30]
    alphas = [0.1, 0.5, 1.0]
    n_seeds = 3

    conditions = [
        (n, tr, a, s)
        for tr in true_ranks
        for a in alphas
        for s in range(n_seeds)
    ]

    print(f"Testing {len(conditions)} conditions...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate)(n, tr, a, s)
        for n, tr, a, s in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "test_results.csv", index=False)

    print("\nResults by alpha and true_rank:")
    for alpha in alphas:
        print(f"\nalpha={alpha}:")
        for tr in true_ranks:
            sub = df[(df["alpha"] == alpha) & (df["true_rank"] == tr)]
            acc = sub["is_correct"].mean()
            mae = sub["abs_error"].mean()
            print(f"  k={tr}: Acc={acc:.0%}, MAE={mae:.1f}")

    # Test plot function
    print("\nTesting plot function...")
    sys.path.insert(0, str(Path(__file__).parents[2] / "experiments" / "simulation"))
    from plotting import create_rank_detection_plot
    create_rank_detection_plot(df, output_dir)
    print(f"Saved plot to {output_dir / 'rank_detection.pdf'}")


if __name__ == "__main__":
    main()
