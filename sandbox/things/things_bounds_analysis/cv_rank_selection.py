"""Cross-validation for rank selection on THINGS with α=1.0 smoothing."""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.utils import get_output_dir
from pysrf import cross_val_score

OUTPUT_DIR = get_output_dir()

N_OBJECTS = 1854


def build_rsm_laplace(triplets, n, alpha=1.0):
    """Build RSM from triplets with Laplace smoothing."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))

    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    S = (counts + alpha) / (shown + 2 * alpha)
    np.fill_diagonal(S, 1.0)
    return S


def main():
    print("Loading triplets and building RSM with α=1.0...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')
    S = build_rsm_laplace(train_triplets, N_OBJECTS, alpha=1.0)

    # pmax for α=1.0 was ~0.67
    pmax = 0.67

    # Test a range of ranks
    ranks = [2, 5, 10, 20, 30, 40, 50, 60, 70]

    print(f"\nRunning cross-validation at p={pmax} sampling fraction...")
    print(f"Testing ranks: {ranks}")
    print(f"This will run in parallel using all available cores.\n")

    cv_result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=5,
        sampling_fraction=pmax,
        random_state=42,
        n_jobs=-1,  # Use all cores
        verbose=1,
    )

    # Extract results from GridSearchCV object
    results_df = cv_result.cv_results_

    # Compute mean and std per rank
    mean_scores = results_df.groupby('rank')['score'].mean().values
    std_scores = results_df.groupby('rank')['score'].std().values

    # Print results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (Frobenius reconstruction error)")
    print("="*60)
    print(f"{'Rank':<10} {'Mean Error':<15} {'Std':<15}")
    print("-"*40)

    best_idx = np.argmin(mean_scores)  # Lower is better for reconstruction error
    for i, rank in enumerate(ranks):
        marker = " <-- BEST" if i == best_idx else ""
        print(f"{rank:<10} {mean_scores[i]:<15.6f} {std_scores[i]:<15.6f}{marker}")

    print(f"\nOptimal rank: {ranks[best_idx]}")
    print(f"Best mean score: {mean_scores[best_idx]:.6f}")

    # Save results
    np.savez(
        OUTPUT_DIR / "cv_results.npz",
        ranks=ranks,
        mean_scores=mean_scores,
        std_scores=std_scores,
        pmax=pmax,
        alpha=1.0,
    )
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
