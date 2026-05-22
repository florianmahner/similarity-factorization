"""CV for optimal rank at 5% data WITHOUT Laplace smoothing."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra

N_OBJECTS = 1854


def build_rsm(triplets, n, alpha=0.0):
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

    if alpha > 0:
        S = (counts + alpha) / (shown + 2 * alpha)
    else:
        with np.errstate(invalid='ignore'):
            S = np.divide(counts, shown, out=np.zeros_like(counts), where=shown > 0)

    np.fill_diagonal(S, 1.0)
    return S


def subsample_triplets(triplets, percentage, seed=42):
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed)
    return triplets[rng.choice(len(triplets), size=n_samples, replace=False)]


def main():
    print("Loading triplets...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')

    print("\nBuilding RSM at 5% data, alpha=0...")
    triplets_5pct = subsample_triplets(train_triplets, 0.05)
    S = build_rsm(triplets_5pct, N_OBJECTS, alpha=0.0)

    # Estimate bounds
    print("Estimating bounds...")
    pmin, pmax, _ = estimate_sampling_bounds_ultra(S)
    pmean = (pmin + pmax) / 2
    print(f"  pmin={pmin:.4f}, pmax={pmax:.4f}, pmean={pmean:.4f}")

    # Run CV
    ranks = [2, 3, 5, 10, 15, 20, 30, 40]
    print(f"\nRunning CV at pmean={pmean:.3f}...")
    print(f"Testing ranks: {ranks}")

    cv_result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=3,
        sampling_fraction=pmean,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    results_df = cv_result.cv_results_
    mean_scores = results_df.groupby('rank')['score'].mean()
    std_scores = results_df.groupby('rank')['score'].std()

    print(f"\n{'Rank':<10} {'Mean Error':<15} {'Std':<15}")
    print("-"*40)
    best_rank = mean_scores.idxmin()
    for rank in ranks:
        marker = " <-- BEST" if rank == best_rank else ""
        print(f"{rank:<10} {mean_scores[rank]:<15.6f} {std_scores[rank]:<15.6f}{marker}")

    print(f"\nOptimal rank: {best_rank}")


if __name__ == "__main__":
    main()
