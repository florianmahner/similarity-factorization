"""Test dimensionality estimation at different data percentages with Laplace smoothing."""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.utils import get_output_dir
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra

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


def subsample_triplets(triplets, percentage, seed=42):
    """Subsample triplets to given percentage."""
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


def main():
    print("Loading triplets...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')
    print(f"Total triplets: {len(train_triplets):,}")

    percentages = [0.05, 0.10, 0.20, 1.0]
    ranks_to_test = [2, 5, 10, 15, 20, 30, 40, 50]
    fixed_sampling = 0.5  # Use fixed sampling for fair comparison

    results = []

    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Processing {pct*100:.0f}% of data ({int(len(train_triplets)*pct):,} triplets)")
        print("="*60)

        # Subsample and build RSM
        triplets_sub = subsample_triplets(train_triplets, pct)
        S = build_rsm_laplace(triplets_sub, N_OBJECTS, alpha=1.0)

        # Estimate bounds
        print("Estimating bounds...")
        pmin, pmax, _ = estimate_sampling_bounds_ultra(S)
        pmean = (pmin + pmax) / 2

        print(f"  pmin = {pmin:.4f}")
        print(f"  pmax = {pmax:.4f}")
        print(f"  pmean = {pmean:.4f}")

        # Run cross-validation at fixed sampling fraction for fair comparison
        # Also run at pmean for bounds-informed estimate
        print(f"\nRunning CV at fixed p={fixed_sampling:.2f}...")
        cv_result = cross_val_score(
            S,
            param_grid={"rank": ranks_to_test},
            n_repeats=3,
            sampling_fraction=fixed_sampling,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        # Get results
        results_df = cv_result.cv_results_
        mean_scores = results_df.groupby('rank')['score'].mean()
        std_scores = results_df.groupby('rank')['score'].std()

        best_rank = mean_scores.idxmin()
        best_score = mean_scores.min()

        print(f"\nCV Results:")
        print(f"{'Rank':<10} {'Mean Error':<15} {'Std':<15}")
        print("-"*40)
        for rank in ranks_to_test:
            marker = " <-- BEST" if rank == best_rank else ""
            print(f"{rank:<10} {mean_scores[rank]:<15.6f} {std_scores[rank]:<15.6f}{marker}")

        results.append({
            'percentage': pct,
            'n_triplets': len(triplets_sub),
            'pmin': pmin,
            'pmax': pmax,
            'pmean': pmean,
            'optimal_rank': best_rank,
            'best_cv_error': best_score,
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Dimensionality vs Data Percentage")
    print("="*70)
    print(f"{'%Data':<10} {'#Triplets':<12} {'pmin':<10} {'pmax':<10} {'pmean':<10} {'Opt.Rank':<10}")
    print("-"*62)
    for r in results:
        print(f"{r['percentage']*100:<10.0f} {r['n_triplets']:<12,} {r['pmin']:<10.4f} {r['pmax']:<10.4f} {r['pmean']:<10.4f} {r['optimal_rank']:<10}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
