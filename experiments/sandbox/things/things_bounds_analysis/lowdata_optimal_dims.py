"""Optimal dimensionality vs data percentage (no smoothing, CV at pmax)."""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra
from src.colors import TEAL
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

N_OBJECTS = 1854


def build_rsm(triplets, n):
    """Build RSM without smoothing (alpha=0)."""
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

    percentages = [0.05, 0.10, 0.20, 0.50, 1.0]
    ranks = [2, 5, 10, 15, 20, 30, 40, 50, 60]

    results = []

    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Processing {pct*100:.0f}% of data")
        print("="*60)

        triplets_sub = subsample_triplets(train_triplets, pct)
        S = build_rsm(triplets_sub, N_OBJECTS)

        # Estimate bounds
        print("Estimating bounds...")
        pmin, pmax, _ = estimate_sampling_bounds_ultra(S)
        print(f"  pmin={pmin:.4f}, pmax={pmax:.4f}")

        # Run CV at pmax
        print(f"Running CV at pmax={pmax:.3f}...")
        cv_result = cross_val_score(
            S,
            param_grid={"rank": ranks},
            n_repeats=3,
            sampling_fraction=pmax,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        results_df = cv_result.cv_results_
        mean_scores = results_df.groupby('rank')['score'].mean()

        optimal_rank = mean_scores.idxmin()
        print(f"  Optimal rank: {optimal_rank}")

        results.append({
            'percentage': pct,
            'pmin': pmin,
            'pmax': pmax,
            'optimal_rank': optimal_rank,
            'cv_scores': mean_scores.to_dict(),
        })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'%Data':<10} {'pmax':<10} {'Optimal Rank':<15}")
    print("-"*35)
    for r in results:
        print(f"{r['percentage']*100:<10.0f} {r['pmax']:<10.4f} {r['optimal_rank']:<15}")

    # Plot
    fig, ax = create_figure("single")

    pcts = [r['percentage'] * 100 for r in results]
    opt_ranks = [r['optimal_rank'] for r in results]

    ax.plot(pcts, opt_ranks, 'o-', color=TEAL, linewidth=2, markersize=8)

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Optimal rank (CV at pmax)")
    ax.set_xticks(pcts)

    despine(ax)
    save_figure(fig, OUTPUT_DIR / "optimal_rank_vs_data.pdf")
    print(f"\nSaved: {OUTPUT_DIR / 'optimal_rank_vs_data.pdf'}")


if __name__ == "__main__":
    main()
