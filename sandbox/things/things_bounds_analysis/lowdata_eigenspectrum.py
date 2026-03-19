"""Analyze eigenspectrum structure at different data percentages."""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

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

    if alpha > 0:
        S = (counts + alpha) / (shown + 2 * alpha)
    else:
        # No smoothing: unobserved pairs get 0
        with np.errstate(invalid='ignore'):
            S = np.divide(counts, shown, out=np.zeros_like(counts), where=shown > 0)

    np.fill_diagonal(S, 1.0)
    return S, shown


def subsample_triplets(triplets, percentage, seed=42):
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


def main():
    print("Loading triplets...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')

    percentages = [0.05, 0.10, 0.20, 1.0]
    alphas = [0.0, 1.0]

    results = []

    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"ALPHA = {alpha}")
        print("#"*70)

        for pct in percentages:
            print(f"\n{'='*60}")
            print(f"Processing {pct*100:.0f}% of data (alpha={alpha})")
            print("="*60)

            triplets_sub = subsample_triplets(train_triplets, pct)
            S, shown = build_rsm_laplace(triplets_sub, N_OBJECTS, alpha=alpha)

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(S)[::-1]  # Descending order

            # Compute statistics
            n_pairs = N_OBJECTS * (N_OBJECTS - 1) // 2
            n_observed = np.sum(shown > 0) // 2
            obs_fraction = n_observed / n_pairs

            # Effective dimension
            fro_norm = np.linalg.norm(S, 'fro')
            spec_norm = eigenvalues[0]
            eff_dim = (fro_norm / spec_norm) ** 2

            # Variance explained by top-k
            total_var = np.sum(eigenvalues**2)
            var_explained = {}
            for k in [2, 5, 10, 20, 50]:
                var_explained[k] = np.sum(eigenvalues[:k]**2) / total_var

            print(f"  Observed pairs: {n_observed:,} / {n_pairs:,} ({obs_fraction*100:.1f}%)")
            print(f"  Effective dimension: {eff_dim:.2f}")
            print(f"  Top eigenvalues: {eigenvalues[:5]}")
            print(f"  Variance explained:")
            for k, v in var_explained.items():
                print(f"    Top {k}: {v*100:.1f}%")

            results.append({
                'alpha': alpha,
                'percentage': pct,
                'eigenvalues': eigenvalues,
                'obs_fraction': obs_fraction,
                'eff_dim': eff_dim,
                'var_explained': var_explained,
            })

    # Print summary comparison
    print("\n" + "="*80)
    print("COMPARISON: Alpha=0 vs Alpha=1")
    print("="*80)
    print(f"{'%Data':<10} {'Alpha':<8} {'EffDim':<10} {'Top5 Var%':<12} {'Top10 Var%':<12} {'Top20 Var%':<12}")
    print("-"*64)
    for res in results:
        print(f"{res['percentage']*100:<10.0f} {res['alpha']:<8} {res['eff_dim']:<10.2f} "
              f"{res['var_explained'][5]*100:<12.1f} {res['var_explained'][10]*100:<12.1f} "
              f"{res['var_explained'][20]*100:<12.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
