"""Compare CV rank selection at pmin, mean, and pmax sampling fractions."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from pysrf import cross_val_score
from src.colors import ROSE, TEAL, CYAN
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

    S = (counts + alpha) / (shown + 2 * alpha)
    np.fill_diagonal(S, 1.0)
    return S


def run_cv_for_fraction(S, sampling_fraction, ranks, n_repeats=5, random_state=42):
    """Run cross-validation for a specific sampling fraction."""
    cv_result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=n_repeats,
        sampling_fraction=sampling_fraction,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    results_df = cv_result.cv_results_
    mean_scores = results_df.groupby('rank')['score'].mean().values
    std_scores = results_df.groupby('rank')['score'].std().values
    return mean_scores, std_scores


def main():
    print("Loading triplets and building RSM with α=1.0...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')
    S = build_rsm_laplace(train_triplets, N_OBJECTS, alpha=1.0)

    # Bounds for α=1.0 THINGS
    pmin = 0.0084
    pmax = 0.67
    pmean = (pmin + pmax) / 2

    sampling_fractions = {
        'pmin': pmin,
        'mean': pmean,
        'pmax': pmax,
    }

    # Ranks to test: 30 to 60 in steps of 5
    ranks = list(range(30, 65, 5))  # [30, 35, 40, 45, 50, 55, 60]

    print(f"\nSampling fractions:")
    print(f"  pmin = {pmin:.4f}")
    print(f"  mean = {pmean:.4f}")
    print(f"  pmax = {pmax:.4f}")
    print(f"\nTesting ranks: {ranks}")
    print(f"\nRunning cross-validation sequentially...\n")

    # Run CV for each sampling fraction sequentially (each uses internal parallelization)
    all_results = {}
    for name, frac in sampling_fractions.items():
        print(f"  Running {name} (p={frac:.4f})...")
        mean_scores, std_scores = run_cv_for_fraction(S, frac, ranks)
        all_results[name] = {
            'mean': mean_scores,
            'std': std_scores,
            'fraction': frac,
        }
        print(f"  Finished {name}")

    # Print results table
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)

    for name in ['pmin', 'mean', 'pmax']:
        res = all_results[name]
        print(f"\n{name.upper()} (p={res['fraction']:.4f}):")
        print(f"{'Rank':<10} {'Mean Error':<15} {'Std':<15}")
        print("-"*40)
        best_idx = np.argmin(res['mean'])
        for i, rank in enumerate(ranks):
            marker = " <-- BEST" if i == best_idx else ""
            print(f"{rank:<10} {res['mean'][i]:<15.6f} {res['std'][i]:<15.6f}{marker}")
        print(f"Optimal rank: {ranks[best_idx]}")

    # =========================================================================
    # Plot 1: CV error vs rank for each sampling fraction
    # =========================================================================
    fig, ax = create_figure("wide")

    colors = {'pmin': ROSE, 'mean': CYAN, 'pmax': TEAL}
    labels = {
        'pmin': f'pmin = {pmin:.3f}',
        'mean': f'mean = {pmean:.2f}',
        'pmax': f'pmax = {pmax:.2f}',
    }

    for name in ['pmin', 'mean', 'pmax']:
        res = all_results[name]
        ax.errorbar(
            ranks, res['mean'], yerr=res['std'],
            marker='o', markersize=6, linewidth=2,
            color=colors[name], label=labels[name],
            capsize=3,
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Reconstruction error (Frobenius)")
    ax.legend()
    ax.set_xticks(ranks)
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "cv_error_vs_rank.pdf")
    print(f"\nSaved: cv_error_vs_rank.pdf")

    # =========================================================================
    # Plot 2: Optimal rank for each sampling fraction
    # =========================================================================
    fig, ax = create_figure("single")

    optimal_ranks = []
    fractions = []
    names = ['pmin', 'mean', 'pmax']

    for name in names:
        res = all_results[name]
        best_idx = np.argmin(res['mean'])
        optimal_ranks.append(ranks[best_idx])
        fractions.append(res['fraction'])

    x = np.arange(len(names))
    bars = ax.bar(x, optimal_ranks, color=[colors[n] for n in names], alpha=0.8)

    # Add value labels on bars
    for bar, rank in zip(bars, optimal_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(rank), ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(p={f:.2f})' for n, f in zip(names, fractions)])
    ax.set_ylabel("Optimal rank")
    ax.set_ylim(0, max(optimal_ranks) + 10)
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "optimal_rank_by_fraction.pdf")
    print(f"Saved: optimal_rank_by_fraction.pdf")

    # =========================================================================
    # Plot 3: Minimum error achieved at each sampling fraction
    # =========================================================================
    fig, ax = create_figure("single")

    min_errors = [np.min(all_results[n]['mean']) for n in names]

    bars = ax.bar(x, min_errors, color=[colors[n] for n in names], alpha=0.8)

    for bar, err in zip(bars, min_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{err:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(p={f:.2f})' for n, f in zip(names, fractions)])
    ax.set_ylabel("Minimum CV error")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "min_error_by_fraction.pdf")
    print(f"Saved: min_error_by_fraction.pdf")

    # Save all results
    np.savez(
        OUTPUT_DIR / "cv_results.npz",
        ranks=ranks,
        pmin_mean=all_results['pmin']['mean'],
        pmin_std=all_results['pmin']['std'],
        mean_mean=all_results['mean']['mean'],
        mean_std=all_results['mean']['std'],
        pmax_mean=all_results['pmax']['mean'],
        pmax_std=all_results['pmax']['std'],
        sampling_fractions=np.array([pmin, pmean, pmax]),
    )

    # Save as CSV for easy viewing
    df_rows = []
    for name in names:
        res = all_results[name]
        for i, rank in enumerate(ranks):
            df_rows.append({
                'sampling': name,
                'fraction': res['fraction'],
                'rank': rank,
                'mean_error': res['mean'][i],
                'std_error': res['std'][i],
            })
    df = pd.DataFrame(df_rows)
    df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
