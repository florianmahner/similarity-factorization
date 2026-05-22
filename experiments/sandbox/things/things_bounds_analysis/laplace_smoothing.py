"""Analyze effect of Laplace smoothing on THINGS RSM and bounds."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.colors import ROSE, TEAL, CYAN
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir
from pysrf.bounds import compute_effective_dimension, estimate_sampling_bounds_ultra
from scipy.sparse.linalg import eigsh

OUTPUT_DIR = get_output_dir()

N_OBJECTS = 1854


def build_rsm_laplace(triplets, n, alpha=0.0):
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

    # Laplace smoothing: (count + α) / (shown + 2α)
    # When α > 0: unobserved pairs get (0 + α) / (0 + 2α) = 0.5 automatically
    # When α = 0: unobserved pairs get 0/0 = NaN, set to 0
    if alpha > 0:
        similarity = (counts + alpha) / (shown + 2 * alpha)
    else:
        with np.errstate(invalid='ignore'):
            similarity = np.divide(counts, shown, out=np.zeros_like(counts), where=shown != 0)

    np.fill_diagonal(similarity, 1.0)
    return similarity, shown


def analyze_rsm(S, name):
    """Compute key statistics for RSM."""
    eigvals, _ = eigsh(S, k=20, which='LM')
    eigvals = np.sort(eigvals)[::-1]

    fro = np.linalg.norm(S, 'fro')
    spec = eigvals[0]
    eff_dim = compute_effective_dimension(fro, spec)

    return {
        'name': name,
        'eigvals': eigvals,
        'fro_norm': fro,
        'spec_norm': spec,
        'eff_dim': eff_dim,
        'lambda_ratio_12': eigvals[0] / eigvals[1],
        'lambda_ratio_23': eigvals[1] / eigvals[2],
    }


def main():
    print("Loading triplets...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')

    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    print("\nBuilding RSMs with different smoothing...")
    rsms = {}
    for alpha in alphas:
        print(f"  α = {alpha}...")
        S, shown = build_rsm_laplace(train_triplets, N_OBJECTS, alpha)
        rsms[alpha] = S
        stats = analyze_rsm(S, f"α={alpha}")
        stats['alpha'] = alpha
        results.append(stats)

    # =========================================================================
    # Figure 1: Eigenvalue spectra for different alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for i, alpha in enumerate(alphas):
        stats = results[i]
        ax.semilogy(range(1, 21), stats['eigvals'], 'o-',
                    color=colors[i], markersize=4, linewidth=1.5,
                    label=f"α={alpha} (λ₁/λ₂={stats['lambda_ratio_12']:.2f})")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Effect of Laplace smoothing on eigenvalue spectrum")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eigenvalue_vs_alpha.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: λ₁/λ₂ ratio vs alpha
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ratios_12 = [r['lambda_ratio_12'] for r in results]
    ratios_23 = [r['lambda_ratio_23'] for r in results]

    axes[0].plot(alphas, ratios_12, 'o-', color=TEAL, linewidth=2, markersize=8)
    axes[0].set_xlabel("Smoothing parameter α")
    axes[0].set_ylabel("λ₁/λ₂ ratio")
    axes[0].set_title("First eigenvalue gap vs smoothing")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(8.63, color=ROSE, linestyle='--', label='mur92 (8.63)')
    axes[0].legend()

    axes[1].plot(alphas, ratios_23, 'o-', color=CYAN, linewidth=2, markersize=8)
    axes[1].set_xlabel("Smoothing parameter α")
    axes[1].set_ylabel("λ₂/λ₃ ratio")
    axes[1].set_title("Second eigenvalue gap vs smoothing")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eigenvalue_ratios_vs_alpha.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Value distributions
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, alpha in enumerate(alphas):
        S = rsms[alpha]
        triu_idx = np.triu_indices(N_OBJECTS, k=1)
        values = S[triu_idx]

        axes[i].hist(values, bins=50, alpha=0.7, color=colors[i], density=True)
        axes[i].axvline(np.mean(values), color='red', linestyle='--', linewidth=2)
        axes[i].set_xlabel("Similarity value")
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"α = {alpha}")
        axes[i].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "value_distributions_vs_alpha.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Compute actual bounds for select alpha values (in parallel)
    # =========================================================================
    print("\n" + "="*70)
    print("COMPUTING ACTUAL BOUNDS IN PARALLEL...")
    print("="*70)

    from joblib import Parallel, delayed

    test_alphas = [0.0, 1.0, 2.0]  # Test a few key values

    def compute_bounds_for_alpha(alpha):
        S = rsms[alpha]
        pmin, pmax, _ = estimate_sampling_bounds_ultra(
            S, random_state=42, verbose=False, n_jobs=1
        )
        return {'alpha': alpha, 'pmin': pmin, 'pmax': pmax, 'mean': (pmin + pmax) / 2}

    bounds_results = Parallel(n_jobs=len(test_alphas))(
        delayed(compute_bounds_for_alpha)(alpha) for alpha in test_alphas
    )

    for r in bounds_results:
        print(f"α={r['alpha']}: pmin={r['pmin']:.4f}, pmax={r['pmax']:.4f}, mean={r['mean']:.4f}")

    # =========================================================================
    # Figure 4: Bounds vs alpha
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas_bounds = [r['alpha'] for r in bounds_results]
    pmins = [r['pmin'] for r in bounds_results]
    pmaxs = [r['pmax'] for r in bounds_results]
    means = [r['mean'] for r in bounds_results]

    ax.plot(alphas_bounds, pmins, 'o-', color=TEAL, linewidth=2, markersize=10, label='pmin')
    ax.plot(alphas_bounds, pmaxs, 's-', color=ROSE, linewidth=2, markersize=10, label='pmax')
    ax.plot(alphas_bounds, means, '^--', color=CYAN, linewidth=2, markersize=10, label='mean')

    ax.set_xlabel("Smoothing parameter α")
    ax.set_ylabel("Sampling fraction")
    ax.set_title("Effect of Laplace smoothing on sampling bounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bounds_vs_alpha.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n--- Eigenvalue Structure ---")
    print(f"{'Alpha':<10} {'λ₁/λ₂':>10} {'λ₂/λ₃':>10} {'eff_dim':>10}")
    print("-"*45)
    for r in results:
        print(f"{r['alpha']:<10} {r['lambda_ratio_12']:>10.2f} {r['lambda_ratio_23']:>10.2f} {r['eff_dim']:>10}")

    print("\n--- Sampling Bounds ---")
    print(f"{'Alpha':<10} {'pmin':>10} {'pmax':>10} {'mean':>10}")
    print("-"*45)
    for r in bounds_results:
        print(f"{r['alpha']:<10} {r['pmin']:>10.4f} {r['pmax']:>10.4f} {r['mean']:>10.4f}")

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
