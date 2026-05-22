"""Debug factor recovery at different alpha values.

Questions to answer:
1. What does factor recovery actually measure?
2. Why is it ~0.4-0.5 at alpha=10 instead of ~0 (random)?
3. Is the alignment/correlation metric appropriate?

Usage:
    poetry run python sandbox/factor_recovery_debug/run.py
"""

from __future__ import annotations

from pathlib import Path

from src.utils import get_output_dir

import numpy as np
from pysrf import SRF
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy

from utils.simulation import simulation_dirichlet

OUTPUT_DIR = get_output_dir()

N, K = 300, 5


def normalize(w):
    return w / (w.sum(axis=1, keepdims=True) + 1e-10)


def align_factors(true_w, learned_w):
    """Align learned factors to true factors using Hungarian algorithm."""
    sim = normalize(true_w).T @ normalize(learned_w)
    _, col_ind = linear_sum_assignment(-sim)
    return learned_w[:, col_ind]


def factor_correlation(true_w, learned_w):
    """Mean absolute correlation between aligned factors."""
    aligned = align_factors(true_w, learned_w)
    corrs = []
    for i in range(true_w.shape[1]):
        c = np.corrcoef(true_w[:, i], aligned[:, i])[0, 1]
        if not np.isnan(c):
            corrs.append(abs(c))
    return np.mean(corrs) if corrs else 0.0


def random_baseline_correlation(n, k, n_trials=100):
    """What correlation do we expect from random factors?"""
    rng = np.random.default_rng(42)
    corrs = []
    for _ in range(n_trials):
        w1 = rng.dirichlet(np.ones(k), size=n)
        w2 = rng.dirichlet(np.ones(k), size=n)
        corrs.append(factor_correlation(w1, w2))
    return np.mean(corrs), np.std(corrs)


def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("Factor Recovery Debug")
    print("=" * 60)

    # Random baseline
    rand_mean, rand_std = random_baseline_correlation(N, K)
    print(f"\nRandom baseline (uniform Dirichlet): {rand_mean:.4f} ± {rand_std:.4f}")

    print("\n" + "-" * 60)
    print("Analysis by alpha")
    print("-" * 60)

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for alpha in alphas:
        print(f"\n=== Alpha = {alpha} ===")

        # Generate data
        factors = simulation_dirichlet(N, K, alpha, rng)
        rsm = factors @ factors.T

        # Ground truth properties
        row_entropies = [entropy(row) for row in factors]
        mean_entropy = np.mean(row_entropies) / np.log(K)
        max_membership = np.mean(np.max(factors, axis=1))
        rsm_contrast = rsm.max() - rsm.min()

        print(f"Ground truth:")
        print(f"  Normalized entropy: {mean_entropy:.4f}")
        print(f"  Max membership: {max_membership:.4f}")
        print(f"  RSM contrast: {rsm_contrast:.4f}")

        # Fit SRF
        srf = SRF(rank=K, random_state=42)
        srf.fit(rsm)
        learned = srf.w_

        # Learned properties
        learned_entropy = np.mean([entropy(row) for row in learned]) / np.log(K)
        learned_max = np.mean(np.max(learned, axis=1))

        print(f"Learned factors:")
        print(f"  Normalized entropy: {learned_entropy:.4f}")
        print(f"  Max membership: {learned_max:.4f}")

        # Factor recovery
        corr = factor_correlation(factors, learned)
        print(f"Factor recovery: {corr:.4f}")

        # Reconstruction
        recon = srf.reconstruct()
        recon_r2 = np.corrcoef(rsm.flatten(), recon.flatten())[0, 1] ** 2
        print(f"Reconstruction R²: {recon_r2:.4f}")

        # Check individual factor correlations
        aligned = align_factors(factors, learned)
        print("Per-factor correlations:")
        for i in range(K):
            c = np.corrcoef(factors[:, i], aligned[:, i])[0, 1]
            print(f"  Factor {i}: {c:.4f}")

        # Check if learned factors are just uniform
        print(f"Learned factor variance (should be >0 if not uniform):")
        for i in range(K):
            print(f"  Factor {i}: var={aligned[:, i].var():.6f}, range=[{aligned[:, i].min():.4f}, {aligned[:, i].max():.4f}]")

    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    print(f"Random baseline: {rand_mean:.4f}")
    print("If factor recovery at alpha=10 is significantly above random,")
    print("then SRF is still capturing some structure.")


if __name__ == "__main__":
    main()
