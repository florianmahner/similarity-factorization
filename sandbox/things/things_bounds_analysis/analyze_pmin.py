"""Analyze why pmin is so low for THINGS."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.datasets import load_dataset


def build_rsm_laplace(triplets, n, alpha=1.0):
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
    # Load THINGS
    print("Loading THINGS...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')
    n = 1854
    S = build_rsm_laplace(train_triplets, n, alpha=1.0)

    # Load mur92
    print("Loading mur92...")
    ds = load_dataset('mur92', root='/SSD/datasets/similarity_datasets/mur92')
    S_mur = ds.rsm
    n_mur = S_mur.shape[0]

    # Compute pmin components for both
    gamma = 1.05
    eta = 0.05

    def compute_pmin_components(S, name):
        n = S.shape[0]
        row_sq = (S**2).sum(axis=1) - np.diag(S)**2
        L_max = np.max(row_sq)
        L_max_95 = np.quantile(row_sq, 0.95)
        L_inf = 2 * np.max(np.abs(S))
        S_norm = np.linalg.norm(S, 2)
        fro_norm = np.linalg.norm(S, 'fro')
        eff_dim = (fro_norm / S_norm) ** 2

        N = (gamma * L_inf * S_norm) / (3 * L_max_95) + 1
        D = ((gamma * S_norm)**2 / (2 * L_max_95) + 1) / np.log(2 * eff_dim / eta)
        pmin = N / D

        return {
            'name': name,
            'n': n,
            'S_norm': S_norm,
            'fro_norm': fro_norm,
            'L_max': L_max,
            'L_max_95': L_max_95,
            'L_inf': L_inf,
            'eff_dim': eff_dim,
            'N': N,
            'D': D,
            'pmin': pmin,
            'L_ratio': L_max_95 / S_norm**2,
        }

    things = compute_pmin_components(S, "THINGS")
    mur92 = compute_pmin_components(S_mur, "mur92")

    print("\n" + "="*70)
    print("PMIN BOUND ANALYSIS")
    print("="*70)
    print("\nFormula: pmin = N / D")
    print("  N = (gamma * L_inf * ||S||) / (3 * L_max) + 1")
    print("  D = ((gamma * ||S||)^2 / (2 * L_max) + 1) / log(2 * eff_dim / eta)")
    print()

    print(f"{'Metric':<30} {'THINGS':>15} {'mur92':>15} {'Ratio':>10}")
    print("-"*70)
    print(f"{'n':<30} {things['n']:>15} {mur92['n']:>15}")
    print(f"{'||S||_2 (spectral)':<30} {things['S_norm']:>15.2f} {mur92['S_norm']:>15.2f} {things['S_norm']/mur92['S_norm']:>10.1f}x")
    print(f"{'||S||_F (frobenius)':<30} {things['fro_norm']:>15.2f} {mur92['fro_norm']:>15.2f} {things['fro_norm']/mur92['fro_norm']:>10.1f}x")
    print(f"{'L_max (row concentration)':<30} {things['L_max_95']:>15.2f} {mur92['L_max_95']:>15.2f} {things['L_max_95']/mur92['L_max_95']:>10.1f}x")
    print(f"{'L_inf (2*max|S_ij|)':<30} {things['L_inf']:>15.4f} {mur92['L_inf']:>15.4f} {things['L_inf']/mur92['L_inf']:>10.1f}x")
    print(f"{'eff_dim':<30} {things['eff_dim']:>15.2f} {mur92['eff_dim']:>15.2f}")
    print()
    print(f"{'N (numerator)':<30} {things['N']:>15.4f} {mur92['N']:>15.4f} {things['N']/mur92['N']:>10.1f}x")
    print(f"{'D (denominator)':<30} {things['D']:>15.4f} {mur92['D']:>15.4f} {things['D']/mur92['D']:>10.1f}x")
    print(f"{'pmin = N/D':<30} {things['pmin']:>15.6f} {mur92['pmin']:>15.6f} {mur92['pmin']/things['pmin']:>10.1f}x")
    print()

    print("="*70)
    print("KEY INSIGHT: L_max / ||S||^2 ratio (row concentration vs spectral norm)")
    print("="*70)
    print(f"  THINGS: {things['L_ratio']:.6f}")
    print(f"  mur92:  {mur92['L_ratio']:.6f}")
    print(f"  Ratio:  {things['L_ratio']/mur92['L_ratio']:.1f}x higher for THINGS")
    print()
    print("THINGS has MUCH higher row concentration relative to its spectral norm.")
    print("This means energy is spread across rows rather than concentrated in")
    print("a low-rank structure -> harder to recover from sparse samples -> low pmin.")


if __name__ == "__main__":
    main()
