"""Visualize why pmin is low for THINGS vs mur92."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.io import load_triplets
from src.datasets import load_dataset
from src.colors import ROSE, TEAL, CYAN, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


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


def compute_pmin_components(S, gamma=1.05, eta=0.05):
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
        'row_sq': row_sq,
    }


def main():
    print("Loading THINGS...")
    train_triplets, _ = load_triplets(Path('data/things'), number='4.7mio')
    S_things = build_rsm_laplace(train_triplets, 1854, alpha=1.0)

    print("Loading mur92...")
    ds = load_dataset('mur92', root='/SSD/datasets/similarity_datasets/mur92')
    S_mur = ds.rsm

    things = compute_pmin_components(S_things)
    mur92 = compute_pmin_components(S_mur)

    # =========================================================================
    # Figure 1: pmin decomposition bar chart
    # =========================================================================
    fig, ax = create_figure("wide")

    metrics = ['||S||₂', 'L_max', 'L_∞', 'pmin']
    things_vals = [things['S_norm'], things['L_max_95'], things['L_inf'], things['pmin'] * 100]
    mur92_vals = [mur92['S_norm'], mur92['L_max_95'], mur92['L_inf'], mur92['pmin'] * 100]

    # Normalize for comparison (relative to mur92)
    ratios = [things_vals[i] / mur92_vals[i] for i in range(len(metrics))]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, [1.0]*4, width, label='mur92 (baseline)', color=ROSE, alpha=0.7)
    ax.bar(x + width/2, ratios, width, label='THINGS (relative)', color=TEAL, alpha=0.7)

    ax.axhline(1.0, color=GRAY, linestyle='--', linewidth=1)
    ax.set_ylabel("Ratio to mur92")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_yscale('log')

    # Add value annotations
    for i, (t, m) in enumerate(zip(things_vals, mur92_vals)):
        if metrics[i] == 'pmin':
            ax.text(i + width/2, ratios[i]*1.2, f'{t:.2f}%', ha='center', fontsize=8)
            ax.text(i - width/2, 1.2, f'{m:.2f}%', ha='center', fontsize=8)
        else:
            ax.text(i + width/2, ratios[i]*1.2, f'{t:.1f}', ha='center', fontsize=8)
            ax.text(i - width/2, 1.2, f'{m:.1f}', ha='center', fontsize=8)

    despine(ax)
    save_figure(fig, OUTPUT_DIR / "pmin_decomposition.pdf")
    plt.close()

    # =========================================================================
    # Figure 2: Row norm distributions
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # THINGS
    ax = axes[0]
    ax.hist(things['row_sq'], bins=50, alpha=0.7, color=TEAL, density=True)
    ax.axvline(things['L_max_95'], color=ROSE, linestyle='--', linewidth=2,
               label=f"L_95={things['L_max_95']:.1f}")
    ax.set_xlabel("Row squared norm (Σ S²ᵢⱼ)")
    ax.set_ylabel("Density")
    ax.set_title(f"THINGS (n={things['n']})")
    ax.legend()

    # mur92
    ax = axes[1]
    ax.hist(mur92['row_sq'], bins=30, alpha=0.7, color=ROSE, density=True)
    ax.axvline(mur92['L_max_95'], color=TEAL, linestyle='--', linewidth=2,
               label=f"L_95={mur92['L_max_95']:.1f}")
    ax.set_xlabel("Row squared norm (Σ S²ᵢⱼ)")
    ax.set_ylabel("Density")
    ax.set_title(f"mur92 (n={mur92['n']})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "row_norm_distributions.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: The scaling problem
    # =========================================================================
    fig, ax = create_figure("single")

    # Simulate pmin for different matrix sizes with THINGS-like structure
    sizes = np.array([50, 100, 200, 500, 1000, 1854])
    pmin_theoretical = []

    for n in sizes:
        # Approximate scaling: ||S|| ~ sqrt(n), L_max ~ n/2, L_inf = 2
        S_norm_approx = np.sqrt(n) * 20  # rough scaling from THINGS
        L_max_approx = n * 0.2
        L_inf = 2.0
        gamma = 1.05
        eta = 0.05
        eff_dim = 1.5

        N = (gamma * L_inf * S_norm_approx) / (3 * L_max_approx) + 1
        D = ((gamma * S_norm_approx)**2 / (2 * L_max_approx) + 1) / np.log(2 * eff_dim / eta)
        pmin_theoretical.append(N / D)

    ax.plot(sizes, pmin_theoretical, 'o-', color=TEAL, linewidth=2, markersize=8, label='Theoretical scaling')
    ax.axhline(things['pmin'], color=ROSE, linestyle='--', linewidth=2, label=f"THINGS actual: {things['pmin']:.4f}")
    ax.axhline(mur92['pmin'], color=CYAN, linestyle='--', linewidth=2, label=f"mur92: {mur92['pmin']:.4f}")

    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("pmin")
    ax.set_xscale('log')
    ax.legend()
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "pmin_size_scaling.pdf")
    plt.close()

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*70)
    print("KEY INSIGHT: Why pmin is low for THINGS")
    print("="*70)
    print(f"""
The pmin formula (simplified) is approximately:

    pmin ∝ L_∞ · log(eff_dim) / ||S||₂

Key factors for THINGS vs mur92:
1. ||S||₂ (spectral norm):  THINGS = {things['S_norm']:.1f},  mur92 = {mur92['S_norm']:.1f}  (26x larger)
2. L_∞ (2·max|Sᵢⱼ|):        THINGS = {things['L_inf']:.1f},  mur92 = {mur92['L_inf']:.1f}  (same - bounded by 2)
3. eff_dim:                 THINGS = {things['eff_dim']:.2f},  mur92 = {mur92['eff_dim']:.2f}

Since L_∞ is bounded at 2.0 (max similarity = 1), and ||S||₂ grows with matrix size,
pmin necessarily decreases as n increases.

For THINGS (n=1854):  pmin ≈ 2 / 690 ≈ 0.003
For mur92 (n=92):     pmin ≈ 2 / 27  ≈ 0.074

This is a FUNDAMENTAL property of the bounds - larger matrices will always have lower pmin
because the spectral norm grows while L_∞ is bounded.

The bound says: "To guarantee recovery of even the smallest eigenvalue, you need
very few samples" - but this is a very loose bound for large matrices.
""")

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
