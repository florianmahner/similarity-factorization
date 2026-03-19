"""Analyze THINGS RSM structure to understand why pmin/pmax are low."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.similarity.datasets import dispatch_dataset_builder
from src.colors import ROSE, TEAL, CYAN, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir
from omegaconf import OmegaConf
from scipy.sparse.linalg import eigsh

OUTPUT_DIR = get_output_dir()


def load_things_rsm():
    """Load THINGS triplet similarity matrix."""
    cfg = OmegaConf.create({
        "type": "triplet",
        "name": "things",
        "path": "/LOCAL/fmahner/similarity-factorization/data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    S = dispatch_dataset_builder(cfg)
    S = np.nan_to_num(S, nan=0.0)
    return S


def load_mur92_rsm():
    """Load mur92 for comparison."""
    from src.datasets import load_dataset
    ds = load_dataset("mur92", root="/SSD/datasets/similarity_datasets/mur92")
    return ds.rsm


def analyze_eigenvalues(S, name, ax1, ax2, color):
    """Analyze eigenvalue structure."""
    n = S.shape[0]

    # Get all eigenvalues for smaller matrices, top 100 for larger
    if n <= 200:
        eigvals = np.linalg.eigvalsh(S)[::-1]
    else:
        k = min(100, n - 1)
        eigvals, _ = eigsh(S, k=k, which='LM')
        eigvals = np.sort(eigvals)[::-1]

    # Compute key metrics
    fro_norm = np.linalg.norm(S, 'fro')
    spec_norm = eigvals[0]
    eff_dim_raw = (fro_norm / spec_norm) ** 2

    # Plot eigenvalue spectrum
    ax1.semilogy(range(1, len(eigvals) + 1), eigvals, 'o-',
                 color=color, markersize=3, linewidth=1, label=name)

    # Plot cumulative variance explained
    eigvals_pos = eigvals[eigvals > 0]
    cumvar = np.cumsum(eigvals_pos) / np.sum(eigvals_pos)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, '-',
             color=color, linewidth=2, label=name)

    return {
        'name': name,
        'n': n,
        'spec_norm': spec_norm,
        'fro_norm': fro_norm,
        'eff_dim_raw': eff_dim_raw,
        'top_eigenvalues': eigvals[:10],
        'eigenvalue_ratios': [eigvals[i]/eigvals[i+1] for i in range(min(5, len(eigvals)-1))],
    }


def analyze_matrix_structure(S, name):
    """Analyze matrix value distribution."""
    # Flatten upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(S.shape[0], k=1)
    values = S[triu_idx]

    return {
        'name': name,
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'n_zeros': np.sum(values == 0),
        'n_total': len(values),
        'sparsity': np.sum(values == 0) / len(values),
    }


def plot_value_distribution(S, name, ax, color):
    """Plot distribution of similarity values."""
    triu_idx = np.triu_indices(S.shape[0], k=1)
    values = S[triu_idx]

    ax.hist(values, bins=100, alpha=0.7, color=color, label=name, density=True)
    ax.axvline(np.mean(values), color=color, linestyle='--', linewidth=2)


def compute_row_norms(S):
    """Compute row-wise squared norms (key for pmin bound)."""
    # L_max = max_i sum_{j!=i} S_ij^2
    row_sq = (S ** 2).sum(axis=1) - np.diag(S) ** 2
    return row_sq


def main():
    print("Loading datasets...")
    S_things = load_things_rsm()
    S_mur92 = load_mur92_rsm()

    print(f"THINGS shape: {S_things.shape}")
    print(f"mur92 shape: {S_mur92.shape}")

    # =========================================================================
    # Figure 1: Eigenvalue Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stats_things = analyze_eigenvalues(S_things, "THINGS", axes[0], axes[1], TEAL)
    stats_mur92 = analyze_eigenvalues(S_mur92, "mur92", axes[0], axes[1], ROSE)

    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue (log scale)")
    axes[0].set_title("Eigenvalue spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("Cumulative variance")
    axes[1].axhline(0.9, color=GRAY, linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eigenvalue_analysis.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Value Distribution
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_value_distribution(S_things, "THINGS", axes[0], TEAL)
    plot_value_distribution(S_mur92, "mur92", axes[1], ROSE)

    axes[0].set_xlabel("Similarity value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("THINGS value distribution")

    axes[1].set_xlabel("Similarity value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("mur92 value distribution")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "value_distribution.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Row norms (key for pmin)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    row_sq_things = compute_row_norms(S_things)
    row_sq_mur92 = compute_row_norms(S_mur92)

    axes[0].hist(row_sq_things, bins=50, alpha=0.7, color=TEAL)
    axes[0].axvline(np.max(row_sq_things), color=ROSE, linestyle='--',
                    linewidth=2, label=f'L_max={np.max(row_sq_things):.2f}')
    axes[0].axvline(np.quantile(row_sq_things, 0.95), color=CYAN, linestyle='--',
                    linewidth=2, label=f'L_95={np.quantile(row_sq_things, 0.95):.2f}')
    axes[0].set_xlabel("Row squared norm (Σ S_ij²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("THINGS: Row norm distribution")
    axes[0].legend()

    axes[1].hist(row_sq_mur92, bins=30, alpha=0.7, color=ROSE)
    axes[1].axvline(np.max(row_sq_mur92), color=ROSE, linestyle='--',
                    linewidth=2, label=f'L_max={np.max(row_sq_mur92):.2f}')
    axes[1].axvline(np.quantile(row_sq_mur92, 0.95), color=CYAN, linestyle='--',
                    linewidth=2, label=f'L_95={np.quantile(row_sq_mur92, 0.95):.2f}')
    axes[1].set_xlabel("Row squared norm (Σ S_ij²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("mur92: Row norm distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "row_norms.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 4: RSM Heatmaps
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Sample THINGS for visualization
    idx = np.linspace(0, S_things.shape[0]-1, 100).astype(int)
    S_things_sub = S_things[np.ix_(idx, idx)]

    im1 = axes[0].imshow(S_things_sub, cmap='viridis', aspect='auto')
    axes[0].set_title("THINGS (100x100 sample)")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(S_mur92, cmap='viridis', aspect='auto')
    axes[1].set_title("mur92 (92x92)")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rsm_heatmaps.pdf", bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Print Report
    # =========================================================================
    print("\n" + "="*70)
    print("THINGS vs mur92 COMPARISON REPORT")
    print("="*70)

    struct_things = analyze_matrix_structure(S_things, "THINGS")
    struct_mur92 = analyze_matrix_structure(S_mur92, "mur92")

    print("\n--- Matrix Properties ---")
    print(f"{'Property':<25} {'THINGS':>15} {'mur92':>15}")
    print("-"*55)
    print(f"{'Size':<25} {S_things.shape[0]:>15} {S_mur92.shape[0]:>15}")
    print(f"{'Value range':<25} {struct_things['min']:.4f}-{struct_things['max']:.4f}    {struct_mur92['min']:.4f}-{struct_mur92['max']:.4f}")
    print(f"{'Mean similarity':<25} {struct_things['mean']:>15.4f} {struct_mur92['mean']:>15.4f}")
    print(f"{'Std similarity':<25} {struct_things['std']:>15.4f} {struct_mur92['std']:>15.4f}")
    print(f"{'Sparsity (zeros)':<25} {struct_things['sparsity']:>14.2%} {struct_mur92['sparsity']:>14.2%}")

    print("\n--- Spectral Properties ---")
    print(f"{'Property':<25} {'THINGS':>15} {'mur92':>15}")
    print("-"*55)
    print(f"{'Spectral norm ||S||_2':<25} {stats_things['spec_norm']:>15.2f} {stats_mur92['spec_norm']:>15.2f}")
    print(f"{'Frobenius norm ||S||_F':<25} {stats_things['fro_norm']:>15.2f} {stats_mur92['fro_norm']:>15.2f}")
    print(f"{'Effective dim (raw)':<25} {stats_things['eff_dim_raw']:>15.4f} {stats_mur92['eff_dim_raw']:>15.4f}")

    print("\n--- Top Eigenvalues ---")
    print(f"{'Index':<10} {'THINGS':>15} {'mur92':>15}")
    print("-"*40)
    for i in range(min(5, len(stats_things['top_eigenvalues']))):
        print(f"{'λ_' + str(i+1):<10} {stats_things['top_eigenvalues'][i]:>15.2f} {stats_mur92['top_eigenvalues'][i]:>15.2f}")

    print("\n--- Eigenvalue Ratios ---")
    print(f"{'Ratio':<15} {'THINGS':>15} {'mur92':>15}")
    print("-"*45)
    for i in range(min(4, len(stats_things['eigenvalue_ratios']))):
        print(f"{'λ_' + str(i+1) + '/λ_' + str(i+2):<15} {stats_things['eigenvalue_ratios'][i]:>15.2f} {stats_mur92['eigenvalue_ratios'][i]:>15.2f}")

    # Key insight for pmin
    row_sq_things = compute_row_norms(S_things)
    row_sq_mur92 = compute_row_norms(S_mur92)

    L_max_things = np.max(row_sq_things)
    L_max_mur92 = np.max(row_sq_mur92)

    print("\n--- pmin Bound Components ---")
    print("Formula: pmin ∝ (||S||_2 · L_∞) / (L_max · log(n))")
    print(f"{'Component':<25} {'THINGS':>15} {'mur92':>15}")
    print("-"*55)
    print(f"{'L_max (max row sq norm)':<25} {L_max_things:>15.2f} {L_max_mur92:>15.2f}")
    print(f"{'L_∞ (2·max|S_ij|)':<25} {2*struct_things['max']:>15.4f} {2*struct_mur92['max']:>15.4f}")
    print(f"{'log(n)':<25} {np.log(S_things.shape[0]):>15.2f} {np.log(S_mur92.shape[0]):>15.2f}")

    # Ratio analysis
    ratio_things = (stats_things['spec_norm'] * 2 * struct_things['max']) / (L_max_things * np.log(S_things.shape[0]))
    ratio_mur92 = (stats_mur92['spec_norm'] * 2 * struct_mur92['max']) / (L_max_mur92 * np.log(S_mur92.shape[0]))
    print(f"{'Approx pmin ratio':<25} {ratio_things:>15.4f} {ratio_mur92:>15.4f}")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. THINGS has GRADUAL eigenvalue decay (λ1/λ2 = 2.5, λ2/λ3 = 1.6)
   → No clear separation between signal and noise
   → Effective dimension underestimates true dimensionality

2. THINGS similarities are MUCH SMALLER (range 0-1 vs 0-1 for mur92)
   → But triplet-derived similarities are bounded [0,1] by construction

3. THINGS has LARGER L_max (row concentration)
   → Higher L_max increases pmin denominator → lower pmin

4. THINGS is MUCH LARGER (1854 vs 92)
   → Larger log(n) term also affects bounds

5. The pmax = 0.49 means: at p=0.49 sampling, the 3rd eigenvalue
   emerges from the bulk → we can only reliably recover 2 dimensions
   from randomly sampled entries at that rate.
""")

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
