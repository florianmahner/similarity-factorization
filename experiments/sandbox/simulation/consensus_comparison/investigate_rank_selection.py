"""
Investigate rank selection for mur92.

Check why bounds estimation might not give rank 2 even though
purity analysis suggests 2 clean categories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvalsh

from pysrf import SRF
from pysrf.bounds import (
    compute_effective_dimension,
    estimate_sampling_bounds_ultra,
    precompute_matrix_info,
)
from pysrf.cross_validation import cross_val_score, EntryMaskSplit, GridSearchCV
from src.datasets.loaders import load_mur92
from src.colors import TEAL, CYAN, ROSE, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "rank_selection_investigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    n = rsm.shape[0]

    print("=" * 60)
    print("MUR92 RANK SELECTION INVESTIGATION")
    print("=" * 60)

    # =========================================================================
    # 1. Eigenvalue analysis
    # =========================================================================
    print("\n1. EIGENVALUE ANALYSIS")
    print("-" * 40)

    eigvals = np.sort(eigvalsh(rsm))[::-1]
    print(f"Top 10 eigenvalues: {eigvals[:10].round(3)}")
    print("Eigenvalue ratios (λ_k / λ_{k+1}):")
    for k in range(1, 8):
        ratio = eigvals[k-1] / eigvals[k] if eigvals[k] > 0 else np.inf
        print(f"  λ_{k} / λ_{k+1} = {eigvals[k-1]:.3f} / {eigvals[k]:.3f} = {ratio:.2f}")

    # =========================================================================
    # 2. Effective dimension from bounds
    # =========================================================================
    print("\n2. EFFECTIVE DIMENSION (Frobenius / Spectral norm)")
    print("-" * 40)

    info = precompute_matrix_info(rsm)
    print(f"Frobenius norm: {info.fro_norm:.4f}")
    print(f"Spectral norm: {info.s_norm:.4f}")
    print(f"Effective dimension (raw): {(info.fro_norm / info.s_norm) ** 2:.4f}")
    print(f"Effective dimension (rounded): {info.eff_dim}")

    # =========================================================================
    # 3. Sampling bounds estimation
    # =========================================================================
    print("\n3. SAMPLING BOUNDS ESTIMATION")
    print("-" * 40)

    pmin, pmax, _ = estimate_sampling_bounds_ultra(rsm, verbose=True)
    print(f"pmin = {pmin:.4f}")
    print(f"pmax = {pmax:.4f}")
    print(f"Suggested sampling range: [{pmin:.2f}, {pmax:.2f}]")

    # =========================================================================
    # 4. Cross-validation with many repeats
    # =========================================================================
    print("\n4. CROSS-VALIDATION (20 repeats)")
    print("-" * 40)

    ranks_to_test = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    cv_results = cross_val_score(
        rsm,
        estimator=SRF(random_state=42),
        param_grid={"rank": ranks_to_test},
        n_repeats=20,
        sampling_fraction=0.8,
        random_state=42,
        verbose=0,
        n_jobs=-1,
    )

    print(f"Best rank: {cv_results.best_params_}")
    print(f"Best score (MSE): {cv_results.best_score_:.6f}")

    # Get mean scores per rank
    mean_scores = cv_results.cv_results_.groupby("rank")["score"].agg(["mean", "std"])
    print("\nCV scores by rank:")
    for rank in ranks_to_test:
        if rank in mean_scores.index:
            m, s = mean_scores.loc[rank]
            print(f"  Rank {rank:2d}: {m:.6f} ± {s:.6f}")

    # =========================================================================
    # 5. Reconstruction error vs rank (full data)
    # =========================================================================
    print("\n5. RECONSTRUCTION ERROR (full data)")
    print("-" * 40)

    recon_errors = []
    for rank in ranks_to_test:
        model = SRF(rank=rank, random_state=42)
        model.fit(rsm)
        recon = model.reconstruct()
        error = np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")
        recon_errors.append(error)
        print(f"  Rank {rank:2d}: {error:.4f}")

    # =========================================================================
    # 6. Variance explained by rank
    # =========================================================================
    print("\n6. VARIANCE EXPLAINED BY EIGENVALUES")
    print("-" * 40)

    total_var = np.sum(eigvals ** 2)
    cumvar = np.cumsum(eigvals ** 2) / total_var

    print("Cumulative variance explained:")
    for k in [1, 2, 3, 4, 5, 6, 10, 15, 20]:
        if k <= len(cumvar):
            print(f"  Top {k:2d} eigenvalues: {100 * cumvar[k-1]:.1f}%")

    # =========================================================================
    # Plot 1: Eigenvalue spectrum
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    ax = axes[0]
    ax.plot(range(1, n+1), eigvals, "o-", markersize=4, color=TEAL)
    ax.axhline(0, color=GRAY_LIGHT, linestyle="--")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue spectrum")
    ax.set_xlim(0, 20)
    despine(ax)

    ax = axes[1]
    ax.plot(range(1, min(21, n+1)), cumvar[:20], "o-", markersize=4, color=CYAN)
    ax.axhline(0.9, color=GRAY_LIGHT, linestyle="--", label="90%")
    ax.axhline(0.95, color=GRAY, linestyle="--", label="95%")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Variance explained")
    ax.legend(fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "plot1_eigenvalues.pdf")
    plt.close(fig)
    print("\nSaved plot1_eigenvalues.pdf")

    # =========================================================================
    # Plot 2: CV scores vs rank
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    ax = axes[0]
    means = [mean_scores.loc[r, "mean"] for r in ranks_to_test]
    stds = [mean_scores.loc[r, "std"] for r in ranks_to_test]
    ax.errorbar(ranks_to_test, means, yerr=stds, fmt="o-", capsize=3, color=TEAL)
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV MSE (lower is better)")
    ax.set_title("Cross-validation scores")
    despine(ax)

    ax = axes[1]
    ax.plot(ranks_to_test, recon_errors, "o-", color=ROSE, label="Reconstruction")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Reconstruction error (full data)")
    despine(ax)

    save_figure(fig, output_dir / "plot2_cv_scores.pdf")
    plt.close(fig)
    print("Saved plot2_cv_scores.pdf")

    # =========================================================================
    # Plot 3: Eigenvalue gaps
    # =========================================================================
    fig, ax = create_figure("single")

    gaps = eigvals[:-1] - eigvals[1:]
    ax.bar(range(1, min(16, len(gaps)+1)), gaps[:15], color=TEAL)
    ax.set_xlabel("Gap index (λ_k - λ_{k+1})")
    ax.set_ylabel("Gap size")
    ax.set_title("Eigenvalue gaps")
    despine(ax)

    save_figure(fig, output_dir / "plot3_eigenvalue_gaps.pdf")
    plt.close(fig)
    print("Saved plot3_eigenvalue_gaps.pdf")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"""
Effective dimension from bounds: {info.eff_dim}
Best rank from CV: {cv_results.best_params_['rank']}

Key observations:
1. Effective dimension formula: (||S||_F / ||S||_2)^2 = {(info.fro_norm / info.s_norm) ** 2:.2f}
   This rounds to {info.eff_dim}, which is used for pmax estimation.

2. The eigenvalue spectrum shows:
   - λ_1 = {eigvals[0]:.2f} (dominant)
   - λ_2 = {eigvals[1]:.2f}
   - λ_3 = {eigvals[2]:.2f}
   - Gap λ_1-λ_2 = {eigvals[0] - eigvals[1]:.2f}
   - Gap λ_2-λ_3 = {eigvals[1] - eigvals[2]:.2f}

3. Variance explained:
   - Top 2: {100 * cumvar[1]:.1f}%
   - Top 3: {100 * cumvar[2]:.1f}%
   - Top 5: {100 * cumvar[4]:.1f}%

4. The effective dimension formula captures spectral properties,
   not semantic "cleanliness" of categories.

The purity analysis (rank 2 being cleanest) reflects interpretability,
but the bounds/CV analysis reflects reconstruction accuracy.
""")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
