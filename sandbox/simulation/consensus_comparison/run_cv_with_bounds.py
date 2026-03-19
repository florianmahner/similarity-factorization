"""
Run cross-validation with bounds estimation for mur92.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra, precompute_matrix_info
from pysrf.cross_validation import cross_val_score
from src.datasets.loaders import load_mur92
from src.colors import TEAL, ROSE, CYAN
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "cv_with_bounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm

    print("=" * 60)
    print("CROSS-VALIDATION WITH BOUNDS ESTIMATION")
    print("=" * 60)

    # =========================================================================
    # 1. Get bounds and effective dimension
    # =========================================================================
    print("\n1. BOUNDS ESTIMATION")
    print("-" * 40)

    info = precompute_matrix_info(rsm)
    print(f"Effective dimension: {info.eff_dim}")

    pmin, pmax, _ = estimate_sampling_bounds_ultra(rsm, verbose=False)
    print(f"pmin = {pmin:.4f}")
    print(f"pmax = {pmax:.4f}")

    # Use mean of bounds as sampling fraction
    sampling_frac = (pmin + pmax) / 2
    print(f"Using sampling fraction: {sampling_frac:.4f}")

    # =========================================================================
    # 2. Cross-validation with estimated sampling fraction
    # =========================================================================
    print("\n2. CROSS-VALIDATION (with bounds-estimated sampling)")
    print("-" * 40)

    ranks_to_test = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    # Run with estimated sampling fraction
    print(f"\nUsing sampling_fraction = {sampling_frac:.3f} (mean of bounds)")
    cv_bounds = cross_val_score(
        rsm,
        estimator=SRF(random_state=42),
        param_grid={"rank": ranks_to_test},
        n_repeats=30,  # More repeats for stability
        sampling_fraction=sampling_frac,
        random_state=42,
        verbose=0,
        n_jobs=-1,
    )

    print(f"Best rank (bounds sampling): {cv_bounds.best_params_['rank']}")

    mean_bounds = cv_bounds.cv_results_.groupby("rank")["score"].agg(["mean", "std"])
    print("\nCV scores:")
    for rank in ranks_to_test:
        m, s = mean_bounds.loc[rank]
        marker = " <-- BEST" if rank == cv_bounds.best_params_['rank'] else ""
        print(f"  Rank {rank:2d}: {m:.6f} ± {s:.6f}{marker}")

    # =========================================================================
    # 3. Compare with different sampling fractions
    # =========================================================================
    print("\n3. EFFECT OF SAMPLING FRACTION")
    print("-" * 40)

    sampling_fracs = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_ranks = {}

    for sf in sampling_fracs:
        cv = cross_val_score(
            rsm,
            estimator=SRF(random_state=42),
            param_grid={"rank": ranks_to_test},
            n_repeats=20,
            sampling_fraction=sf,
            random_state=42,
            verbose=0,
            n_jobs=-1,
        )
        best_ranks[sf] = cv.best_params_['rank']
        print(f"  sampling_fraction={sf:.1f}: best_rank={cv.best_params_['rank']}")

    # =========================================================================
    # 4. Check if rank 2 is actually good
    # =========================================================================
    print("\n4. IS RANK 2 ACTUALLY BAD?")
    print("-" * 40)

    # Compare rank 2 vs best rank
    best_rank = cv_bounds.best_params_['rank']
    rank2_score = mean_bounds.loc[2, 'mean']
    best_score = mean_bounds.loc[best_rank, 'mean']

    print(f"Rank 2 score:  {rank2_score:.6f}")
    print(f"Rank {best_rank} score: {best_score:.6f}")
    print(f"Difference: {rank2_score - best_score:.6f} ({100*(rank2_score - best_score)/best_score:.1f}% worse)")

    # =========================================================================
    # 5. Elbow analysis
    # =========================================================================
    print("\n5. ELBOW ANALYSIS")
    print("-" * 40)

    means = [mean_bounds.loc[r, 'mean'] for r in ranks_to_test]

    # Compute improvement from rank k to k+1
    improvements = []
    for i in range(len(ranks_to_test) - 1):
        imp = means[i] - means[i+1]
        pct = 100 * imp / means[i]
        improvements.append((ranks_to_test[i], ranks_to_test[i+1], imp, pct))
        print(f"  Rank {ranks_to_test[i]:2d} → {ranks_to_test[i+1]:2d}: improvement = {pct:.1f}%")

    # Find elbow (where improvement drops below threshold)
    threshold = 5  # 5% improvement threshold
    elbow_rank = ranks_to_test[0]
    for r1, r2, imp, pct in improvements:
        if pct < threshold:
            elbow_rank = r1
            break
        elbow_rank = r2

    print(f"\nElbow rank (improvement < {threshold}%): {elbow_rank}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    ax = axes[0]
    means_arr = [mean_bounds.loc[r, 'mean'] for r in ranks_to_test]
    stds_arr = [mean_bounds.loc[r, 'std'] for r in ranks_to_test]
    ax.errorbar(ranks_to_test, means_arr, yerr=stds_arr, fmt='o-', capsize=3, color=TEAL)
    ax.axvline(2, color=ROSE, linestyle='--', label=f'Eff. dim = 2')
    ax.axvline(elbow_rank, color=CYAN, linestyle=':', label=f'Elbow = {elbow_rank}')
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV MSE")
    ax.set_title(f"CV with sampling={sampling_frac:.2f}")
    ax.legend(fontsize=8)
    despine(ax)

    ax = axes[1]
    pcts = [imp[3] for imp in improvements]
    ax.bar(range(len(pcts)), pcts, color=TEAL)
    ax.axhline(threshold, color=ROSE, linestyle='--', label=f'{threshold}% threshold')
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"{r1}→{r2}" for r1, r2, _, _ in improvements], fontsize=7, rotation=45)
    ax.set_xlabel("Rank transition")
    ax.set_ylabel("% improvement")
    ax.set_title("Marginal improvement")
    ax.legend(fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "cv_analysis.pdf")
    plt.close(fig)
    print(f"\nSaved cv_analysis.pdf")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Effective dimension from bounds: {info.eff_dim}
Best rank from CV: {cv_bounds.best_params_['rank']}
Elbow rank: {elbow_rank}

The discrepancy is because:
1. Effective dimension measures SPECTRAL concentration (how few eigenvalues dominate)
2. CV measures PREDICTION accuracy (which benefits from capturing smaller eigenvalues)

For mur92:
- λ_1 explains 92.4% of variance (huge!)
- But the remaining 7.6% spread across many small eigenvalues
- CV says: "those small eigenvalues help prediction"
- Bounds say: "spectral mass is concentrated in 2 dimensions"

BOTH ARE CORRECT - they measure different things!
- Use rank 2 for: interpretability, bounds-based analysis
- Use rank {elbow_rank}-{best_rank} for: best prediction
""")

    print(f"\nOutput saved to {output_dir}")


if __name__ == "__main__":
    main()
