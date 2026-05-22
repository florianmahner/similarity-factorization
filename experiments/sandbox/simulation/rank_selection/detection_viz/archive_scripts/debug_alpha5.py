"""Debug why α=5.0 overestimates rank even with more samples.

Investigate:
1. CV score curves - is the minimum clear?
2. Effect of n on the CV curve shape
3. Difference between α=1.0 and α=5.0

Usage:
    poetry run python sandbox/rank_detection_viz/debug_alpha5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir
from utils.simulation import simulation_dirichlet

OUTPUT_DIR = get_output_dir()


def get_cv_curve(n: int, k: int, alpha: float, seed: int = 0) -> dict:
    """Get full CV score curve for analysis."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    S = w @ w.T

    ranks = list(range(2, min(k + 15, n // 2), 2))

    result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=5,
        estimate_sampling_fraction=True,
        verbose=0,
        n_jobs=-1,
    )

    # Extract scores per rank
    cv_results = result.cv_results_
    mean_scores = cv_results["mean_test_score"]
    std_scores = cv_results["std_test_score"]
    param_ranks = [p["rank"] for p in cv_results["params"]]

    return {
        "ranks": param_ranks,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "best_rank": result.best_params_["rank"],
        "true_rank": k,
        "p_mean": result.best_estimator_.p_mean_ if hasattr(result.best_estimator_, "p_mean_") else None,
    }


def plot_cv_curves_comparison(output_dir: Path) -> None:
    """Compare CV curves for α=1.0 vs α=5.0 at different n."""
    k = 30
    ns = [200, 400, 600]
    alphas = [1.0, 5.0]

    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey="row")

    for i, alpha in enumerate(alphas):
        for j, n in enumerate(ns):
            ax = axes[i, j]

            print(f"Computing CV curve: n={n}, k={k}, alpha={alpha}...")
            result = get_cv_curve(n, k, alpha, seed=42)

            ranks = result["ranks"]
            scores = result["mean_scores"]
            stds = result["std_scores"]

            ax.plot(ranks, scores, "o-", color=TEAL, markersize=4, linewidth=1.5)
            ax.fill_between(ranks, scores - stds, scores + stds,
                           color=TEAL, alpha=0.2, linewidth=0)

            # Mark true rank
            ax.axvline(k, color=ROSE, linestyle="--", linewidth=1.5, label=f"true k={k}")

            # Mark selected rank
            ax.axvline(result["best_rank"], color=CYAN, linestyle=":", linewidth=1.5,
                      label=f"selected={result['best_rank']}")

            ax.set_xlabel("Rank")
            if j == 0:
                ax.set_ylabel(f"α={alpha}\nCV score")

            obs_dof = (n * (n-1) / 2) / (n * k)
            ax.set_title(f"n={n} (obs/dof={obs_dof:.1f})")

            if i == 0 and j == 2:
                ax.legend(frameon=False, fontsize=7, loc="upper right")

            despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "cv_curves_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  cv_curves_comparison.pdf")


def plot_score_gap_analysis(output_dir: Path) -> None:
    """Analyze how distinguishable the true rank is from neighbors."""
    k = 30
    ns = [200, 300, 400, 600]
    alphas = [1.0, 5.0]

    results = []

    for n in ns:
        for alpha in alphas:
            print(f"Analyzing score gap: n={n}, alpha={alpha}...")
            for seed in range(3):
                cv = get_cv_curve(n, k, alpha, seed=seed)

                ranks = np.array(cv["ranks"])
                scores = np.array(cv["mean_scores"])

                # Find score at true rank and neighbors
                true_idx = np.where(ranks == k)[0]
                if len(true_idx) == 0:
                    continue
                true_idx = true_idx[0]

                score_at_true = scores[true_idx]
                min_score = scores.min()
                min_idx = scores.argmin()

                # Gap between true rank score and minimum
                gap = score_at_true - min_score

                results.append({
                    "n": n,
                    "alpha": alpha,
                    "seed": seed,
                    "selected_rank": ranks[min_idx],
                    "error": ranks[min_idx] - k,
                    "score_gap": gap,
                    "score_at_true": score_at_true,
                    "min_score": min_score,
                })

    df = pd.DataFrame(results)

    # Plot
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Left: Score gap by n and alpha
    ax = axes[0]
    for alpha in alphas:
        subset = df[df["alpha"] == alpha].groupby("n")["score_gap"].mean()
        color = TEAL if alpha == 1.0 else ROSE
        ax.plot(subset.index, subset.values, "o-", color=color,
                markersize=6, linewidth=1.5, label=f"α={alpha}")

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("Score gap (true rank − best)")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Right: Selected rank error by n and alpha
    ax = axes[1]
    for alpha in alphas:
        subset = df[df["alpha"] == alpha].groupby("n")["error"].mean()
        color = TEAL if alpha == 1.0 else ROSE
        ax.plot(subset.index, subset.values, "o-", color=color,
                markersize=6, linewidth=1.5, label=f"α={alpha}")

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("Rank error (selected − true)")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "score_gap_analysis.pdf")
    print("  score_gap_analysis.pdf")

    # Print summary
    print("\nSummary:")
    print(df.groupby(["n", "alpha"])[["error", "score_gap"]].mean().round(2))


def analyze_matrix_properties(output_dir: Path) -> None:
    """Analyze why α=5.0 behaves differently - look at matrix properties."""
    k = 30
    n = 400
    seed = 42

    print("\nMatrix properties analysis:")
    print("=" * 50)

    for alpha in [1.0, 5.0]:
        rng = np.random.default_rng(seed)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
        S = w @ w.T

        # Eigenvalue analysis
        eigvals = np.linalg.eigvalsh(S)
        eigvals = np.sort(eigvals)[::-1]

        # Effective rank (using entropy-based measure)
        eigvals_norm = eigvals / eigvals.sum()
        eigvals_norm = eigvals_norm[eigvals_norm > 1e-10]
        effective_rank = np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm)))

        # How many eigenvalues explain 99% variance?
        cumsum = np.cumsum(eigvals) / eigvals.sum()
        rank_99 = np.searchsorted(cumsum, 0.99) + 1

        print(f"\nα={alpha}:")
        print(f"  Top 5 eigenvalues: {eigvals[:5].round(2)}")
        print(f"  Effective rank (entropy): {effective_rank:.1f}")
        print(f"  Rank for 99% variance: {rank_99}")
        print(f"  Eigenvalue ratio (1st/30th): {eigvals[0]/eigvals[29]:.1f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Debugging α=5.0 rank overestimation...")
    print("=" * 50)

    # Quick matrix analysis first (no CV needed)
    analyze_matrix_properties(OUTPUT_DIR)

    print("\nGenerating CV curve comparison (this takes a few minutes)...")
    plot_cv_curves_comparison(OUTPUT_DIR)

    print("\nAnalyzing score gaps...")
    plot_score_gap_analysis(OUTPUT_DIR)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
