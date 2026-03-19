"""Debug rank detection with simulation_dirichlet.

Systematically test what makes CV work for higher ranks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def make_similarity(n: int, k: int, alpha: float, seed: int) -> np.ndarray:
    """Generate similarity using simulation_dirichlet."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=alpha)
    return compute_similarity(w, w, "gaussian_kernel")


def evaluate_with_details(
    n: int, true_rank: int, alpha: float, seed: int,
    cv_repeats: int = 5, sampling_fraction: float | None = None,
) -> dict:
    """Evaluate with detailed CV curve output."""
    similarity = make_similarity(n, true_rank, alpha, seed)

    # Wider grid for debugging
    lower = max(2, true_rank - 15)
    upper = true_rank + 15
    candidate_ranks = list(range(lower, upper + 1, 3))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    cv_kwargs = dict(
        estimator=SRF(init="random_sqrt", random_state=seed, max_outer=50, max_inner=30),
        param_grid={"rank": candidate_ranks},
        n_repeats=cv_repeats,
        random_state=seed,
        verbose=0,
        n_jobs=1,
        fit_final_estimator=False,
    )

    if sampling_fraction is not None:
        cv_kwargs["sampling_fraction"] = sampling_fraction
        cv_kwargs["estimate_sampling_fraction"] = False
    else:
        cv_kwargs["estimate_sampling_fraction"] = True
        cv_kwargs["sampling_selection"] = "mean"

    cv_results = cross_val_score(similarity, **cv_kwargs)

    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()
    std_scores = cv_df.groupby("rank")["score"].std()

    best_rank = int(mean_scores.idxmin())

    return {
        "n": n,
        "true_rank": true_rank,
        "alpha": alpha,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - true_rank),
        "is_correct": best_rank == true_rank,
        "ranks": list(mean_scores.index),
        "scores": list(mean_scores.values),
        "stds": list(std_scores.values),
    }


def plot_cv_curves(results: list[dict], output_dir: Path, title: str, filename: str) -> None:
    """Plot CV curves for multiple conditions."""
    n_plots = len(results)
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows), squeeze=False)
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        ranks = np.array(result["ranks"])
        scores = np.array(result["scores"])
        stds = np.array(result["stds"])

        ax.errorbar(ranks, scores, yerr=stds, marker="o", markersize=4, capsize=2,
                   color="steelblue", alpha=0.8)
        ax.axvline(result["true_rank"], color="red", linestyle="--", lw=1.5, label="True")
        ax.axvline(result["selected_rank"], color="green", linestyle=":", lw=1.5, label="Selected")

        ax.set_title(f"k={result['true_rank']}, α={result['alpha']}", fontsize=9)
        ax.set_xlabel("Rank", fontsize=8)
        ax.set_ylabel("CV Score", fontsize=8)
        sns.despine(ax=ax)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {filename}")


def test_alpha_and_n():
    """Test different alpha and n combinations."""
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Test parameters
    true_ranks = [10, 20, 30]
    alphas = [0.1, 0.3, 0.5, 1.0]
    n_values = [200, 300, 400]
    n_seeds = 5

    all_results = []

    # First: vary alpha with fixed n=300
    log.info("=== Testing alpha values (n=300) ===")
    n = 300
    for alpha in alphas:
        conditions = [(n, tr, alpha, s) for tr in true_ranks for s in range(n_seeds)]
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_with_details)(n, tr, a, s) for n, tr, a, s in conditions
        )

        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ranks", "scores", "stds"]} for r in results])

        log.info(f"\nalpha={alpha}:")
        for tr in true_ranks:
            sub = df[df["true_rank"] == tr]
            log.info(f"  k={tr}: Acc={sub['is_correct'].mean():.0%}, MAE={sub['abs_error'].mean():.1f}")

        all_results.extend(results)

        # Plot CV curves for seed=0
        seed0_results = [r for r in results if r["seed"] == 0]
        plot_cv_curves(seed0_results, output_dir, f"CV Curves (n={n}, α={alpha})", f"cv_alpha_{alpha}.pdf")

    # Second: vary n with fixed alpha=0.1
    log.info("\n=== Testing n values (alpha=0.1) ===")
    alpha = 0.1
    for n in n_values:
        conditions = [(n, tr, alpha, s) for tr in true_ranks for s in range(n_seeds)]
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_with_details)(n, tr, a, s) for n, tr, a, s in conditions
        )

        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ranks", "scores", "stds"]} for r in results])

        log.info(f"\nn={n}:")
        for tr in true_ranks:
            sub = df[df["true_rank"] == tr]
            log.info(f"  k={tr}: Acc={sub['is_correct'].mean():.0%}, MAE={sub['abs_error'].mean():.1f}")

    # Third: test with fixed sampling fraction (bypass estimation)
    log.info("\n=== Testing fixed sampling_fraction=0.8 (n=300, alpha=0.1) ===")
    n, alpha = 300, 0.1
    conditions = [(n, tr, alpha, s) for tr in true_ranks for s in range(n_seeds)]
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(evaluate_with_details)(n, tr, a, s, sampling_fraction=0.8) for n, tr, a, s in conditions
    )

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ranks", "scores", "stds"]} for r in results])
    for tr in true_ranks:
        sub = df[df["true_rank"] == tr]
        log.info(f"  k={tr}: Acc={sub['is_correct'].mean():.0%}, MAE={sub['abs_error'].mean():.1f}")

    seed0_results = [r for r in results if r["seed"] == 0]
    plot_cv_curves(seed0_results, output_dir, "CV Curves (fixed sampling=0.8)", "cv_fixed_sampling.pdf")


def main():
    test_alpha_and_n()


if __name__ == "__main__":
    main()
