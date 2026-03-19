"""Test rank detection with simulation_dirichlet (same as imputation)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def make_similarity(n: int, k: int, alpha: float, seed: int) -> np.ndarray:
    """Generate similarity using simulation_dirichlet (same as imputation)."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=alpha)
    similarity = compute_similarity(w, w, "gaussian_kernel")
    return similarity


def evaluate_condition(
    n: int, true_rank: int, alpha: float, seed: int,
    grid_step: int = 5, grid_span: int = 10, cv_repeats: int = 5,
) -> dict:
    """Evaluate single condition."""
    similarity = make_similarity(n, true_rank, alpha, seed)

    lower = max(2, true_rank - grid_span)
    upper = true_rank + grid_span
    candidate_ranks = list(range(lower, upper + 1, grid_step))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    cv_results = cross_val_score(
        similarity,
        estimator=SRF(init="random_sqrt", random_state=seed, max_outer=30, max_inner=20),
        param_grid={"rank": candidate_ranks},
        n_repeats=cv_repeats,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        random_state=seed,
        verbose=0,
        n_jobs=1,
        fit_final_estimator=False,
    )

    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()

    best_rank = int(mean_scores.idxmin())
    scores_sorted = mean_scores.sort_values()
    score_gap = float(scores_sorted.iloc[1] - scores_sorted.iloc[0]) if len(scores_sorted) > 1 else 0

    return {
        "n": n,
        "true_rank": true_rank,
        "alpha": alpha,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - true_rank),
        "is_correct": best_rank == true_rank,
        "score_gap": score_gap,
    }


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    n = 200
    true_ranks = [5, 10, 15, 20, 25, 30]
    alpha = 1.0  # Same as imputation
    n_seeds = 10

    conditions = [
        (n, tr, alpha, seed)
        for tr in true_ranks
        for seed in range(n_seeds)
    ]

    log.info(f"Testing with simulation_dirichlet (alpha={alpha})")
    log.info(f"Running {len(conditions)} conditions...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_condition)(n, tr, alpha, seed)
        for n, tr, alpha, seed in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "dirichlet_results.csv", index=False)

    log.info("\n=== Results with simulation_dirichlet (alpha=1.0) ===")
    summary = df.groupby("true_rank").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
        score_gap=("score_gap", "median"),
    ).reset_index()

    for _, row in summary.iterrows():
        log.info(f"  k={row['true_rank']:2.0f}: Accuracy={row['accuracy']:.0%}, MAE={row['mae']:.1f}, gap={row['score_gap']:.2e}")

    overall = df["is_correct"].mean()
    log.info(f"\nOverall accuracy: {overall:.0%}")


if __name__ == "__main__":
    main()
