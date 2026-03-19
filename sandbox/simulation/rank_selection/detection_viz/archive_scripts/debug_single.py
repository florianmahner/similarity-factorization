"""Debug single rank detection to verify it works.

Usage:
    poetry run python sandbox/rank_detection_viz/debug_single.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF, estimate_sampling_bounds_fast
from pysrf.cross_validation import EntryMaskSplit, fit_and_score
from joblib import Parallel, delayed

from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet


def run_single_condition(
    n: int,
    true_rank: int,
    alpha: float,
    snr: float,
    seed: int,
    grid_span: int = 8,
    cv_repeats: int = 5,
) -> dict:
    """Run rank detection for a single condition - matching stable code."""
    rng = np.random.default_rng(seed)

    # Generate embeddings
    w = simulation_dirichlet(n=n, k=true_rank, alpha=alpha, rng=rng)

    # Add noise to W (not S!) if SNR < 1
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)

    # Compute similarity
    similarity = w @ w.T

    # Generate candidate ranks
    lower = max(2, true_rank - grid_span)
    upper = true_rank + grid_span
    candidate_ranks = list(range(lower, upper + 1, 2))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    # Estimate sampling bounds
    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2
    if p_mean <= 0 or p_mean >= 1:
        print(f"Warning: p_mean out of bounds: {p_mean}, using 0.5")
        p_mean = 0.5

    print(f"  Bounds: pmin={pmin:.3f}, pmax={pmax:.3f}, p_mean={p_mean:.3f}")

    # Setup CV
    splitter = EntryMaskSplit(
        n_repeats=cv_repeats,
        sampling_fraction=p_mean,
        random_state=seed,
        missing_values=np.nan,
    )
    masks = list(splitter.split(similarity))

    # SRF estimator matching stable code
    base_estimator = SRF(
        init="random_sqrt",
        random_state=seed,
        max_outer=100,
        max_inner=30,
    )

    # Run CV
    tasks = [
        delayed(fit_and_score)(
            estimator=base_estimator,
            x=similarity,
            train_mask=train_mask,
            validation_mask=validation_mask,
            fit_params={"rank": rank},
            split_idx=split_idx,
        )
        for split_idx, (train_mask, validation_mask) in enumerate(masks)
        for rank in candidate_ranks
    ]

    cv_results = Parallel(n_jobs=-1, verbose=0)(tasks)

    # Aggregate results
    cv_df = pd.DataFrame({
        "rank": res["params"]["rank"],
        "score": res["score"],
        "split": res["split"],
    } for res in cv_results)

    mean_scores = cv_df.groupby("rank")["score"].mean().sort_index()
    best_rank = int(mean_scores.idxmin())

    return {
        "true_rank": true_rank,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - true_rank),
        "is_correct": best_rank == true_rank,
        "p_mean": p_mean,
    }


def main():
    print("Testing rank detection with stable code approach...")
    print()

    # Test parameters
    n = 400
    alphas = [0.5, 1.0, 5.0]
    snrs = [0.2, 0.6, 1.0]
    true_ranks = [5, 10, 15, 20]

    results = []

    for true_rank in true_ranks:
        for alpha in alphas:
            for snr in snrs:
                print(f"Testing: rank={true_rank}, alpha={alpha}, snr={snr}")
                result = run_single_condition(
                    n=n,
                    true_rank=true_rank,
                    alpha=alpha,
                    snr=snr,
                    seed=42,
                    cv_repeats=5,
                )
                results.append({
                    "true_rank": true_rank,
                    "alpha": alpha,
                    "snr": snr,
                    **result,
                })
                print(f"  -> Selected: {result['selected_rank']}, Error: {result['abs_error']}")
                print()

    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nMAE by alpha:")
    print(df.groupby("alpha")["abs_error"].mean().round(2))

    print("\nMAE by SNR:")
    print(df.groupby("snr")["abs_error"].mean().round(2))

    print("\nMAE by true_rank:")
    print(df.groupby("true_rank")["abs_error"].mean().round(2))

    print(f"\nOverall accuracy: {df['is_correct'].mean():.1%}")

    # Save results
    output_path = Path(__file__).parent / "outputs/debug_single.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
