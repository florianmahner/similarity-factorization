"""Run comprehensive rank detection simulation - matching stable code.

n=400, ranks 5,10,15,20, proper bounds estimation.

Usage:
    poetry run python sandbox/rank_detection_viz/run_full_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF, estimate_sampling_bounds_fast
from pysrf.cross_validation import EntryMaskSplit, fit_and_score

from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

# Parameters
N = 400
TRUE_RANKS = [5, 10, 15, 20]
ALPHAS = [0.5, 1.0, 5.0]
SNRS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 5
CV_REPEATS = 5
GRID_SPAN = 8


def run_single(n: int, k: int, alpha: float, snr: float, seed: int) -> dict:
    """Run single rank detection - matching stable code exactly."""
    rng = np.random.default_rng(seed)

    # Generate embeddings
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)

    # Add noise to W (not S!) if SNR < 1
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)

    # Compute similarity
    similarity = w @ w.T

    # Generate candidate ranks
    lower = max(2, k - GRID_SPAN)
    upper = k + GRID_SPAN
    candidate_ranks = list(range(lower, upper + 1, 2))
    if k not in candidate_ranks:
        candidate_ranks.append(k)
    candidate_ranks = sorted(set(candidate_ranks))

    # Estimate sampling bounds
    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2
    if p_mean <= 0 or p_mean >= 1:
        p_mean = 0.5  # fallback

    # Setup CV splitter
    splitter = EntryMaskSplit(
        n_repeats=CV_REPEATS,
        sampling_fraction=p_mean,
        random_state=seed,
        missing_values=np.nan,
    )
    masks = list(splitter.split(similarity))

    # SRF estimator - matching stable code
    base_estimator = SRF(
        init="random_sqrt",
        random_state=seed,
        max_outer=100,
        max_inner=30,
    )

    # Run CV fits
    cv_results = []
    for split_idx, (train_mask, validation_mask) in enumerate(masks):
        for rank in candidate_ranks:
            result = fit_and_score(
                estimator=base_estimator,
                x=similarity,
                train_mask=train_mask,
                validation_mask=validation_mask,
                fit_params={"rank": rank},
                split_idx=split_idx,
            )
            cv_results.append(result)

    # Aggregate results
    cv_df = pd.DataFrame({
        "rank": res["params"]["rank"],
        "score": res["score"],
        "split": res["split"],
    } for res in cv_results)

    mean_scores = cv_df.groupby("rank")["score"].mean().sort_index()
    best_rank = int(mean_scores.idxmin())

    return {
        "n": n,
        "true_rank": k,
        "alpha": alpha,
        "snr": snr,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - k),
        "signed_error": best_rank - k,
        "is_correct": best_rank == k,
        "pmin": pmin,
        "pmax": pmax,
        "p_mean": p_mean,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all conditions
    conditions = [
        (N, k, alpha, snr, seed)
        for k in TRUE_RANKS
        for alpha in ALPHAS
        for snr in SNRS
        for seed in range(N_SEEDS)
    ]

    n_conditions = len(conditions)
    print(f"Running {n_conditions} conditions...")
    print(f"  n = {N}")
    print(f"  True ranks: {TRUE_RANKS}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  SNRs: {SNRS}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  CV repeats: {CV_REPEATS}")

    results = Parallel(n_jobs=128, verbose=10)(
        delayed(run_single)(n, k, alpha, snr, seed)
        for n, k, alpha, snr, seed in conditions
    )

    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "rank_detection_full_grid.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} results to {output_path}")

    # Summary
    print("\nMAE by alpha:")
    print(df.groupby("alpha")["abs_error"].mean().round(2))

    print("\nMAE by SNR:")
    print(df.groupby("snr")["abs_error"].mean().round(2))

    print("\nOverall accuracy (exact):", f"{df['is_correct'].mean():.1%}")
    print("Within ±1:", f"{(df['abs_error'] <= 1).mean():.1%}")
    print("Within ±2:", f"{(df['abs_error'] <= 2).mean():.1%}")


if __name__ == "__main__":
    main()
