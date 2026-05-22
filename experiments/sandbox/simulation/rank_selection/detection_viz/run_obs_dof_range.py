"""Simulation covering obs/dof from 1 to 50.

n=200, ranks chosen to span the full obs/dof range.

Usage:
    poetry run python sandbox/rank_detection_viz/run_obs_dof_range.py
"""

from __future__ import annotations

import sys
from datetime import datetime
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
N = 200
# Ranks chosen to give obs/dof ≈ [1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50]
# obs/dof = (n-1)/(2k) for n=200 → obs/dof = 99.5/k
TRUE_RANKS = [100, 66, 50, 33, 20, 14, 10, 7, 5, 3, 2]
ALPHA = 1.0  # Reference case
SNR = 1.0    # Reference case
N_SEEDS = 10
CV_REPEATS = 5
GRID_SPAN = 6


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def run_single(n: int, k: int, alpha: float, snr: float, seed: int) -> dict:
    """Run single rank detection."""
    rng = np.random.default_rng(seed)

    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)
    similarity = w @ w.T

    # Candidate ranks
    lower = max(2, k - GRID_SPAN)
    upper = min(k + GRID_SPAN, n - 1)
    candidate_ranks = list(range(lower, upper + 1, 2))
    if k not in candidate_ranks:
        candidate_ranks.append(k)
    candidate_ranks = sorted(set(candidate_ranks))

    # Estimate bounds
    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2
    if p_mean <= 0 or p_mean >= 1:
        p_mean = 0.5

    # CV
    splitter = EntryMaskSplit(
        n_repeats=CV_REPEATS,
        sampling_fraction=p_mean,
        random_state=seed,
    )
    masks = list(splitter.split(similarity))

    base_estimator = SRF(
        init="random_sqrt",
        random_state=seed,
        max_outer=100,
        max_inner=30,
    )

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

    cv_df = pd.DataFrame({
        "rank": res["params"]["rank"],
        "score": res["score"],
    } for res in cv_results)

    mean_scores = cv_df.groupby("rank")["score"].mean()
    best_rank = int(mean_scores.idxmin())

    return {
        "n": n,
        "true_rank": k,
        "obs_per_dof": compute_obs_dof(n, k),
        "alpha": alpha,
        "snr": snr,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - k),
        "signed_error": best_rank - k,
        "is_correct": best_rank == k,
        "p_mean": p_mean,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Show obs/dof for each rank
    print("Rank → obs/dof mapping:")
    for k in TRUE_RANKS:
        print(f"  k={k:3d} → obs/dof = {compute_obs_dof(N, k):.1f}")
    print()

    conditions = [
        (N, k, ALPHA, SNR, seed)
        for k in TRUE_RANKS
        for seed in range(N_SEEDS)
    ]

    print(f"Running {len(conditions)} conditions (n={N}, alpha={ALPHA}, SNR={SNR})...")

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, alpha, snr, seed)
        for n, k, alpha, snr, seed in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "rank_detection_obs_dof_range.csv", index=False)

    print("\nBias by obs/dof:")
    summary = df.groupby("true_rank").agg(
        obs_dof=("obs_per_dof", "first"),
        bias=("signed_error", "mean"),
        mae=("abs_error", "mean"),
        acc=("is_correct", "mean"),
    ).sort_values("obs_dof")
    print(summary.round(2))


if __name__ == "__main__":
    main()
