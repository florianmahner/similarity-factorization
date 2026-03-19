"""Test if higher n rescues high-k rank detection.

If obs/dof is the right metric, then k=50 should work with large enough n.

Usage:
    poetry run python sandbox/rank_detection_viz/run_n_comparison.py
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

from src.utils.simulation import simulation_dirichlet
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

# Test conditions: vary n to get similar obs/dof for different k
# Smaller n values for faster runtime
CONDITIONS = [
    # (n, k) pairs chosen to give obs/dof ≈ 10
    (200, 10),   # obs/dof = 199/20 = 10.0
    (300, 15),   # obs/dof = 299/30 = 10.0
    (400, 20),   # obs/dof = 399/40 = 10.0
    # Also test same k=20 with different n (different obs/dof)
    (100, 20),   # obs/dof = 99/40 = 2.5 (should fail)
    (200, 20),   # obs/dof = 199/40 = 5.0 (borderline)
    (300, 20),   # obs/dof = 299/40 = 7.5 (borderline)
    (400, 20),   # obs/dof = 399/40 = 10.0 (should work)
]

ALPHA = 1.0
SNR = 1.0
N_SEEDS = 5
CV_REPEATS = 5
GRID_SPAN = 6


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def run_single(n: int, k: int, seed: int) -> dict:
    """Run single rank detection."""
    rng = np.random.default_rng(seed)

    w = simulation_dirichlet(n=n, k=k, alpha=ALPHA, rng=rng)
    similarity = w @ w.T

    # Candidate ranks
    lower = max(2, k - GRID_SPAN)
    upper = min(k + GRID_SPAN, n - 1)
    candidate_ranks = list(range(lower, upper + 1, 2))
    if k not in candidate_ranks:
        candidate_ranks.append(k)
    candidate_ranks = sorted(set(candidate_ranks))

    # Bounds estimation
    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p = (pmin + pmax) / 2
    if p <= 0 or p >= 1:
        p = 0.5

    # CV
    splitter = EntryMaskSplit(
        n_repeats=CV_REPEATS,
        sampling_fraction=p,
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
        "p_used": p,
        "seed": seed,
        "selected_rank": best_rank,
        "abs_error": abs(best_rank - k),
        "signed_error": best_rank - k,
        "is_correct": best_rank == k,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Show conditions
    print("Conditions (n, k) → obs/dof:")
    for n, k in CONDITIONS:
        print(f"  n={n:4d}, k={k:2d} → obs/dof = {compute_obs_dof(n, k):.1f}")
    print()

    # Deduplicate conditions
    unique_conditions = list(set(CONDITIONS))

    all_conditions = [
        (n, k, seed)
        for n, k in unique_conditions
        for seed in range(N_SEEDS)
    ]

    print(f"Running {len(all_conditions)} conditions with {N_SEEDS} seeds each...")
    print()

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, seed)
        for n, k, seed in all_conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "n_comparison.csv", index=False)

    # Summary
    print("\n" + "="*70)
    print("RESULTS: Same k=50, varying n")
    print("="*70)
    k50 = df[df["true_rank"] == 50].copy()
    summary = k50.groupby("n").agg(
        obs_dof=("obs_per_dof", "first"),
        accuracy=("is_correct", "mean"),
        bias=("signed_error", "mean"),
        p_mean=("p_used", "mean"),
    ).round(2)
    print(summary)

    print("\n" + "="*70)
    print("RESULTS: Different k, all at obs/dof ≈ 10")
    print("="*70)
    obs10 = df[df["obs_per_dof"].between(9, 11)].copy()
    summary2 = obs10.groupby(["n", "true_rank"]).agg(
        obs_dof=("obs_per_dof", "first"),
        accuracy=("is_correct", "mean"),
        bias=("signed_error", "mean"),
    ).round(2)
    print(summary2)


if __name__ == "__main__":
    main()
