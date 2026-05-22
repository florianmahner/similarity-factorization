"""Compare rank detection with different sampling strategies.

Compares:
1. Bounds-estimated p (current approach)
2. Fixed p = 0.8 (80/20 split)
3. Fixed p = 0.5 (50/50 split)

Usage:
    poetry run python sandbox/rank_detection_viz/run_p_comparison.py
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

# Parameters
N = 200
TRUE_RANKS = [50, 33, 20, 14, 10, 7, 5, 3]  # obs/dof from 2 to 33
ALPHA = 1.0
SNR = 1.0
N_SEEDS = 10
CV_REPEATS = 5
GRID_SPAN = 6

P_METHODS = ["bounds", "0.8", "0.5"]


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def run_single(n: int, k: int, p_method: str, seed: int) -> dict:
    """Run single rank detection with specified p method."""
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

    # Determine p based on method
    if p_method == "bounds":
        pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
        p = (pmin + pmax) / 2
        if p <= 0 or p >= 1:
            p = 0.5
    elif p_method == "0.8":
        p = 0.8
    elif p_method == "0.5":
        p = 0.5
    else:
        raise ValueError(f"Unknown p_method: {p_method}")

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
        "p_method": p_method,
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

    # Show obs/dof for each rank
    print("Rank → obs/dof mapping:")
    for k in TRUE_RANKS:
        print(f"  k={k:3d} → obs/dof = {compute_obs_dof(N, k):.1f}")
    print()

    conditions = [
        (N, k, p_method, seed)
        for k in TRUE_RANKS
        for p_method in P_METHODS
        for seed in range(N_SEEDS)
    ]

    print(f"Running {len(conditions)} conditions...")
    print(f"  Ranks: {TRUE_RANKS}")
    print(f"  P methods: {P_METHODS}")
    print(f"  Seeds: {N_SEEDS}")
    print()

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, p_method, seed)
        for n, k, p_method, seed in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "p_comparison.csv", index=False)

    # Summary
    print("\n" + "="*70)
    print("RESULTS: Accuracy by p_method and obs/dof")
    print("="*70)

    pivot = df.pivot_table(
        values="is_correct",
        index="true_rank",
        columns="p_method",
        aggfunc="mean"
    )
    pivot["obs_dof"] = [compute_obs_dof(N, k) for k in pivot.index]
    pivot = pivot.sort_values("obs_dof")
    print(pivot.round(2))

    print("\n" + "="*70)
    print("RESULTS: Bias by p_method and obs/dof")
    print("="*70)

    pivot_bias = df.pivot_table(
        values="signed_error",
        index="true_rank",
        columns="p_method",
        aggfunc="mean"
    )
    pivot_bias["obs_dof"] = [compute_obs_dof(N, k) for k in pivot_bias.index]
    pivot_bias = pivot_bias.sort_values("obs_dof")
    print(pivot_bias.round(2))

    print("\n" + "="*70)
    print("RESULTS: Mean p_used for bounds method")
    print("="*70)
    bounds_df = df[df["p_method"] == "bounds"]
    p_summary = bounds_df.groupby("true_rank")["p_used"].mean()
    print(p_summary.round(3))


if __name__ == "__main__":
    main()
