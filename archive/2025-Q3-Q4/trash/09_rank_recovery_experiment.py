#!/usr/bin/env python3
"""
Experiment to test rank recovery accuracy across different sample sizes,
true ranks, and ADMM hyperparameters using cross-validation. What we test here
is actually how well we can recover the true rank of a similarity matrix as a function of 
the training duration. It seems to me that maybe the fraction is training similarities might not be important, 
but rather the training duration. 
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import time

from cross_validation import cross_val_score


def simulate_data(
    n_samples: int, true_rank: int, noise_level: float = 0.0, seed: int = 42
) -> np.ndarray:
    """Generate simulated similarity matrix with known ground truth rank."""
    rng = np.random.default_rng(seed)

    w_true = rng.random((n_samples, true_rank))
    s_true = w_true @ w_true.T

    if noise_level > 0:
        noise = rng.normal(0, noise_level, s_true.shape)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        s_true = s_true + noise

    s_true = np.clip(s_true, 0, None)
    np.fill_diagonal(s_true, np.diag(w_true @ w_true.T))

    return s_true


def evaluate_single_condition(
    n_samples: int,
    true_rank: int,
    max_outer: int,
    max_inner: int,
    rho: float,
    trial_id: int,
    seed: int,
) -> dict:
    """Evaluate rank recovery for a single parameter combination."""

    # Generate data
    s_matrix = simulate_data(n_samples, true_rank, noise_level=0.0, seed=seed)

    # Define rank search range around true rank
    rank_range = list(range(max(1, true_rank - 2), true_rank + 4))

    param_grid = {
        "rank": rank_range,
        "max_outer": [max_outer],
        "max_inner": [max_inner],
        "rho": [rho],
        "tol": [0.0],
    }

    # Run cross-validation
    grid_search = cross_val_score(
        s_matrix,
        param_grid=param_grid,
        n_repeats=1,
        observed_fraction=0.8,
        n_jobs=1,  # Single job since we parallelize at higher level
        random_state=seed,
        verbose=0,
    )

    # Find best rank
    cv_results = grid_search.cv_results_
    mean_scores = cv_results.groupby("rank")["score"].mean()
    best_rank = mean_scores.idxmin()
    best_score = mean_scores.min()

    return {
        "n_samples": n_samples,
        "true_rank": true_rank,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "rho": rho,
        "trial_id": trial_id,
        "best_rank": best_rank,
        "best_score": best_score,
        "rank_correct": best_rank == true_rank,
        "rank_error": abs(best_rank - true_rank),
        "seed": seed,
    }


def run_rank_recovery_experiment(
    n_samples_list: list[int] = None,
    true_ranks: list[int] = None,
    max_outer_values: list[int] = None,
    max_inner_values: list[int] = None,
    rho_values: list[float] = None,
    n_trials: int = 10,
    n_jobs: int = -1,
    output_dir: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the complete rank recovery experiment."""

    # Generate all parameter combinations
    all_conditions = []
    seed_counter = 0

    for n_samples in n_samples_list:
        for true_rank in true_ranks:
            for max_outer in max_outer_values:
                for max_inner in max_inner_values:
                    for rho in rho_values:
                        for trial_id in range(n_trials):
                            all_conditions.append(
                                (
                                    n_samples,
                                    true_rank,
                                    max_outer,
                                    max_inner,
                                    rho,
                                    trial_id,
                                    seed_counter,
                                )
                            )
                            seed_counter += 1

    if verbose:
        total_conditions = len(all_conditions)
        print(f"\nTotal evaluations: {total_conditions}")
        print("Starting experiment...")

    # Run experiment
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(evaluate_single_condition)(
            n_samples, true_rank, max_outer, max_inner, rho, trial_id, seed
        )
        for n_samples, true_rank, max_outer, max_inner, rho, trial_id, seed in tqdm(
            all_conditions, desc="Evaluating conditions", disable=not verbose
        )
    )

    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "cross_validation/rank_recovery.csv"
    results_df.to_csv(output_file, index=False)

    return results_df


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Run rank recovery experiment")

    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=np.logspace(np.log10(100), np.log10(2000), 10).astype(int).tolist(),
        help="Sample sizes to test",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="True ranks to test",
    )
    parser.add_argument(
        "--max-outer",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="Max outer iteration values",
    )
    parser.add_argument(
        "--max-inner",
        nargs="+",
        type=int,
        default=[50, 100],
        help="Max inner iteration values",
    )
    parser.add_argument(
        "--rho",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Rho values to test",
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of trials per condition"
    )
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    run_rank_recovery_experiment(
        n_samples_list=args.samples,
        true_ranks=args.ranks,
        max_outer_values=args.max_outer,
        max_inner_values=args.max_inner,
        rho_values=args.rho,
        n_trials=args.trials,
        n_jobs=args.jobs,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
