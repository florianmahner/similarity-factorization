#!/usr/bin/env python3


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import argparse
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tools.rsa import compute_similarity

warnings.filterwarnings("ignore")

from cross_validation import cross_validate_admm

METRIC = "linear"


def simulate_rsm_data(n_objects, true_rank, snr=1.0, seed=None):
    """Generate synthetic RSM data with known rank"""
    rng = np.random.default_rng(seed)

    # Generate ground truth embedding
    w_true = rng.random((n_objects, true_rank))

    # TODO Check coherence here!
    rsm = compute_similarity(w_true, w_true, METRIC)

    # Add noise
    if snr > 0:
        noise_std = np.std(rsm) / snr
        noise = rng.normal(0, noise_std, rsm.shape)
        rsm += noise
        rsm = np.maximum(rsm, 0)  # Keep non-negative

    return rsm, true_rank


def evaluate_trial_gridsearch(
    n_objects, train_ratio, true_rank, trial_id, seed, candidate_ranks
):
    """Evaluate all candidate ranks for a single trial using GridSearchCV"""

    # Generate synthetic RSM data
    rsm, _ = simulate_rsm_data(n_objects, true_rank, snr=0.0, seed=seed)

    # Create parameter grid for the candidate ranks
    param_grid = {"rank": candidate_ranks}

    # Use cross-validation to find best rank
    # Note: We use only 1 CV repeat since we want to match the original approach
    results = cross_validate_admm(
        similarity_matrix=rsm,
        param_grid=param_grid,
        n_repeats=1,  # Single split to match original
        train_ratio=train_ratio,
        random_state=seed,
        verbose=0,
        n_jobs=1,  # Avoid nested parallelism
    )

    # Extract results for each rank tested
    cv_results = results.cv_results_
    trial_results = []

    for _, row in cv_results.iterrows():
        # The cross_validate_admm returns negative RMSE as score
        # GridSearchResults.best_score_ converts back to positive RMSE
        # We want MSE, so we need to square the RMSE
        rmse = -row["mean_test_score"]  # Convert back to positive RMSE
        mse = rmse**2  # Convert RMSE to MSE

        trial_results.append(
            {
                "n_objects": n_objects,
                "train_ratio": train_ratio,
                "trial_id": trial_id,
                "rank": row["param_rank"],
                "true_rank": true_rank,
                "mse": mse,
                "seed": seed,
            }
        )

    return trial_results


def run_experiment_gridsearch(
    n_objects_list=[20, 30, 50, 70, 100, 150, 200],
    train_ratios=None,
    true_ranks=[5, 10, 20, 30],
    n_trials_per_condition=10,
    n_jobs=-1,
    output_dir="./results",
    verbose=True,
):
    """Run the full rank detection experiment using GridSearchCV"""
    if train_ratios is None:
        train_ratios = np.linspace(0.1, 0.8, 15)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create trial evaluations (now each trial tests all candidate ranks at once)
    all_trials = []
    seed_counter = 0

    for true_rank in true_ranks:
        for n_obj in n_objects_list:
            for ratio in train_ratios:
                for trial_id in range(n_trials_per_condition):
                    candidate_ranks = list(range(max(1, true_rank - 2), true_rank + 3))
                    all_trials.append(
                        (
                            n_obj,
                            ratio,
                            true_rank,
                            trial_id,
                            seed_counter,
                            candidate_ranks,
                        )
                    )
                    seed_counter += 1

    # Record start time
    start_time = time.time()

    # Run ALL trials in parallel (each trial now tests multiple ranks internally)
    trial_results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(evaluate_trial_gridsearch)(
            n_obj, ratio, true_rank, trial_id, seed, candidate_ranks
        )
        for n_obj, ratio, true_rank, trial_id, seed, candidate_ranks in tqdm(
            all_trials, desc="Running trials (GridSearchCV)", disable=not verbose
        )
    )

    # Flatten results (each trial returns a list of results for different ranks)
    evaluation_results = []
    for trial_result_list in trial_results:
        evaluation_results.extend(trial_result_list)

    # Record end time
    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60

    # Convert to DataFrame
    eval_df = pd.DataFrame(evaluation_results)

    # Save the full evaluation results (all ranks tested)
    results_output_file = output_dir / "rank_detection_full_results_gridsearch.csv"
    eval_df.to_csv(results_output_file, index=False)

    if verbose:
        print(f"Experiment completed in {runtime_minutes:.2f} minutes")
        print(f"Results saved to {results_output_file}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Run rank detection experiment with GridSearchCV"
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=np.logspace(2, 3, 10).astype(int),
        help="List of object counts to test",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[20],
        help="True ranks for synthetic data",
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of trials per condition"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)",
    )
    parser.add_argument(
        "--output", type=str, default="./results", help="Output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    train_ratios = np.arange(0.2, 0.9, 0.1).round(2)

    run_experiment_gridsearch(
        n_objects_list=args.objects,
        train_ratios=train_ratios,
        true_ranks=args.ranks,
        n_trials_per_condition=args.trials,
        n_jobs=args.jobs,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
