#!/usr/bin/env python3
"""
Rank Detection Experiment Script

This script tests how the optimal training ratio for rank detection
depends on the number of objects in RSM matrices, across multiple true ranks.

Usage:
    python rank_detection_experiment.py

Outputs:
    - rank_detection_trial_results.csv: Individual trial results
    - rank_detection_summary.csv: Aggregated results per condition
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import argparse
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Import your SRF functions
from srf.mixed.admm import _evaluate_rank, train_val_split


def simulate_rsm_data(n_objects, true_rank, snr=1.0, seed=None):
    """Generate synthetic RSM data with known rank"""
    rng = np.random.default_rng(seed)

    # Generate ground truth embedding
    w_true = rng.random((n_objects, true_rank))

    # Create RSM
    # rsm = w_true @ w_true.T
    rsm = cosine_similarity(w_true)

    # Add noise
    if snr > 0:
        noise_std = np.std(rsm) / snr
        noise = rng.normal(0, noise_std, rsm.shape)
        rsm += noise
        rsm = np.maximum(rsm, 0)  # Keep non-negative

    return rsm, true_rank


def evaluate_rank_for_trial(n_objects, train_ratio, true_rank, trial_id, seed, rank):
    """Evaluate a single rank for a single trial - this will be parallelized"""
    try:
        # Generate synthetic data
        rng = np.random.default_rng(seed)
        rsm, _ = simulate_rsm_data(n_objects, true_rank, snr=0.0, seed=seed)

        # Create train/val split manually
        mask_train, mask_val = train_val_split(rsm.shape[0], train_ratio, rng)
        bounds = (None, None)
        # Evaluate the rank
        try:
            _, rmse = _evaluate_rank(
                rank, rsm, mask_train, mask_val, rng, bounds, "cosine"
            )
        except Exception as e:
            rmse = np.inf

        return {
            "n_objects": n_objects,
            "train_ratio": train_ratio,
            "trial_id": trial_id,
            "rank": rank,
            "true_rank": true_rank,
            "rmse": rmse,
            "seed": seed,
        }

    except Exception as e:
        print(
            f"Evaluation failed: n={n_objects}, ratio={train_ratio:.3f}, trial={trial_id}, rank={rank}, error={e}"
        )
        return {
            "n_objects": n_objects,
            "train_ratio": train_ratio,
            "trial_id": trial_id,
            "rank": rank,
            "true_rank": true_rank,
            "rmse": np.inf,
            "seed": seed,
        }


def run_experiment(
    n_objects_list=[20, 30, 50, 70, 100, 150, 200],
    train_ratios=None,
    true_ranks=[5, 10, 20],
    n_trials_per_condition=10,
    n_jobs=-1,
    output_dir="./results",
    verbose=True,
):
    """Run the full rank detection experiment"""
    if train_ratios is None:
        train_ratios = np.linspace(0.1, 0.8, 15)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if verbose:
        print("🔬 RANK DETECTION EXPERIMENT")
        print("=" * 50)
        print(f"Object counts: {n_objects_list}")
        print(
            f"Training ratios: {len(train_ratios)} values from {train_ratios.min():.2f} to {train_ratios.max():.2f}"
        )
        print(f"True ranks: {true_ranks}")
        print(f"Trials per condition: {n_trials_per_condition}")
        print(
            f"Total conditions: {len(n_objects_list) * len(train_ratios) * len(true_ranks)}"
        )
        print(
            f"Total trials: {len(n_objects_list) * len(train_ratios) * len(true_ranks) * n_trials_per_condition}"
        )
        print(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all cores'}")
        print(f"Output directory: {output_dir}")
        print()

    # Create ALL individual rank evaluations for FULL parallelization
    all_evaluations = []
    seed_counter = 0

    for true_rank in true_ranks:
        for n_obj in n_objects_list:
            for ratio in train_ratios:
                for trial_id in range(n_trials_per_condition):
                    # For each trial, evaluate all candidate ranks
                    candidate_ranks = range(max(1, true_rank - 2), true_rank + 2)
                    for rank in candidate_ranks:
                        all_evaluations.append(
                            (n_obj, ratio, true_rank, trial_id, seed_counter, rank)
                        )
                    seed_counter += 1

    if verbose:
        print(
            f"🚀 Running {len(all_evaluations)} individual rank evaluations in parallel..."
        )
        print()

    # Record start time
    start_time = time.time()

    # Run ALL evaluations in parallel
    evaluation_results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(evaluate_rank_for_trial)(n_obj, ratio, true_rank, trial_id, seed, rank)
        for n_obj, ratio, true_rank, trial_id, seed, rank in tqdm(
            all_evaluations, desc="Running evaluations", disable=not verbose
        )
    )

    # Record end time
    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60

    # Convert to DataFrame
    eval_df = pd.DataFrame(evaluation_results)

    # Find best rank for each trial
    trial_results = []
    for (true_rank, n_obj, ratio, trial_id), group in eval_df.groupby(
        ["true_rank", "n_objects", "train_ratio", "trial_id"]
    ):
        best_rank = group.loc[group["rmse"].idxmin()]
        trial_results.append(
            {
                "n_objects": n_obj,
                "train_ratio": ratio,
                "trial_id": trial_id,
                "detected_rank": best_rank["rank"],
                "true_rank": true_rank,
                "success": 1 if best_rank["rank"] == true_rank else 0,
                "seed": best_rank["seed"],
                "best_rmse": best_rank["rmse"],
                "n_candidate_ranks": len(group),
            }
        )

    # Convert to DataFrame
    trial_df = pd.DataFrame(trial_results)

    # Add experiment metadata
    trial_df["experiment_timestamp"] = pd.Timestamp.now()
    trial_df["runtime_minutes"] = runtime_minutes

    # Save individual trial results
    trial_output_file = output_dir / "rank_detection_trial_results.csv"
    trial_df.to_csv(trial_output_file, index=False)

    # Aggregate to get accuracy per condition (now including true_rank)
    summary_df = (
        trial_df.groupby(["true_rank", "n_objects", "train_ratio"])
        .agg(
            {
                "success": ["mean", "std", "count"],
                "best_rmse": ["mean", "std"],
                "detected_rank": lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else -1
                ),
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary_df.columns = [
        "true_rank",
        "n_objects",
        "train_ratio",
        "accuracy_mean",
        "accuracy_std",
        "n_trials_completed",
        "rmse_mean",
        "rmse_std",
        "most_common_detected_rank",
    ]

    # Add confidence intervals (assuming normal distribution)
    summary_df["accuracy_ci_lower"] = summary_df["accuracy_mean"] - 1.96 * summary_df[
        "accuracy_std"
    ] / np.sqrt(summary_df["n_trials_completed"])
    summary_df["accuracy_ci_upper"] = summary_df["accuracy_mean"] + 1.96 * summary_df[
        "accuracy_std"
    ] / np.sqrt(summary_df["n_trials_completed"])

    # Save summary results
    summary_output_file = output_dir / "rank_detection_summary.csv"
    summary_df.to_csv(summary_output_file, index=False)

    if verbose:
        print("Experiment completed!")
        print(f"Runtime: {runtime_minutes:.1f} minutes")
        print(f"Individual trials: {len(trial_df)}")
        print(f"Rank not detected: {len(trial_df[trial_df['success'] == 0])}")
        print(f"Overall success rate: {trial_df['success'].mean():.1%}")
        print()
        print("Success rate by true rank:")
        for true_rank in true_ranks:
            subset = trial_df[trial_df["true_rank"] == true_rank]
            print(
                f"  Rank {true_rank}: {subset['success'].mean():.1%} ({len(subset)} trials)"
            )
        print()
        print("Output files:")
        print(f"  - Individual trials: {trial_output_file}")
        print(f"  - Summary statistics: {summary_output_file}")
        print()
        print("Sample results:")
        print(summary_df.head())

    return trial_df, summary_df


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run rank detection experiment")
    parser.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 2000],
        help="List of object counts to test",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[10],
        help="True ranks for synthetic data",
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials per condition"
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

    # train_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # train_ratios = np.linspace(0.1, 0.8, 5)
    # train_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    train_ratios = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

    run_experiment(
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
