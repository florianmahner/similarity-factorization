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

# Import your SRF functions
from experiments.cross_validation import evaluate_rank, train_val_split

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


def evaluate_rank_for_trial(n_objects, train_ratio, true_rank, trial_id, seed, rank):
    """Evaluate a single rank for a single trial - this will be parallelized"""

    # Generate synthetic data
    rng = np.random.default_rng(seed)
    rsm, _ = simulate_rsm_data(n_objects, true_rank, snr=0.0, seed=seed)

    # Create train/val split manually
    mask_train, mask_val = train_val_split(rsm.shape[0], train_ratio, rng)
    # NOTE: I think no matter of the metric used to generate the RSM, I need to use a linear kernel here!!!
    # Make this more explicit by not passing any metric here!
    _, mse = evaluate_rank(rank, rsm, mask_train, mask_val, rng, (None, None), "linear")

    return {
        "n_objects": n_objects,
        "train_ratio": train_ratio,
        "trial_id": trial_id,
        "rank": rank,
        "true_rank": true_rank,
        "mse": mse,
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

    # Create ALL individual rank evaluations for FULL parallelization
    all_evaluations = []
    seed_counter = 0

    # TODO Maybe also check rho range in the analysis!

    for true_rank in true_ranks:
        for n_obj in n_objects_list:
            for ratio in train_ratios:
                for trial_id in range(n_trials_per_condition):

                    candidate_ranks = range(max(1, true_rank - 2), true_rank + 3)

                    for rank in candidate_ranks:
                        all_evaluations.append(
                            (n_obj, ratio, true_rank, trial_id, seed_counter, rank)
                        )
                    seed_counter += 1

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

    # Save the full evaluation results (all ranks tested)
    results_output_file = output_dir / "rank_detection_full_results.csv"
    eval_df.to_csv(results_output_file, index=False)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run rank detection experiment")
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
        default=[10],
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
