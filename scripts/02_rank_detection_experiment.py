#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from tools.rsa import compute_similarity
from utils.simulation import add_noise_with_snr
from cross_validation import mask_missing_entries, fit_and_score
from models.admm import ADMM

METRIC = "linear"


def simulate_rsm_data(n_objects, true_rank, snr=0.0, seed=None):
    """Generate synthetic RSM data with known rank"""
    rng = np.random.default_rng(seed)

    # Generate ground truth embedding
    w_true = rng.random((n_objects, true_rank))
    w_true = add_noise_with_snr(w_true, snr, rng=rng)
    rsm = compute_similarity(w_true, w_true, METRIC)

    return rsm


def evaluate_rank_for_trial(
    n_objects: int,
    observed_fraction: float,
    true_rank: int,
    trial_id: int,
    seed: int,
    rank: int,
    snr: float,
):
    """Evaluate a single rank for a single trial - this will be parallelized"""

    rng = np.random.default_rng(seed)
    rsm = simulate_rsm_data(n_objects, true_rank, snr=snr, seed=seed)

    val_mask = mask_missing_entries(rsm, observed_fraction, rng=rng)

    estimator = ADMM(
        random_state=seed,
        init="random_sqrt",
        rho=1.0,
        max_outer=100,
        max_inner=10,
        tol=0.0,
    )
    params = {"rank": rank}
    result = fit_and_score(estimator, rsm, val_mask, params, split_idx=0)
    mse = result["score"]

    return {
        "n_objects": n_objects,
        "observed_fraction": observed_fraction,
        "trial_id": trial_id,
        "rank": rank,
        "true_rank": true_rank,
        "mse": mse,
        "seed": seed,
        "snr": snr,
    }


def run_experiment(
    n_objects_list=[20, 30, 50, 70, 100, 150, 200],
    observed_fractions=None,
    true_ranks=[5, 10, 20],
    snrs=np.arange(0.0, 1.1, 0.1),
    n_trials_per_condition=5,
    n_jobs=-1,
    output_dir="./results",
    verbose=True,
):
    """Run the full rank detection experiment"""
    if observed_fractions is None:
        observed_fractions = np.linspace(0.1, 0.8, 15)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_evaluations = []
    seed_counter = 0

    # TODO Maybe also check rho range in the analysis!
    for true_rank in true_ranks:
        for n_obj in n_objects_list:
            for frac in observed_fractions:
                for trial_id in range(n_trials_per_condition):
                    for snr in snrs:

                        candidate_ranks = range(max(1, true_rank - 2), true_rank + 3)

                        for rank in candidate_ranks:
                            all_evaluations.append(
                                (
                                    n_obj,
                                    frac,
                                    true_rank,
                                    trial_id,
                                    seed_counter,
                                    rank,
                                    snr,
                                )
                            )
                        seed_counter += 1

    # Record start time
    start_time = time.time()

    # Run ALL evaluations in parallel
    evaluation_results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(evaluate_rank_for_trial)(
            n_obj, frac, true_rank, trial_id, seed, rank, snr
        )
        for n_obj, frac, true_rank, trial_id, seed, rank, snr in tqdm(
            all_evaluations, desc="Running evaluations", disable=not verbose
        )
    )

    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60

    eval_df = pd.DataFrame(evaluation_results)
    results_output_file = output_dir / "rank_detection_results_test.csv"
    eval_df.to_csv(results_output_file, index=False)

    if verbose:
        print(f"Experiment completed in {runtime_minutes:.2f} minutes")
        print(f"Results saved to {results_output_file}")


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
        "--trials", type=int, default=10, help="Number of trials per condition"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=140,
        help="Number of parallel jobs (-1 for all cores)",
    )
    parser.add_argument(
        "--output", type=str, default="./results", help="Output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    observed_fractions = np.arange(0.2, 0.9, 0.1).round(2)
    # snrs = np.array([0.3, 0.7, 1.0]).round(2)
    observed_fractions = np.array([0.8]).round(2)
    snrs = np.array([1.0])

    run_experiment(
        n_objects_list=args.objects,
        observed_fractions=observed_fractions,
        true_ranks=args.ranks,
        snrs=snrs,
        n_trials_per_condition=args.trials,
        n_jobs=args.jobs,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
