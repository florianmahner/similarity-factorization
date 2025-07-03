#!/usr/bin/env python3
"""
Unified rank detection experiment combining hyperparameter sensitivity and robustness testing.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from cross_validation import mask_missing_entries, fit_and_score
from models.admm import ADMM
from utils.helpers import add_noise_with_snr
from tools.rsa import compute_similarity


def simulate_similarity_matrix(
    n_objects, true_rank, snr=1.0, seed=42, similarity_measure="cosine"
):
    """Generate similarity matrix with optional noise and masking."""
    rng = np.random.default_rng(seed)

    w_true = rng.random((n_objects, true_rank))
    w_true = add_noise_with_snr(w_true, snr)
    s_matrix = compute_similarity(w_true, w_true, similarity_measure)

    return s_matrix


def evaluate_single_rank(
    n_objects,
    true_rank,
    observed_fraction,
    snr,
    rho,
    max_outer,
    max_inner,
    trial_id,
    seed,
    similarity_measure,
    rank,
):
    """Evaluate a single rank for a single condition - fully parallelizable."""

    # Generate similarity matrix
    s_matrix = simulate_similarity_matrix(
        n_objects, true_rank, snr, seed, similarity_measure
    )

    # Generate validation mask
    rng = np.random.default_rng(seed)
    val_mask = mask_missing_entries(s_matrix, observed_fraction, rng)

    # Create estimator and parameters
    estimator = ADMM(random_state=seed)
    params = {
        "rank": rank,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "rho": rho,
        "tol": 1e-6,
    }

    # Fit and score
    result = fit_and_score(estimator, s_matrix, val_mask, params, split_idx=0)

    return {
        "n_objects": n_objects,
        "true_rank": true_rank,
        "observed_fraction": observed_fraction,
        "snr": snr,
        "rho": rho,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "trial_id": trial_id,
        "rank": rank,
        "score": result["score"],
        "seed": seed,
        "similarity_measure": similarity_measure,
    }


def run_rank_experiment(
    n_objects_list=[200, 500],
    true_ranks=[10, 20],
    observed_fractions=[1.0],
    snr_values=[np.inf],
    rho_values=[1.0],
    max_outer_values=[20],
    max_inner_values=[50],
    n_trials=5,
    n_jobs=-1,
    output_dir="results",
    similarity_measure="cosine",
):
    """Main experiment orchestrator."""
    jobs = []
    seed_counter = 0

    for n_objects in n_objects_list:
        for true_rank in true_ranks:
            for observed_fraction in observed_fractions:
                for snr in snr_values:
                    for rho in rho_values:
                        for max_outer in max_outer_values:
                            for max_inner in max_inner_values:
                                for trial_id in range(n_trials):
                                    # Create jobs for each rank
                                    rank_range = list(
                                        range(max(1, true_rank - 2), true_rank + 3)
                                    )
                                    for rank in rank_range:
                                        jobs.append(
                                            (
                                                n_objects,
                                                true_rank,
                                                observed_fraction,
                                                snr,
                                                rho,
                                                max_outer,
                                                max_inner,
                                                trial_id,
                                                seed_counter,
                                                similarity_measure,
                                                rank,
                                            )
                                        )
                                    seed_counter += 1

    print(f"Running {len(jobs)} individual rank evaluations in parallel")

    all_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_single_rank)(*job) for job in jobs
    )

    # Group by condition and find best rank
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        Path(output_dir) / "cross_validation" / "rank_experiment_simulation_raw.csv",
        index=False,
    )

    condition_cols = [
        "n_objects",
        "true_rank",
        "observed_fraction",
        "snr",
        "rho",
        "max_outer",
        "max_inner",
    ]

    final_results = []
    for condition_values, group in results_df.groupby(condition_cols):
        condition_dict = dict(zip(condition_cols, condition_values))

        # Compute mean score for each rank across trials
        rank_scores = group.groupby("rank")["score"].mean()
        best_rank = rank_scores.idxmin()
        best_score = rank_scores.min()

        final_results.append(
            {
                **condition_dict,
                "best_rank": int(best_rank),
                "best_score": best_score,
                "rank_correct": best_rank == condition_dict["true_rank"],
                "rank_error": abs(best_rank - condition_dict["true_rank"]),
                "similarity_measure": similarity_measure,
            }
        )

    output_path = (
        Path(output_dir) / "cross_validation" / "rank_experiment_simulation.csv"
    )
    final_df = pd.DataFrame(final_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    return final_df


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Unified rank detection experiment")
    parser.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=np.logspace(np.log10(50), np.log10(300), 10).astype(int).tolist(),
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=np.arange(0.2, 1.0, 0.2).round(2),
    )
    parser.add_argument("--snr", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--rho", nargs="+", type=float, default=[1.0])
    parser.add_argument("--max-outer", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--max-inner", nargs="+", type=int, default=[100])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--similarity", type=str, default="cosine")
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    run_rank_experiment(
        n_objects_list=args.objects,
        true_ranks=args.ranks,
        observed_fractions=args.fractions,
        snr_values=args.snr,
        rho_values=args.rho,
        max_outer_values=args.max_outer,
        max_inner_values=args.max_inner,
        n_trials=args.trials,
        n_jobs=args.jobs,
        output_dir=args.output,
        similarity_measure=args.similarity,
    )


if __name__ == "__main__":
    main()
