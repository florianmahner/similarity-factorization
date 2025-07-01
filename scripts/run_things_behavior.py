"""
Sparse SPoSE Reconstruction Analysis

This script compares ADMM and SPoSE models for reconstructing similarity matrices
from human triplet judgments and evaluates their performance against ground truth.
Runs multiple seeds in parallel for statistical robustness.
"""

import argparse
from pathlib import Path
import numpy as np
from experiments.things.things import (
    low_data_experiment,
    pairwise_reconstruction_experiment,
    spose_performance_experiment,
    run_dimension_reliability_analysis,
    run_spose_dimensionality_analysis,
)
from utils.io import load_shared_data, load_triplets
from models.admm import ADMM


def main():
    """Main analysis pipeline with parallel execution across seeds."""
    parser = argparse.ArgumentParser(description="Run THINGS analysis experiments")
    parser.add_argument(
        "--experiment",
        choices=[
            "pairwise",
            "spose_performance",
            "low_data",
            "dimension_reliability",
            "spose_dimensionality",
            "all",
        ],
        required=True,
        help="Which experiment to run",
    )

    args = parser.parse_args()

    # Setup paths and results directory
    THINGS_DATASET_PATH = Path("data/things")
    THINGS_IMAGES_PATH = Path("/SSD/datasets/things")
    DIMS = 49
    RESULTS_DIR = Path(f"results/things")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning experiment: {args.experiment}")

    # Load common data
    spose_embedding, indices_48, rsm_48_true = load_shared_data(
        THINGS_DATASET_PATH,
        THINGS_IMAGES_PATH,
        num_dims=DIMS,
    )

    train_triplets, validation_triplets = load_triplets(THINGS_DATASET_PATH)

    admm_params = {
        "rank": DIMS,
        "max_outer": 10,
        "max_inner": 50,
        "rho": 1.0,
        "tol": 0.0,
    }
    estimator = ADMM(**admm_params)

    # Run selected experiment
    if args.experiment == "pairwise" or args.experiment == "all":
        print("Running pairwise reconstruction experiment...")
        df = pairwise_reconstruction_experiment(
            estimator,
            spose_embedding,
            seeds=range(10),
            snr_values=[1.0],
            similarity_measures=["cosine"],
        )
        df.to_csv(RESULTS_DIR / DIMS / "pairwise_reconstruction.csv", index=False)

    if args.experiment == "spose_performance" or args.experiment == "all":
        print("Running SPoSE performance experiment...")
        df = spose_performance_experiment(
            spose_embedding,
            indices_48,
            rsm_48_true,
            train_triplets,
            validation_triplets,
            n_items=1854,
            admm_params=admm_params,
            seeds=range(10),
        )
        df.to_csv(RESULTS_DIR / DIMS / "spose_comparison.csv", index=False)

    if args.experiment == "low_data" or args.experiment == "all":
        print("Running low data experiment...")
        df = low_data_experiment(
            train_triplets,
            validation_triplets,
            n_items=1854,
            admm_params=admm_params,
            data_percentages=[0.05, 0.10, 0.20, 0.50, 1.0],
            seeds=range(10),
        )
        df.to_csv(RESULTS_DIR / "low_data.csv", index=False)

    if args.experiment == "dimension_reliability" or args.experiment == "all":
        print("Running dimension reliability analysis...")
        df = run_dimension_reliability_analysis(
            train_triplets,
            admm_params,
            n_runs=10,
            n_jobs=-1,
        )
        df.to_csv(RESULTS_DIR / "dimension_reliability.csv", index=False)

    if args.experiment == "spose_dimensionality" or args.experiment == "all":
        print("Running SPoSE dimensionality analysis...")
        df = run_spose_dimensionality_analysis(
            train_triplets,
            rank_range=range(5, 90, 5),
            n_repeats=5,
            observed_fractions=np.arange(0.3, 0.9, 0.1),
            n_jobs=-1,
        )
        df.to_csv(RESULTS_DIR / "spose_cross_validation.csv", index=False)

    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
