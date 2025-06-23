"""
Sparse SPoSE Reconstruction Analysis

This script compares ADMM and SPoSE models for reconstructing similarity matrices
from human triplet judgments and evaluates their performance against ground truth.
Runs multiple seeds in parallel for statistical robustness.
"""

# TODO evaluate this on a low data regime compared to VICE
# TODO replace path with this repository's data directory
import numpy as np
from experiments.spose import (
    load_shared_data,
    run_single_seed,
    run_low_data_experiment,
    compute_summary_statistics,
    compute_low_data_summary_statistics,
    run_spose_reconstruction_simulation,
)
from pathlib import Path
import multiprocessing as mp
from joblib import Parallel, delayed
import pandas as pd

from models.admm import ADMM
from utils.helpers import load_spose_embedding


# Data paths
MAX_DIM = 49
DATA_ROOT = Path("/LOCAL/fmahner/model-comparisons/data")
SPOSE_EMBEDDING_PATH = DATA_ROOT / "misc" / f"spose_embedding_{MAX_DIM}d.txt"
WORDS48_PATH = DATA_ROOT / "misc" / "words48.csv"
GROUND_TRUTH_RSM_PATH = DATA_ROOT / "misc" / "rdm48_human.mat"
TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "trainset.txt"
VALIDATION_TRIPLETS_PATH = DATA_ROOT / "human_triplets" / "validationset.txt"

# Dataset paths
THINGS_DATASET_PATH = "/SSD/datasets/things"

# Output paths
RESULTS_DIR = Path(f"results/spose/{MAX_DIM}")
OUTPUT_CSV = RESULTS_DIR / f"sparse_spose_recon.csv"
SUMMARY_CSV = RESULTS_DIR / f"sparse_spose_recon_summary.csv"
LOW_DATA_CSV = RESULTS_DIR / f"sparse_spose_recon_low_data.csv"
LOW_DATA_SUMMARY_CSV = RESULTS_DIR / f"sparse_spose_recon_low_data_summary.csv"

# Model parameters
ADMM_PARAMS = {
    "rank": MAX_DIM,
    "max_outer": 20,
    "w_inner": 10,
    "verbose": False,
    "tol": 0.0,
    "rho": 1.0,
    "init": "random",
}

# Experiment parameters
N_SEEDS = 10
N_WORKERS = min(mp.cpu_count(), 10)  # 10 is roughly 512 gb mem

# Low data regime parameters
DATA_PERCENTAGES = [0.05, 0.10, 0.20, 0.50, 1.0]  # 5%, 10%, 20%, 50%, 100%

# Data parameters
N_ITEMS = 1854
GROUND_TRUTH_RSM_KEY = "RDM48_triplet"

# Category name replacements
CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}



def main():
    """Main analysis pipeline with parallel execution across seeds."""
    print(f"\nStarting analysis with {N_SEEDS} seeds using {N_WORKERS} workers...")

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (
        spose_embedding,
        indices_48,
        rsm_48_true,
        similarity,
        mask,
        validation_triplets,
        full_triplets
    ) = load_shared_data(
        SPOSE_EMBEDDING_PATH,
        WORDS48_PATH,
        THINGS_DATASET_PATH,
        CATEGORY_REPLACEMENTS,
        GROUND_TRUTH_RSM_PATH,
    )

    print(f"Running main analysis with {N_SEEDS} seeds in parallel...")

    breakpoint()

    # Run main analysis
    results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(run_single_seed)(
            seed,
            spose_embedding,
            indices_48,
            rsm_48_true,
            similarity,
            mask,
            validation_triplets,
            ADMM_PARAMS,
        )
        for seed in range(N_SEEDS)
    )

    all_results = []
    all_similarity_data = []
    successful_seeds = 0
    for seed_results in results:
        if seed_results is not None and seed_results[0] is not None:
            all_results.extend(seed_results[0])
            all_similarity_data.extend(seed_results[1])
            successful_seeds += 1

    print(f"Successfully completed {successful_seeds}/{N_SEEDS} seeds")

    results_df = pd.DataFrame(all_results)
    similarity_df = pd.DataFrame(all_similarity_data)

    summary_df = compute_summary_statistics(results_df)
    
    print(f"\nRunning low data regime experiments...")
    print(f"Data percentages: {[f'{p:.1%}' for p in DATA_PERCENTAGES]}")

    low_data_jobs = []
    for seed in range(N_SEEDS):
        for data_percentage in DATA_PERCENTAGES:
            low_data_jobs.append((seed, data_percentage))

    print(f"Running {len(low_data_jobs)} low data experiments in parallel...")

    low_data_results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(run_low_data_experiment)(
            seed,
            data_percentage,
            full_triplets,
            ADMM_PARAMS,
            validation_triplets,
        )
        for seed, data_percentage in low_data_jobs
    )

    # Process low data results
    low_data_valid_results = [r for r in low_data_results if r is not None]
    print(
        f"Successfully completed {len(low_data_valid_results)}/{len(low_data_jobs)} low data experiments"
    )

    if low_data_valid_results:
        low_data_df = pd.DataFrame(low_data_valid_results)
        low_data_summary_df = compute_low_data_summary_statistics(low_data_df)
    else:
        print("Warning: No successful low data regime results!")
        low_data_df = pd.DataFrame()
        low_data_summary_df = pd.DataFrame()

    results_df.to_csv(OUTPUT_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    similarity_comparison_csv = RESULTS_DIR / "similarity_comparison.csv"
    similarity_df.to_csv(similarity_comparison_csv, index=False)

    if not low_data_df.empty:
        low_data_df.to_csv(LOW_DATA_CSV, index=False)
        low_data_summary_df.to_csv(LOW_DATA_SUMMARY_CSV, index=False)


    print("Running SPoSE reconstruction simulation...")


    spose_embedding = load_spose_embedding(
        max_objects=N_ITEMS, max_dims=MAX_DIM, num_dims=MAX_DIM
    )
    model = ADMM(
        rank=MAX_DIM,
        max_outer=100,
        max_inner=10,
        tol=0.0,
        rho=1.0,
        init="random_sqrt",
        verbose=True,
    )

    # Run the simulation with different seeds
    df = run_spose_reconstruction_simulation(
        model,
        spose_embedding,
        seeds=np.arange(30),
        snr=1.0,  # Fixed SNR (no noise)
        similarity_measure="cosine",
    )
    path = Path(f"results/spose/{MAX_DIM}/spose_reconstruction.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



if __name__ == "__main__":
    main()
