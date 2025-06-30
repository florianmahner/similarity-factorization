"""
Sparse SPoSE Reconstruction Analysis

This script compares ADMM and SPoSE models for reconstructing similarity matrices
from human triplet judgments and evaluates their performance against ground truth.
Runs multiple seeds in parallel for statistical robustness.
"""

from pathlib import Path

from experiments.things import (
    load_shared_data,
    load_triplets,
    low_data_experiment,
    pairwise_reconstruction_experiment,
    spose_comparison_experiment,
)
from models.admm import ADMM


def load_from_toml(path: Path):
    with open(path, "r") as f:
        return toml.load(f)


def main():
    """Main analysis pipeline with parallel execution across seeds."""
    print(f"\nStarting analysis with {N_SEEDS} seeds using {N_WORKERS} workers...")

    THINGS_DATASET_PATH = Path("data/things")
    THINGS_IMAGES_PATH = Path("/SSD/datasets/things/images")

    spose_embedding, indices_48, rsm_48_true = load_shared_data(
        THINGS_DATASET_PATH,
        THINGS_IMAGES_PATH,
        max_dim=49,
    )

    train_triplets, validation_triplets = load_triplets(THINGS_DATASET_PATH)

    admm_params = {
        "rank": 49,
        "max_outer": 20,
        "max_inner": 50,
        "rho": 1.0,
        "tol": 0.0,
    }
    estimator = ADMM(**admm_params)

    df = pairwise_reconstruction_experiment(
        estimator,
        spose_embedding,
        seeds=range(10),
        snr_values=[1.0],
        similarity_measures=["cosine"],
    )
    df.to_csv(RESULTS_DIR / "pairwise_reconstruction.csv", index=False)

    df = spose_comparison_experiment(
        spose_embedding,
        indices_48,
        rsm_48_true,
        train_triplets,
        validation_triplets,
        n_items=1854,
        admm_params=admm_params,
        seeds=range(10),
    )
    df.to_csv(RESULTS_DIR / "spose_comparison.csv", index=False)

    df = low_data_experiment(
        train_triplets,
        validation_triplets,
        n_items=1854,
        admm_params=admm_params,
        data_percentages=[0.05, 0.10, 0.20, 0.50, 1.0],
        seeds=range(10),
    )
    df.to_csv(RESULTS_DIR / "low_data.csv", index=False)


if __name__ == "__main__":
    main()
