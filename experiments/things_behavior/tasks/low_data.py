from __future__ import annotations

import numpy as np

from analyses.things.common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
)

from .utils import fit_srf_model, run_experiment


def subsample_triplets(
    triplets: np.ndarray, percentage: float, seed: int
) -> np.ndarray:
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed + 1000)
    indices = rng.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


def low_data_trial(
    train_triplets: np.ndarray,
    validation_triplets: np.ndarray,
    n_items: int,
    srf_params: dict,
    data_percentage: float = 1.0,
    seed: int = 0,
):
    subsampled = subsample_triplets(train_triplets, data_percentage, seed)
    similarity = compute_similarity_matrix_from_triplets(n_items, subsampled)

    embedding = fit_srf_model(similarity, srf_params, seed=seed)
    accuracy = compute_triplet_prediction_accuracy(embedding, validation_triplets)

    return [
        {
            "model": "SRF",
            "data_percentage": data_percentage,
            "n_triplets": len(subsampled),
            "accuracy": accuracy,
            "seed": seed,
        }
    ]


def low_data_experiment(
    train_triplets: np.ndarray,
    validation_triplets: np.ndarray,
    n_items: int,
    srf_params: dict,
    data_percentages=(0.1, 0.5, 1.0),
    seeds=range(5),
    **kwargs,
):
    param_grid = {
        "train_triplets": [train_triplets],
        "validation_triplets": [validation_triplets],
        "n_items": [n_items],
        "srf_params": [srf_params],
        "data_percentage": data_percentages,
        "seed": seeds,
    }
    return run_experiment(low_data_trial, param_grid, **kwargs)
