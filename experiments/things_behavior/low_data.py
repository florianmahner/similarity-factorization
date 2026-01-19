"""THINGS Behavior: Low data regime experiment with non-overlapping partitions.

Tests SRF performance on subsampled triplet data using non-overlapping partitions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from .common import compute_triplet_prediction_accuracy, fit_srf_model
from .resources import load_resources

from src.utils.helpers import compute_similarity_matrix_from_triplets


def _get_partition(
    triplets: np.ndarray, percentage: float, partition_idx: int, seed: int = 42
) -> np.ndarray:
    """Get partition k of non-overlapping splits at given percentage.

    Args:
        triplets: Full training triplets array
        percentage: Fraction of data per partition (e.g., 0.05 for 5%)
        partition_idx: Which partition to return (0-indexed)
        seed: Random seed for shuffling (default 42 for reproducibility)

    Returns:
        Triplet array for the specified partition
    """
    if percentage >= 1.0:
        return triplets

    n_partitions = int(1.0 / percentage)
    if partition_idx >= n_partitions:
        raise ValueError(f"partition_idx {partition_idx} >= n_partitions {n_partitions}")

    rng = np.random.default_rng(seed)
    shuffled = triplets.copy()
    rng.shuffle(shuffled)

    part_size = len(triplets) // n_partitions
    start = partition_idx * part_size
    end = start + part_size
    return shuffled[start:end]


def _subsample_triplets(triplets: np.ndarray, percentage: float, seed: int) -> np.ndarray:
    """Legacy random subsampling (kept for backwards compatibility)."""
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed + 1000)
    return triplets[rng.choice(len(triplets), size=n_samples, replace=False)]


def run(cfg: DictConfig) -> None:
    resources = load_resources(cfg)

    use_partitions = getattr(cfg, "partition_idx", None) is not None

    if use_partitions:
        triplets = _get_partition(
            resources.train_triplets,
            cfg.percentage,
            cfg.partition_idx,
            seed=42,
        )
    else:
        triplets = _subsample_triplets(resources.train_triplets, cfg.percentage, cfg.seed)

    similarity = compute_similarity_matrix_from_triplets(cfg.n_items, triplets)

    srf_params = {
        "rank": cfg.dims,
        "max_outer": 2000,
        "max_inner": 50,
        "tol": 1e-4,
        "verbose": False,
    }
    embedding = fit_srf_model(similarity, srf_params, seed=cfg.seed)
    accuracy = compute_triplet_prediction_accuracy(embedding, resources.validation_triplets)

    result = {
        "model": "SRF",
        "percentage": cfg.percentage,
        "n_triplets": len(triplets),
        "accuracy": float(accuracy),
        "seed": cfg.seed,
    }

    if use_partitions:
        result["partition_idx"] = cfg.partition_idx

    with open(Path.cwd() / "results.json", "w") as f:
        json.dump([result], f)
