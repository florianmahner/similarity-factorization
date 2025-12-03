from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from ..lib import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
    fit_srf_model,
    load_resources,
)


def _subsample_triplets(triplets: np.ndarray, percentage: float, seed: int) -> np.ndarray:
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed + 1000)
    return triplets[rng.choice(len(triplets), size=n_samples, replace=False)]


def run(cfg: DictConfig) -> None:
    resources = load_resources(cfg)

    triplets = _subsample_triplets(resources.train_triplets, cfg.percentage, cfg.seed)
    similarity = compute_similarity_matrix_from_triplets(cfg.n_items, triplets)
    embedding = fit_srf_model(similarity, rank=cfg.dims, seed=cfg.seed)
    accuracy = compute_triplet_prediction_accuracy(embedding, resources.validation_triplets)

    result = {
        "model": "SRF",
        "percentage": cfg.percentage,
        "n_triplets": len(triplets),
        "accuracy": float(accuracy),
        "seed": cfg.seed,
    }

    with open(Path.cwd() / "results.json", "w") as f:
        json.dump([result], f)
