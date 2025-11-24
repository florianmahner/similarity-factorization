from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from tools.rsa import correlate_rsms, reconstruct_rsm

from ..lib.common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
)
from ..lib.resources import load_resources
from ..lib.utils import fit_srf_model


def run(cfg: DictConfig) -> None:
    """
    Atomic task: Run ONE performance trial.
    Hydra handles sweeping over seeds.
    """
    resources = load_resources(cfg)
    seed = cfg.seed

    # Compute similarity matrix
    similarity = compute_similarity_matrix_from_triplets(
        cfg.n_items, resources.train_triplets
    )

    # Fit SRF model
    srf_embedding = fit_srf_model(similarity, rank=cfg.dims, seed=seed)

    # Reconstruct RSMs
    rsm_48_spose = reconstruct_rsm(resources.spose_embedding[resources.indices_48])
    rsm_48_vice = reconstruct_rsm(resources.vice_embedding[resources.indices_48])
    rsm_srf = reconstruct_rsm(srf_embedding)
    rsm_48_srf = rsm_srf[np.ix_(resources.indices_48, resources.indices_48)]

    # Compute correlations
    corr_srf = correlate_rsms(rsm_48_srf, resources.rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, resources.rsm_48_true)
    corr_vice = correlate_rsms(rsm_48_vice, resources.rsm_48_true)

    # Compute accuracies
    acc_srf = compute_triplet_prediction_accuracy(
        srf_embedding, resources.validation_triplets
    )
    acc_spose = compute_triplet_prediction_accuracy(
        resources.spose_embedding, resources.validation_triplets
    )
    acc_vice = compute_triplet_prediction_accuracy(
        resources.vice_embedding, resources.validation_triplets
    )

    # Save results as JSON (multiple rows per job - one per model)
    results = [
        {
            "model": "SRF",
            "correlation": float(corr_srf),
            "accuracy": float(acc_srf),
            "seed": int(seed),
        },
        {
            "model": "VICE",
            "correlation": float(corr_vice),
            "accuracy": float(acc_vice),
            "seed": int(seed),
        },
        {
            "model": "SPoSE",
            "correlation": float(corr_spose),
            "accuracy": float(acc_spose),
            "seed": int(seed),
        },
    ]

    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
