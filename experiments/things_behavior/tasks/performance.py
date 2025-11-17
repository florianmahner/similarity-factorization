from __future__ import annotations

import numpy as np

from tools.rsa import correlate_rsms, reconstruct_rsm

from analyses.things.common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
)

from .utils import fit_srf_model, reconstruct_srf_rsm, run_experiment


def spose_performance_trial(
    spose_embedding: np.ndarray,
    vice_embedding: np.ndarray,
    indices_48: np.ndarray,
    rsm_48_true: np.ndarray,
    similarity: np.ndarray,
    validation_triplets: np.ndarray,
    srf_params: dict,
    seed: int = 0,
):
    srf_embedding = fit_srf_model(similarity, srf_params, seed=seed)

    rsm_48_spose = reconstruct_rsm(spose_embedding[indices_48])
    rsm_48_vice = reconstruct_rsm(vice_embedding[indices_48])
    rsm_srf = reconstruct_srf_rsm(srf_embedding)
    rsm_48_srf = rsm_srf[np.ix_(indices_48, indices_48)]

    corr_srf = correlate_rsms(rsm_48_srf, rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)
    corr_vice = correlate_rsms(rsm_48_vice, rsm_48_true)

    acc_srf = compute_triplet_prediction_accuracy(srf_embedding, validation_triplets)
    acc_spose = compute_triplet_prediction_accuracy(
        spose_embedding, validation_triplets
    )
    acc_vice = compute_triplet_prediction_accuracy(vice_embedding, validation_triplets)

    return [
        {"model": "SRF", "correlation": corr_srf, "accuracy": acc_srf, "seed": seed},
        {"model": "VICE", "correlation": corr_vice, "accuracy": acc_vice, "seed": seed},
        {
            "model": "SPoSE",
            "correlation": corr_spose,
            "accuracy": acc_spose,
            "seed": seed,
        },
    ]


def spose_performance_experiment(
    spose_embedding: np.ndarray,
    vice_embedding: np.ndarray,
    indices_48: np.ndarray,
    rsm_48_true: np.ndarray,
    train_triplets: np.ndarray,
    validation_triplets: np.ndarray,
    n_items: int,
    srf_params: dict,
    seeds=range(5),
    **kwargs,
):
    similarity = compute_similarity_matrix_from_triplets(n_items, train_triplets)
    param_grid = {
        "spose_embedding": [spose_embedding],
        "vice_embedding": [vice_embedding],
        "indices_48": [indices_48],
        "rsm_48_true": [rsm_48_true],
        "similarity": [similarity],
        "validation_triplets": [validation_triplets],
        "srf_params": [srf_params],
        "seed": seeds,
    }
    return run_experiment(spose_performance_trial, param_grid, **kwargs)


def spose_48_prediction_trial(
    spose_embedding: np.ndarray,
    indices_48: np.ndarray,
    rsm_48_true: np.ndarray,
    similarity: np.ndarray,
    srf_params: dict,
    seed: int = 0,
):
    srf_embedding = fit_srf_model(similarity, srf_params, seed=seed)

    rsm_48_spose = reconstruct_rsm(spose_embedding[indices_48])
    rsm_srf = reconstruct_srf_rsm(srf_embedding)
    rsm_48_srf = rsm_srf[np.ix_(indices_48, indices_48)]

    n = rsm_48_true.shape[0]
    rows = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "true_similarity": float(rsm_48_true[i, j]),
                    "predicted_similarity": float(rsm_48_srf[i, j]),
                    "model": "SRF",
                    "seed": seed,
                    "pair_idx": pair_idx,
                }
            )
            rows.append(
                {
                    "true_similarity": float(rsm_48_true[i, j]),
                    "predicted_similarity": float(rsm_48_spose[i, j]),
                    "model": "SPoSE",
                    "seed": seed,
                    "pair_idx": pair_idx,
                }
            )
            pair_idx += 1
    return rows


def spose_48_performance_experiment(
    spose_embedding: np.ndarray,
    indices_48: np.ndarray,
    rsm_48_true: np.ndarray,
    train_triplets: np.ndarray,
    n_items: int,
    srf_params: dict,
    seeds=range(5),
    **kwargs,
):
    similarity = compute_similarity_matrix_from_triplets(n_items, train_triplets)
    param_grid = {
        "spose_embedding": [spose_embedding],
        "indices_48": [indices_48],
        "rsm_48_true": [rsm_48_true],
        "similarity": [similarity],
        "srf_params": [srf_params],
        "seed": seeds,
    }
    return run_experiment(spose_48_prediction_trial, param_grid, **kwargs)
