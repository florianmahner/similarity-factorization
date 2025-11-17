from __future__ import annotations

import numpy as np
from sklearn.base import clone

from tools.rsa import compute_similarity
from utils.helpers import best_pairwise_match
from utils.simulation import add_noise_with_snr

from .utils import run_experiment


def pairwise_reconstruction_trial(
    estimator,
    spose_embedding: np.ndarray,
    snr: float = 1.0,
    similarity_measure: str = "linear",
    seed: int = 0,
):
    noisy_spose = add_noise_with_snr(spose_embedding, snr)
    spose_rsm = compute_similarity(noisy_spose, spose_embedding, similarity_measure)

    cloned_estimator = clone(estimator)
    cloned_estimator.set_params(random_state=seed)
    w = cloned_estimator.fit_transform(spose_rsm)
    corrs = best_pairwise_match(spose_embedding, w)

    return [
        {
            "dimension": idx,
            "correlation": corr,
            "snr": snr,
            "similarity_measure": similarity_measure,
            "seed": seed,
        }
        for idx, corr in enumerate(corrs)
    ]


def pairwise_reconstruction_experiment(
    estimator,
    spose_embedding: np.ndarray,
    snr_values: list[float] | tuple[float, ...] = (1.0,),
    similarity_measures: list[str] | tuple[str, ...] = ("linear",),
    seeds=range(5),
    **kwargs,
):
    param_grid = {
        "estimator": [estimator],
        "spose_embedding": [spose_embedding],
        "snr": snr_values,
        "similarity_measure": similarity_measures,
        "seed": seeds,
    }
    return run_experiment(pairwise_reconstruction_trial, param_grid, **kwargs)

