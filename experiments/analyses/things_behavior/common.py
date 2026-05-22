from __future__ import annotations

import numpy as np
from pysrf import SRF

from src.utils.helpers import compute_similarity_matrix_from_triplets  # noqa: F401


def compute_triplet_prediction_accuracy(
    embedding: np.ndarray, triplets: np.ndarray
) -> float:
    idx = triplets.astype(int)
    ei, ej, ek = embedding[idx[:, 0]], embedding[idx[:, 1]], embedding[idx[:, 2]]
    sij = np.sum(ei * ej, axis=1)
    sik = np.sum(ei * ek, axis=1)
    sjk = np.sum(ej * ek, axis=1)
    return float(np.mean((sij > sik) & (sij > sjk)))


def fit_srf_model(
    similarity: np.ndarray,
    rank: int,
    seed: int | None = None,
) -> np.ndarray:
    model = SRF(
        rank=rank,
        random_state=seed,
        max_outer=2000,
        max_inner=50,
        tol=1e-4,
        verbose=0,
    )
    return model.fit_transform(similarity)
