import numpy as np
from pysrf import SRF

from utils.helpers import compute_similarity_matrix_from_triplets

CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


def softmax_triplet_choice(w_i: np.ndarray, w_j: np.ndarray, w_k: np.ndarray) -> bool:
    similarities = np.array([w_i @ w_j, w_i @ w_k, w_j @ w_k])
    probas = np.exp(similarities) / np.sum(np.exp(similarities))
    return np.argmax(probas) == 0


def compute_triplet_prediction_accuracy(
    embedding: np.ndarray, triplets: np.ndarray
) -> float:
    acc = 0
    for i, j, k in triplets:
        acc += softmax_triplet_choice(embedding[i], embedding[j], embedding[k])
    return acc / len(triplets)


def fit_srf_model(similarity: np.ndarray, params: dict, seed: int = None) -> np.ndarray:
    local_params = params.copy()
    local_params["random_state"] = seed
    model = SRF(**local_params)
    return model.fit_transform(similarity)
