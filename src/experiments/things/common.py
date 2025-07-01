import numpy as np
from models.admm import ADMM


CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


def compute_similarity_matrix_from_triplets(n: int, triplets: np.ndarray) -> np.ndarray:
    """Compute similarity matrix from triplet data."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))

    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1

        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1

    similarity = np.divide(
        counts,
        shown,
        out=np.nan * np.ones_like(counts),
        where=shown != 0,
        # This below is another way, but more conservative. We place NaN and not zeors!!
        # where=(shown != 0) & (counts > 0),  # Only where actually chosen
    )

    np.fill_diagonal(similarity, 1.0)
    return similarity


def softmax_triplet_choice(w_i: np.ndarray, w_j: np.ndarray, w_k: np.ndarray) -> bool:
    """Determine if triplet choice is correct using softmax."""
    similarities = np.array([w_i @ w_j, w_i @ w_k, w_j @ w_k])
    probas = np.exp(similarities) / np.sum(np.exp(similarities))
    return np.argmax(probas) == 0


def compute_triplet_prediction_accuracy(
    embedding: np.ndarray, triplets: np.ndarray
) -> float:
    """Compute accuracy of triplet predictions."""
    acc = 0
    for i, j, k in triplets:
        acc += softmax_triplet_choice(embedding[i], embedding[j], embedding[k])
    return acc / len(triplets)


def fit_admm_model(
    similarity: np.ndarray, params: dict, seed: int = None
) -> np.ndarray:
    """Fit ADMM model with given parameters."""
    local_params = params.copy()
    local_params["random_state"] = seed
    model = ADMM(**local_params)
    return model.fit_transform(similarity)
