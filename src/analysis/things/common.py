import numpy as np
from pysrf import SRF


CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


def compute_similarity_matrix_from_triplets(
    n: int, triplets: np.ndarray, alpha: float = 0.0
) -> np.ndarray:
    """Compute similarity matrix from triplet data with Laplace smoothing.

    Triplet format: [i, j, k] where (i,j) is the chosen similar pair and k is odd one out.

    Uses Laplace smoothing: (counts + alpha) / (shown + 2alpha)
    - alpha acts as pseudocounts: assume each pair has alpha virtual "chosen" and alpha "not chosen"
    - Prevents extreme 0.0 and 1.0 from single observations
    - alpha = 1.0 (default): add-one smoothing
    - alpha = 0.5: Jeffreys prior (weaker smoothing)

    With alpha = 1.0:
    - Pair shown once, chosen: (1+1)/(1+2) = 0.67 (instead of 1.0)
    - Pair shown once, not chosen: (0+1)/(1+2) = 0.33 (instead of 0.0)
    - Pair shown many times: converges to counts/shown
    """
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
        counts + alpha,
        shown + 2 * alpha,
        out=np.nan * np.ones_like(counts),
        where=shown != 0,
    )

    np.fill_diagonal(similarity, 1.0)
    return similarity


# def compute_similarity_matrix_from_triplets(n: int, triplets: np.ndarray) -> np.ndarray:
#     """
#     Computes a true count-based matrix from triplet data for Poisson NMF.

#     This version correctly treats "shown-but-not-chosen" (count=0)
#     as missing data (NaN), just like "never-shown" pairs.
#     The model will *only* train on the positive counts.
#     """
#     # 'counts' = how many times (i,j) was the *chosen* pair
#     counts = np.zeros((n, n), dtype=np.float32)

#     # We only need to populate the 'counts'
#     for i, j, k in triplets:
#         if i != j:
#             counts[i, j] += 1
#             counts[j, i] += 1

#     # 1. Start with a matrix full of NaNs (missing)
#     count_matrix = np.full((n, n), np.nan, dtype=np.float32)

#     # 2. Find all pairs that were *chosen at least once*
#     #    This is the new definition of our mask.
#     observed_mask = counts > 0

#     # 3. For those pairs, fill in their raw "chosen" count.
#     #    (Pairs with counts=0 will remain NaN, which is correct)
#     count_matrix[observed_mask] = counts[observed_mask]

#     # 4. Set the diagonal to NaN (no self-interactions)
#     np.fill_diagonal(count_matrix, np.nan)

#     return count_matrix


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


def fit_srf_model(similarity: np.ndarray, params: dict, seed: int = None) -> np.ndarray:
    """Fit SRF model with given parameters."""
    local_params = params.copy()
    local_params["random_state"] = seed
    model = SRF(**local_params)
    return model.fit_transform(similarity)
