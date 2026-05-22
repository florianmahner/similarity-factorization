import numpy as np
from scipy.sparse import csr_matrix, diags

from .utils import compute_link_prediction_metrics


def evaluate_baseline_fast(
    a_sparse: csr_matrix,
    pairs: np.ndarray,
    labels: np.ndarray,
    method: str,
    batch_size: int = 1000000,
    return_scores: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Evaluate baseline link prediction methods.

    Parameters
    ----------
    a_sparse : csr_matrix
        Sparse adjacency matrix (training graph)
    pairs : np.ndarray
        Test pairs (n_pairs, 2) with node indices
    labels : np.ndarray
        Binary labels for test pairs
    method : str
        Method name: "CN", "AA", "RA", or "JC"
    batch_size : int
        Batch size for processing pairs
    return_scores : bool
        If True, return (metrics_dict, scores_array) instead of just metrics_dict

    Returns
    -------
    dict[str, float] or tuple[dict[str, float], np.ndarray]
        Dictionary with metrics, or (metrics_dict, scores_array) if return_scores=True
    """
    n_pairs = len(pairs)
    scores = np.zeros(n_pairs, dtype=np.float32)

    if method == "CN":
        similarity_matrix = a_sparse @ a_sparse
    elif method == "AA":
        degree = np.array(a_sparse.sum(axis=1)).flatten()
        safe_degree = np.maximum(degree, 2.0)
        inv_log_degree = 1.0 / np.log(safe_degree)
        weights_diag = diags(inv_log_degree, format="csr")
        similarity_matrix = a_sparse @ weights_diag @ a_sparse
    elif method == "RA":
        degree = np.array(a_sparse.sum(axis=1)).flatten()
        safe_degree = np.maximum(degree, 1.0)
        inv_degree = 1.0 / safe_degree
        weights_diag = diags(inv_degree, format="csr")
        similarity_matrix = a_sparse @ weights_diag @ a_sparse
    elif method == "JC":
        similarity_matrix = None
    else:
        raise ValueError(f"Unknown method: {method}")

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_pairs = pairs[start:end]

        if method in ["CN", "AA", "RA"]:
            batch_scores = np.array(
                similarity_matrix[batch_pairs[:, 0], batch_pairs[:, 1]]
            ).flatten()

        elif method == "JC":
            batch_size_jc = min(10000, end - start)
            batch_scores = np.zeros(end - start, dtype=np.float32)

            for sub_start in range(0, end - start, batch_size_jc):
                sub_end = min(sub_start + batch_size_jc, end - start)
                sub_pairs = batch_pairs[sub_start:sub_end]

                for idx, (i, j) in enumerate(sub_pairs):
                    neighbors_i = a_sparse[i].indices
                    neighbors_j = a_sparse[j].indices

                    if len(neighbors_i) == 0 or len(neighbors_j) == 0:
                        batch_scores[sub_start + idx] = 0.0
                        continue

                    intersection = np.intersect1d(
                        neighbors_i, neighbors_j, assume_unique=True
                    )
                    union_size = len(neighbors_i) + len(neighbors_j) - len(intersection)

                    batch_scores[sub_start + idx] = (
                        len(intersection) / union_size if union_size > 0 else 0.0
                    )

        scores[start:end] = batch_scores

    metrics = compute_link_prediction_metrics(scores, labels)
    if return_scores:
        return metrics, scores
    return metrics
