"""" 
This module contains the code for the dimension reliability experiment of spose behavior. 
We want to know how reliable the dimensions of the embedding are across model runs when factoring in the model's random seed. 
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def fisher_z_transform(r: float) -> float:
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def find_best_matching_dimension(
    original_dim: np.ndarray, reference_embedding: np.ndarray
) -> float:
    correlations = []
    for dim_idx in range(reference_embedding.shape[1]):
        ref_dim = reference_embedding[:, dim_idx]
        corr = np.corrcoef(original_dim, ref_dim)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))
    return max(correlations) if correlations else 0.0


def fit_single_embedding(
    similarity_matrix: np.ndarray, mask: np.ndarray, admm_params: dict, seed: int
) -> np.ndarray:
    from models.admm import ADMM

    try:
        model = ADMM(**admm_params, mask=mask, random_state=seed)
        W = model.fit_transform(similarity_matrix)
        return W
    except Exception:
        return None


def compute_dimension_reliability(
    similarity_matrix: np.ndarray,
    mask: np.ndarray,
    admm_params: dict,
    n_runs: int = 20,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, List[np.ndarray]]:

    # Run all fits (original + references) in parallel
    all_embeddings = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_embedding)(
            similarity_matrix=similarity_matrix,
            mask=mask,
            admm_params=admm_params,
            seed=seed,
        )
        for seed in range(0, n_runs + 1)
    )
    original_embedding = all_embeddings[0]
    reference_embeddings = all_embeddings[1:]
    if original_embedding is None:
        raise RuntimeError("Failed to fit original embedding")

    reference_embeddings = [emb for emb in reference_embeddings if emb is not None]
    if len(reference_embeddings) == 0:
        raise RuntimeError("No reference embeddings succeeded")

    n_dims = original_embedding.shape[1]
    dimension_reliabilities = np.zeros(n_dims)

    for dim_idx in range(n_dims):
        original_dim = original_embedding[:, dim_idx]
        best_correlations = []

        for ref_embedding in reference_embeddings:
            best_corr = find_best_matching_dimension(original_dim, ref_embedding)
            best_correlations.append(best_corr)

        if best_correlations:
            fisher_z_values = [fisher_z_transform(corr) for corr in best_correlations]
            mean_fisher_z = np.mean(fisher_z_values)
            dimension_reliabilities[dim_idx] = inverse_fisher_z(mean_fisher_z)
        else:
            dimension_reliabilities[dim_idx] = 0.0

    return dimension_reliabilities


def run_reproducibility_analysis(
    similarity_matrix: np.ndarray,
    mask: np.ndarray,
    admm_params: dict,
    n_runs: int = 20,
    n_jobs: int = -1,
    output_path: str = None,
) -> pd.DataFrame:

    dimension_reliabilities = compute_dimension_reliability(
        similarity_matrix, mask, admm_params, n_runs=n_runs, n_jobs=n_jobs
    )

    n_dims = len(dimension_reliabilities)
    results_data = []
    for dim_idx in range(n_dims):
        results_data.append(
            {
                "Dimension": dim_idx,
                "Reliability": dimension_reliabilities[dim_idx],
            }
        )

    results_df = pd.DataFrame(results_data)

    if output_path:
        results_df.to_csv(output_path, index=False)

    return results_df
