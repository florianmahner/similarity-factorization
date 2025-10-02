import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from pysrf import SRF
from tools.rsa import correlate_rsms, reconstruct_rsm_batched, compute_similarity
from utils.simulation import add_noise_with_snr
from utils.helpers import best_pairwise_match

from tqdm import tqdm
import itertools

from experiments.things.common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
)
from pysrf import cross_val_score


def run_experiment(trial_func, param_grid, n_jobs=-1, verbose=True):
    """General experiment runner - like cross_val_score but for any experiment."""
    param_combinations = list(ParameterGrid(param_grid))

    if verbose:
        print(f"Running {len(param_combinations)} parameter combinations")

    results = Parallel(n_jobs=n_jobs)(
        delayed(trial_func)(**params)
        for params in tqdm(param_combinations, desc="Running experiment")
    )

    return pd.DataFrame(itertools.chain(*results))


def fit_admm_model(
    similarity: np.ndarray, params: dict, seed: int = None
) -> np.ndarray:
    """Fit ADMM model with given parameters."""
    local_params = params.copy()
    local_params["random_state"] = seed
    model = SRF(**local_params)
    return model.fit_transform(similarity)


# --------------------------------------------------------------------------------------------
# Pairwise reconstruction of SPoSE dimensions from the similarity matrix at various SNRs
# --------------------------------------------------------------------------------------------


def reconstruct_admm_rsm(w: np.ndarray) -> np.ndarray:
    """Reconstruct similarity matrix from ADMM factors."""
    sim = w @ w.T
    sim_copy = sim.copy()
    np.fill_diagonal(sim_copy, 0)

    min_val, max_val = sim_copy.min(), sim_copy.max()
    normalized = (
        (sim - min_val) / (max_val - min_val) if max_val > min_val else sim.copy()
    )
    np.fill_diagonal(normalized, 1)
    return normalized


# Trial functions - each takes params and returns list of dicts
def pairwise_reconstruction_trial(
    estimator, spose_embedding, snr=1.0, similarity_measure="cosine", seed=0
):
    """Single trial of pairwise reconstruction."""
    noisy_spose = add_noise_with_snr(spose_embedding, snr)
    spose_rsm = compute_similarity(noisy_spose, spose_embedding, similarity_measure)

    cloned_estimator = clone(estimator)
    cloned_estimator.set_params(random_state=seed)
    w = cloned_estimator.fit_transform(spose_rsm)
    corrs = best_pairwise_match(spose_embedding, w)

    return [
        {
            "dimension": i,
            "correlation": corr,
            "snr": snr,
            "similarity_measure": similarity_measure,
            "seed": seed,
        }
        for i, corr in enumerate(corrs)
    ]


# Convenience functions for common experiment patterns
def pairwise_reconstruction_experiment(
    estimator,
    spose_embedding,
    snr_values=[1.0],
    similarity_measures=["cosine"],
    seeds=range(5),
    **kwargs,
):
    """Run pairwise reconstruction experiment with parameter grid."""
    param_grid = {
        "estimator": [estimator],
        "spose_embedding": [spose_embedding],
        "snr": snr_values,
        "similarity_measure": similarity_measures,
        "seed": seeds,
    }
    return run_experiment(pairwise_reconstruction_trial, param_grid, **kwargs)


# --------------------------------------------------------------------------------------------
# SPoSE vs ADMM comparison in terms of performance on the triplet data and 48-word RDM
# --------------------------------------------------------------------------------------------


def spose_performance_trial(
    spose_embedding,
    indices_48,
    rsm_48_true,
    similarity,
    validation_triplets,
    admm_params,
    seed=0,
):
    """Single trial of SPoSE vs ADMM comparison."""
    # Fit ADMM
    admm_embedding = fit_admm_model(similarity, admm_params, seed=seed)

    # Compute correlations and accuracies
    rsm_48_spose = reconstruct_rsm_batched(spose_embedding[indices_48])
    rsm_admm = reconstruct_admm_rsm(admm_embedding)
    rsm_48_admm = rsm_admm[np.ix_(indices_48, indices_48)]

    corr_admm = correlate_rsms(rsm_48_admm, rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)

    acc_admm = compute_triplet_prediction_accuracy(admm_embedding, validation_triplets)
    acc_spose = compute_triplet_prediction_accuracy(
        spose_embedding, validation_triplets
    )

    return [
        {"model": "ADMM", "correlation": corr_admm, "accuracy": acc_admm, "seed": seed},
        {
            "model": "SPoSE",
            "correlation": corr_spose,
            "accuracy": acc_spose,
            "seed": seed,
        },
    ]


def spose_performance_experiment(
    spose_embedding,
    indices_48,
    rsm_48_true,
    train_triplets,
    validation_triplets,
    n_items,
    admm_params,
    seeds=range(5),
    **kwargs,
):
    """Run SPoSE vs ADMM comparison experiment."""
    similarity = compute_similarity_matrix_from_triplets(n_items, train_triplets)
    param_grid = {
        "spose_embedding": [spose_embedding],
        "indices_48": [indices_48],
        "rsm_48_true": [rsm_48_true],
        "similarity": [similarity],
        "validation_triplets": [validation_triplets],
        "admm_params": [admm_params],
        "seed": seeds,
    }
    return run_experiment(spose_performance_trial, param_grid, **kwargs)


def spose_48_prediction_trial(
    spose_embedding,
    indices_48,
    rsm_48_true,
    similarity,
    admm_params,
    seed=0,
):
    """Single trial of SPoSE vs ADMM comparison."""
    # Fit ADMM
    admm_embedding = fit_admm_model(similarity, admm_params, seed=seed)

    # Compute correlations and accuracies
    rsm_48_spose = reconstruct_rsm_batched(spose_embedding[indices_48])
    rsm_admm = reconstruct_admm_rsm(admm_embedding)
    rsm_48_admm = rsm_admm[np.ix_(indices_48, indices_48)]

    n = rsm_48_true.shape[0]
    rows = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "true_similarity": float(rsm_48_true[i, j]),
                    "predicted_similarity": float(rsm_48_admm[i, j]),
                    "model": "ADMM",
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
    spose_embedding,
    indices_48,
    rsm_48_true,
    train_triplets,
    n_items,
    admm_params,
    seeds=range(5),
    **kwargs,
):
    """Run SPoSE vs ADMM comparison experiment."""
    similarity = compute_similarity_matrix_from_triplets(n_items, train_triplets)
    param_grid = {
        "spose_embedding": [spose_embedding],
        "indices_48": [indices_48],
        "rsm_48_true": [rsm_48_true],
        "similarity": [similarity],
        "admm_params": [admm_params],
        "seed": seeds,
    }
    return run_experiment(spose_48_prediction_trial, param_grid, **kwargs)


def subsample_triplets(
    triplets: np.ndarray, percentage: float, seed: int
) -> np.ndarray:
    """Subsample triplets by percentage."""
    if percentage >= 1.0:
        return triplets
    np.random.seed(seed)
    n_samples = int(len(triplets) * percentage)
    indices = np.random.choice(len(triplets), size=n_samples, replace=False)
    return triplets[indices]


def low_data_trial(
    train_triplets,
    validation_triplets,
    n_items,
    admm_params,
    data_percentage=1.0,
    seed=0,
):
    """Single trial of low data experiment."""
    subsampled_triplets = subsample_triplets(train_triplets, data_percentage, seed)
    similarity = compute_similarity_matrix_from_triplets(n_items, subsampled_triplets)

    admm_embedding = fit_admm_model(similarity, admm_params, seed=seed)
    acc_admm = compute_triplet_prediction_accuracy(admm_embedding, validation_triplets)

    return [
        {
            "model": "ADMM",
            "data_percentage": data_percentage,
            "n_triplets": len(subsampled_triplets),
            "accuracy": acc_admm,
            "seed": seed,
        }
    ]


def low_data_experiment(
    train_triplets,
    validation_triplets,
    n_items,
    admm_params,
    data_percentages=[0.1, 0.5, 1.0],
    seeds=range(5),
    **kwargs,
):
    """Run low data experiment."""
    param_grid = {
        "train_triplets": [train_triplets],
        "validation_triplets": [validation_triplets],
        "n_items": [n_items],
        "admm_params": [admm_params],
        "data_percentage": data_percentages,
        "seed": seeds,
    }
    return run_experiment(low_data_trial, param_grid, **kwargs)


# --------------------------------------------------------------------------------------------
# Dimension reliability analysis (reproducibility across model runs)
# --------------------------------------------------------------------------------------------


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


def compute_dimension_reliability(
    similarity_matrix: np.ndarray,
    admm_params: dict,
    n_runs: int = 20,
    n_jobs: int = -1,
) -> tuple[np.ndarray, list[np.ndarray]]:

    # Run all fits (original + references) in parallel
    all_embeddings = Parallel(n_jobs=n_jobs)(
        delayed(fit_admm_model)(
            similarity=similarity_matrix,
            params=admm_params,
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


def run_dimension_reliability_analysis(
    spose_triplets: np.ndarray,
    admm_params: dict,
    n_runs: int = 20,
    n_jobs: int = -1,
    output_path: str = None,
) -> pd.DataFrame:

    similarity_matrix = compute_similarity_matrix_from_triplets(1854, spose_triplets)

    dimension_reliabilities = compute_dimension_reliability(
        similarity_matrix, admm_params, n_runs=n_runs, n_jobs=n_jobs
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


# --------------------------------------------------------------------------------------------
# Dimension estimation experiment - cross-validation of SPoSE dimensions
# --------------------------------------------------------------------------------------------


def run_spose_dimensionality_analysis(
    train_triplets: np.ndarray,
    rank_range: range = range(5, 90, 5),
    n_repeats: int = 5,
    observed_fractions: list[float] = [0.8],
    n_jobs: int = -1,
):
    """Run SPoSE dimension estimation analysis on the training dataset only."""

    similarity = compute_similarity_matrix_from_triplets(1854, train_triplets)

    param_grid = {
        "rank": list(rank_range),
        "max_outer": [10],
        "max_inner": [50],
        "tol": [0.0],
        "rho": [1.0],
    }
    all_cv_results = []
    for observed_fraction in observed_fractions:
        scorer = cross_val_score(
            similarity,
            param_grid=param_grid,
            n_repeats=n_repeats,
            sampling_fraction=observed_fraction,
            random_state=0,
            verbose=0,
            n_jobs=n_jobs,
            fit_final_estimator=False,
        )
        cv_results = pd.DataFrame(scorer.cv_results_)
        cv_results["sampling_fraction"] = observed_fraction
        all_cv_results.append(cv_results)

    return pd.concat(all_cv_results)
