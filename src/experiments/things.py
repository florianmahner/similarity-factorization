import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from models.admm import ADMM
from tools.rsa import correlate_rsms, reconstruct_rsm_batched, compute_similarity
from utils.simulation import add_noise_with_snr
from utils.helpers import best_pairwise_match
from utils.io import load_spose_embedding, load_concept_mappings, load_words48
from tqdm import tqdm
import itertools

CATEGORY_REPLACEMENTS = {"camera": "camera1", "file": "file1"}


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


# Helper functions for data processing
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
    )
    np.fill_diagonal(similarity, 1.0)
    return similarity


def fit_admm_model(
    similarity: np.ndarray, params: dict, seed: int = None
) -> np.ndarray:
    """Fit ADMM model with given parameters."""
    local_params = params.copy()
    local_params["random_state"] = seed
    model = ADMM(**local_params)
    return model.fit_transform(similarity)


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


def spose_comparison_trial(
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


def low_data_trial(
    triplets, validation_triplets, n_items, admm_params, data_percentage=1.0, seed=0
):
    """Single trial of low data experiment."""
    subsampled_triplets = subsample_triplets(triplets, data_percentage, seed)
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


def spose_comparison_experiment(
    spose_embedding,
    indices_48,
    rsm_48_true,
    similarity,
    validation_triplets,
    admm_params,
    seeds=range(5),
    **kwargs,
):
    """Run SPoSE vs ADMM comparison experiment."""
    param_grid = {
        "spose_embedding": [spose_embedding],
        "indices_48": [indices_48],
        "rsm_48_true": [rsm_48_true],
        "similarity": [similarity],
        "validation_triplets": [validation_triplets],
        "admm_params": [admm_params],
        "seed": seeds,
    }
    return run_experiment(spose_comparison_trial, param_grid, **kwargs)


def low_data_experiment(
    triplets,
    validation_triplets,
    n_items,
    admm_params,
    data_percentages=[0.1, 0.5, 1.0],
    seeds=range(5),
    **kwargs,
):
    """Run low data experiment."""
    param_grid = {
        "triplets": [triplets],
        "validation_triplets": [validation_triplets],
        "n_items": [n_items],
        "admm_params": [admm_params],
        "data_percentage": data_percentages,
        "seed": seeds,
    }
    return run_experiment(low_data_trial, param_grid, **kwargs)


def load_shared_data(
    things_data: Path | str,
    things_images_path: Path,
    max_dim: int,
) -> tuple:
    """Load all shared data for experiments."""

    things_data = Path(things_data)

    spose_embedding = load_spose_embedding(
        things_data / f"spose_embedding_{max_dim}d.txt"
    )
    indices_48 = load_concept_mappings(
        things_data / "words48.csv", things_images_path, CATEGORY_REPLACEMENTS
    )
    rsm_48_true = load_words48(things_data / "rdm48_human.mat")
    return spose_embedding, indices_48, rsm_48_true


def load_triplets(things_data: Path | str) -> np.ndarray:
    train_triplets = np.loadtxt(things_data / "triplets" / "trainset.txt").astype(int)
    validation_triplets = np.loadtxt(
        things_data / "triplets" / "validationset.txt"
    ).astype(int)
    return train_triplets, validation_triplets
