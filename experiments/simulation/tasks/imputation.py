"""Imputation performance evaluation for similarity matrices.

This task evaluates different imputation methods (Mean, Median, KNN, SRF) on
synthetic similarity matrices with missing values at various retention rates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import r2_score

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

from ..lib.plotting import create_imputation_r2_plot

log = logging.getLogger(__name__)


def _generate_dataset(n: int, k: int, kernel: str, seed: int) -> np.ndarray:
    """Generate a single synthetic similarity matrix."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, kernel)
    return similarity


def _mask_similarity(
    matrix: np.ndarray, fraction_retained: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask upper triangle entries in similarity matrix."""
    n = matrix.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    total = triu_i.shape[0]
    n_missing = int((1 - fraction_retained) * total)

    mask = np.zeros_like(matrix, dtype=bool)
    if n_missing > 0:
        choice = rng.choice(total, size=n_missing, replace=False)
        mask[triu_i[choice], triu_j[choice]] = True
        mask[triu_j[choice], triu_i[choice]] = True

    observed = matrix.copy()
    observed[mask] = np.nan
    np.fill_diagonal(observed, matrix.diagonal())
    return observed, mask


def _impute_symmetric(
    imputer, observed: np.ndarray, original: np.ndarray
) -> np.ndarray:
    """Apply imputation and symmetrize result."""
    n = observed.shape[0]

    if isinstance(imputer, SRF):
        imputer.fit(observed)
        imputed = imputer.reconstruct()
    else:
        imputed = imputer.fit_transform(observed)

    # Symmetrize upper triangle
    triu_i, triu_j = np.triu_indices(n, k=1)
    result = imputed.copy()
    result[triu_i, triu_j] = (imputed[triu_i, triu_j] + imputed[triu_j, triu_i]) / 2
    result[triu_j, triu_i] = result[triu_i, triu_j]
    np.fill_diagonal(result, np.diag(original))
    return result


def _evaluate_single_condition(
    original: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    dataset_idx: int,
    fraction_retained: float,
    replicate: int,
    rank: int,
    seed: int,
) -> list[dict]:
    """Evaluate all imputation methods for a single condition."""
    records = []

    # Mean imputation
    mean_imputer = SimpleImputer(strategy="mean")
    mean_filled = _impute_symmetric(mean_imputer, observed, original)
    mse_mean = float(np.mean((mean_filled[mask] - original[mask]) ** 2))
    r2_mean = float(r2_score(original[mask], mean_filled[mask]))
    records.append(
        {
            "dataset": dataset_idx,
            "method": "Mean",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_mean,
            "r2": r2_mean,
        }
    )

    # Median imputation
    median_imputer = SimpleImputer(strategy="median")
    median_filled = _impute_symmetric(median_imputer, observed, original)
    mse_median = float(np.mean((median_filled[mask] - original[mask]) ** 2))
    r2_median = float(r2_score(original[mask], median_filled[mask]))
    records.append(
        {
            "dataset": dataset_idx,
            "method": "Median",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_median,
            "r2": r2_median,
        }
    )

    # KNN imputation
    knn_imputer = KNNImputer()
    knn_filled = _impute_symmetric(knn_imputer, observed, original)
    mse_knn = float(np.mean((knn_filled[mask] - original[mask]) ** 2))
    r2_knn = float(r2_score(original[mask], knn_filled[mask]))
    records.append(
        {
            "dataset": dataset_idx,
            "method": "KNN",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_knn,
            "r2": r2_knn,
        }
    )

    # SRF imputation
    srf_imputer = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=100,
        max_inner=500,
        tol=0.0,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf_filled = _impute_symmetric(srf_imputer, observed, original)
    mse_srf = float(np.mean((srf_filled[mask] - original[mask]) ** 2))
    r2_srf = float(r2_score(original[mask], srf_filled[mask]))
    records.append(
        {
            "dataset": dataset_idx,
            "method": "SRF",
            "fraction_retained": fraction_retained,
            "replicate": replicate,
            "mse": mse_srf,
            "r2": r2_srf,
        }
    )

    return records


def _process_task(
    similarity: np.ndarray,
    dataset_idx: int,
    fraction_retained: float,
    replicate: int,
    rank: int,
    mask_seed: int,
) -> list[dict]:
    """Process a single imputation task."""
    mask_rng = np.random.default_rng(mask_seed + replicate)
    observed, mask = _mask_similarity(similarity, fraction_retained, mask_rng)
    return _evaluate_single_condition(
        similarity,
        observed,
        mask,
        dataset_idx,
        fraction_retained,
        replicate,
        rank,
        mask_seed,
    )


def run(cfg: DictConfig) -> None:
    """Run imputation analysis task."""
    log.info(f"Running imputation analysis: n={cfg.n}, k={cfg.k}")
    log.info(f"Kernel: {cfg.kernel}")
    log.info(f"Fraction retained: {list(cfg.fraction_retained)}")

    # Set up output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Outputs: {output_dir}")

    # Initialize RNG
    rng = np.random.default_rng(cfg.seed)

    # Generate datasets
    log.info(f"Generating {cfg.n_datasets} datasets...")
    dataset_seeds = rng.integers(0, 1_000_000, size=cfg.n_datasets)
    datasets = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_generate_dataset)(cfg.n, cfg.k, cfg.kernel, int(seed))
        for seed in dataset_seeds
    )

    # Build task list
    tasks = []
    mask_seed = int(rng.integers(0, 1_000_000))
    for dataset_idx, similarity in enumerate(datasets):
        for fraction_retained in cfg.fraction_retained:
            for replicate in range(cfg.n_replicates):
                tasks.append(
                    (
                        similarity,
                        dataset_idx,
                        fraction_retained,
                        replicate,
                        cfg.k,
                        mask_seed,
                    )
                )

    # Run imputation tasks
    log.info(f"Running {len(tasks)} imputation tasks...")
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_process_task)(sim, idx, frac, rep, rank, seed)
        for sim, idx, frac, rep, rank, seed in tasks
    )

    # Aggregate results
    records = [record for result in results for record in result]
    df = pd.DataFrame(records)

    # Save results
    results_path = output_dir / "imputation_results.csv"
    df.to_csv(results_path, index=False)
    log.info(f"Saved results to {results_path}")

    # Create plot
    log.info("Creating plot...")
    create_imputation_r2_plot(df, output_dir)

    log.info("Imputation analysis complete")
