"""Debug imputation performance at low retention rates.

This script investigates why SRF collapses at very low retention percentages
in the simulation imputation experiment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from pysrf import SRF
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import r2_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def generate_dataset(n: int, k: int, kernel: str, seed: int) -> np.ndarray:
    """Generate a single synthetic similarity matrix."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, kernel)
    return similarity, w


def mask_similarity(
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


def impute_symmetric(imputer, observed: np.ndarray, original: np.ndarray) -> np.ndarray:
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


def diagnose_single_case(
    original: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int,
    seed: int,
    fraction_retained: float,
) -> dict:
    """Diagnose imputation performance for a single case."""
    log.info(f"\n{'='*60}")
    log.info(f"Fraction retained: {fraction_retained:.2%}")
    log.info(f"Total entries (upper triangle): {mask.shape[0] * (mask.shape[0] - 1) // 2}")
    log.info(f"Missing entries: {mask.sum() // 2}")
    log.info(f"Observed entries: {(~mask).sum() // 2 - mask.shape[0]}")

    # Compute observed data statistics
    obs_values = original[~mask & ~np.eye(mask.shape[0], dtype=bool)]
    log.info(f"Observed value range: [{obs_values.min():.4f}, {obs_values.max():.4f}]")
    log.info(f"Observed value mean: {obs_values.mean():.4f}")

    results = {"fraction_retained": fraction_retained}

    # Mean imputation baseline
    mean_imputer = SimpleImputer(strategy="mean")
    mean_filled = impute_symmetric(mean_imputer, observed, original)
    r2_mean = float(r2_score(original[mask], mean_filled[mask]))
    results["r2_mean"] = r2_mean
    log.info(f"Mean imputation R²: {r2_mean:.4f}")

    # Median imputation baseline
    median_imputer = SimpleImputer(strategy="median")
    median_filled = impute_symmetric(median_imputer, observed, original)
    r2_median = float(r2_score(original[mask], median_filled[mask]))
    results["r2_median"] = r2_median
    log.info(f"Median imputation R²: {r2_median:.4f}")

    # KNN imputation baseline
    knn_imputer = KNNImputer()
    knn_filled = impute_symmetric(knn_imputer, observed, original)
    r2_knn = float(r2_score(original[mask], knn_filled[mask]))
    results["r2_knn"] = r2_knn
    log.info(f"KNN imputation R²: {r2_knn:.4f}")

    # SRF with original settings (potential problem)
    log.info("\n--- SRF Original Settings ---")
    srf_original = SRF(
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
    srf_original_filled = impute_symmetric(srf_original, observed, original)
    r2_srf_original = float(r2_score(original[mask], srf_original_filled[mask]))
    results["r2_srf_original"] = r2_srf_original
    log.info(f"SRF (original) R²: {r2_srf_original:.4f}")
    log.info(f"  Iterations: {srf_original.n_iter_}")
    log.info(f"  Final MSE: {srf_original.history_['mse'][-1]:.6f}")

    # Examine reconstruction
    recon_range = srf_original_filled.min(), srf_original_filled.max()
    log.info(f"  Reconstruction range: [{recon_range[0]:.4f}, {recon_range[1]:.4f}]")

    # Test 1: Higher tolerance (allow early convergence)
    log.info("\n--- SRF with tol=1e-4 ---")
    srf_tol = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf_tol_filled = impute_symmetric(srf_tol, observed, original)
    r2_srf_tol = float(r2_score(original[mask], srf_tol_filled[mask]))
    results["r2_srf_tol"] = r2_srf_tol
    log.info(f"SRF (tol=1e-4) R²: {r2_srf_tol:.4f}")
    log.info(f"  Iterations: {srf_tol.n_iter_}")

    # Test 2: Different rho values
    for rho in [1.0, 3.0, 10.0]:
        log.info(f"\n--- SRF with rho={rho} ---")
        srf_rho = SRF(
            rank=rank,
            random_state=seed + 5,
            max_outer=100,
            max_inner=500,
            tol=1e-4,
            init="random_sqrt",
            verbose=0,
            bounds=(0.0, 1.0),
            missing_values=np.nan,
            rho=rho,
        )
        srf_rho_filled = impute_symmetric(srf_rho, observed, original)
        r2_srf_rho = float(r2_score(original[mask], srf_rho_filled[mask]))
        results[f"r2_srf_rho{rho}"] = r2_srf_rho
        log.info(f"SRF (rho={rho}) R²: {r2_srf_rho:.4f}")
        log.info(f"  Iterations: {srf_rho.n_iter_}")

    # Test 3: Different initialization
    log.info("\n--- SRF with random init ---")
    srf_init = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf_init_filled = impute_symmetric(srf_init, observed, original)
    r2_srf_init = float(r2_score(original[mask], srf_init_filled[mask]))
    results["r2_srf_init_random"] = r2_srf_init
    log.info(f"SRF (init=random) R²: {r2_srf_init:.4f}")
    log.info(f"  Iterations: {srf_init.n_iter_}")

    # Test 4: More inner iterations at low retention
    log.info("\n--- SRF with more inner iterations ---")
    srf_inner = SRF(
        rank=rank,
        random_state=seed + 5,
        max_outer=200,
        max_inner=1000,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf_inner_filled = impute_symmetric(srf_inner, observed, original)
    r2_srf_inner = float(r2_score(original[mask], srf_inner_filled[mask]))
    results["r2_srf_inner"] = r2_srf_inner
    log.info(f"SRF (more iters) R²: {r2_srf_inner:.4f}")
    log.info(f"  Iterations: {srf_inner.n_iter_}")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration matching the experiment
    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42

    log.info("=" * 60)
    log.info("IMPUTATION DEBUG: Investigating SRF collapse at low retention")
    log.info("=" * 60)

    # Generate dataset
    rng = np.random.default_rng(seed)
    log.info(f"\nGenerating dataset: n={n}, k={k}, kernel={kernel}")
    similarity, w_true = generate_dataset(n, k, kernel, seed)

    log.info(f"Similarity matrix range: [{similarity.min():.4f}, {similarity.max():.4f}]")
    log.info(f"Similarity matrix mean: {similarity.mean():.4f}")

    # Check the true rank structure
    U, S, Vt = np.linalg.svd(similarity)
    log.info(f"Top 10 singular values: {S[:10].round(2)}")
    log.info(f"Explained variance by k={k}: {S[:k].sum() / S.sum():.2%}")

    # Test different retention fractions
    fractions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    all_results = []

    mask_seed = int(rng.integers(0, 1_000_000))

    for frac in fractions:
        mask_rng = np.random.default_rng(mask_seed)
        observed, mask = mask_similarity(similarity, frac, mask_rng)
        results = diagnose_single_case(similarity, observed, mask, k, seed, frac)
        all_results.append(results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "diagnostic_results.csv", index=False)
    log.info(f"\n\nResults saved to {OUTPUT_DIR / 'diagnostic_results.csv'}")

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
