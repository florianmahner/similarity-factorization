"""Deep analysis of SRF failure at low retention rates.

Investigates:
1. What happens to the embedding at low retention
2. Whether the bounds constraint is causing issues
3. Whether initialization on imputed data helps
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pysrf import SRF
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def generate_dataset(n: int, k: int, kernel: str, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, kernel)
    return similarity, w


def mask_similarity(
    matrix: np.ndarray, fraction_retained: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
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
    n = observed.shape[0]

    if isinstance(imputer, SRF):
        imputer.fit(observed)
        imputed = imputer.reconstruct()
    else:
        imputed = imputer.fit_transform(observed)

    triu_i, triu_j = np.triu_indices(n, k=1)
    result = imputed.copy()
    result[triu_i, triu_j] = (imputed[triu_i, triu_j] + imputed[triu_j, triu_i]) / 2
    result[triu_j, triu_i] = result[triu_i, triu_j]
    np.fill_diagonal(result, np.diag(original))
    return result


def analyze_convergence(similarity, observed, mask, rank, seed, fraction_retained):
    """Track SRF convergence behavior."""
    log.info(f"\n{'='*60}")
    log.info(f"Analyzing convergence at {fraction_retained:.0%} retention")

    # Run SRF with verbose to see what's happening
    srf = SRF(
        rank=rank,
        random_state=seed,
        max_outer=50,
        max_inner=500,
        tol=0.0,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=np.nan,
    )
    srf.fit(observed)

    # Get history
    history = srf.history_

    log.info(f"MSE trajectory: {history['mse'][:5]}...{history['mse'][-5:]}")
    log.info(f"Quality (R²) trajectory: {history['quality'][:5]}...{history['quality'][-5:]}")
    log.info(f"Primal residual: {history['primal_residual'][-1]:.4f}")
    log.info(f"Dual residual: {history['dual_residual'][-1]:.4f}")

    # Check embedding
    w = srf.w_
    log.info(f"\nEmbedding stats:")
    log.info(f"  Shape: {w.shape}")
    log.info(f"  Range: [{w.min():.4f}, {w.max():.4f}]")
    log.info(f"  Mean: {w.mean():.4f}")
    log.info(f"  Sparsity (zeros): {(w < 1e-6).sum() / w.size:.2%}")

    # Check if embedding is degenerate
    col_norms = np.linalg.norm(w, axis=0)
    log.info(f"  Column norms: {col_norms.round(2)}")

    return history


def test_pre_imputation_initialization(similarity, observed, mask, rank, seed):
    """Test if pre-imputing with mean/median helps SRF."""
    log.info("\n--- Testing Pre-Imputation Initialization ---")

    # Strategy 1: Fill NaN with mean, then run SRF without missing values
    mean_imputer = SimpleImputer(strategy="mean")
    pre_filled = mean_imputer.fit_transform(observed)
    pre_filled = 0.5 * (pre_filled + pre_filled.T)

    # SRF on pre-filled matrix (no missing values)
    srf = SRF(
        rank=rank,
        random_state=seed,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        missing_values=None,  # No missing values
    )
    srf.fit(pre_filled)
    recon = srf.reconstruct()
    r2 = r2_score(similarity[mask], recon[mask])
    log.info(f"SRF on mean-prefilled (no missing): R² = {r2:.4f}")

    # Strategy 2: Higher rank
    log.info("\n--- Testing Higher Rank ---")
    for test_rank in [5, 10, 20, 50]:
        srf_hr = SRF(
            rank=test_rank,
            random_state=seed,
            max_outer=100,
            max_inner=500,
            tol=1e-4,
            init="random_sqrt",
            verbose=0,
            bounds=(0.0, 1.0),
            missing_values=np.nan,
        )
        srf_hr.fit(observed)
        recon_hr = srf_hr.reconstruct()
        r2_hr = r2_score(similarity[mask], recon_hr[mask])
        log.info(f"SRF (rank={test_rank}): R² = {r2_hr:.4f}")

    # Strategy 3: No bounds
    log.info("\n--- Testing Without Bounds ---")
    srf_nb = SRF(
        rank=rank,
        random_state=seed,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=None,
        missing_values=np.nan,
    )
    srf_nb.fit(observed)
    recon_nb = srf_nb.reconstruct()
    r2_nb = r2_score(similarity[mask], recon_nb[mask])
    log.info(f"SRF (no bounds): R² = {r2_nb:.4f}")
    log.info(f"  Reconstruction range: [{recon_nb.min():.4f}, {recon_nb.max():.4f}]")


def test_alternative_approaches(similarity, observed, mask, rank, seed):
    """Test alternative approaches that might work better at low retention."""
    log.info("\n--- Testing Alternative Approaches ---")

    # Approach 1: Two-stage - Mean fill then SRF refine
    log.info("\n1. Two-stage: Mean fill -> SRF refinement")
    mean_imputer = SimpleImputer(strategy="mean")
    stage1 = mean_imputer.fit_transform(observed)
    stage1 = 0.5 * (stage1 + stage1.T)

    srf_stage2 = SRF(
        rank=rank,
        random_state=seed,
        max_outer=100,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
    )
    srf_stage2.fit(stage1)
    recon_2stage = srf_stage2.reconstruct()
    r2_2stage = r2_score(similarity[mask], recon_2stage[mask])
    log.info(f"Two-stage R²: {r2_2stage:.4f}")

    # Approach 2: Lower rho (weaker constraint enforcement)
    log.info("\n2. Very low rho (weaker ADMM penalty)")
    srf_lowrho = SRF(
        rank=rank,
        random_state=seed,
        max_outer=200,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        rho=0.1,
        missing_values=np.nan,
    )
    srf_lowrho.fit(observed)
    recon_lowrho = srf_lowrho.reconstruct()
    r2_lowrho = r2_score(similarity[mask], recon_lowrho[mask])
    log.info(f"SRF (rho=0.1): R² = {r2_lowrho:.4f}")

    # Approach 3: Much higher rank with regularization (let model find structure)
    log.info("\n3. Overparameterized rank (rank=50) with low rho")
    srf_over = SRF(
        rank=50,
        random_state=seed,
        max_outer=200,
        max_inner=500,
        tol=1e-4,
        init="random_sqrt",
        verbose=0,
        bounds=(0.0, 1.0),
        rho=0.5,
        missing_values=np.nan,
    )
    srf_over.fit(observed)
    recon_over = srf_over.reconstruct()
    r2_over = r2_score(similarity[mask], recon_over[mask])
    log.info(f"SRF (rank=50, rho=0.5): R² = {r2_over:.4f}")

    return {
        "two_stage": r2_2stage,
        "low_rho": r2_lowrho,
        "overparameterized": r2_over,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42

    log.info("=" * 60)
    log.info("DEEP ANALYSIS: SRF failure at low retention")
    log.info("=" * 60)

    rng = np.random.default_rng(seed)
    similarity, w_true = generate_dataset(n, k, kernel, seed)

    # Test at problematic retention rates
    for frac in [0.05, 0.10, 0.20]:
        mask_rng = np.random.default_rng(123)
        observed, mask = mask_similarity(similarity, frac, mask_rng)

        log.info(f"\n\n{'#'*60}")
        log.info(f"FRACTION RETAINED: {frac:.0%}")
        log.info(f"{'#'*60}")

        # Baseline scores
        median_imputer = SimpleImputer(strategy="median")
        median_filled = impute_symmetric(median_imputer, observed, similarity)
        r2_median = r2_score(similarity[mask], median_filled[mask])

        knn_imputer = KNNImputer()
        knn_filled = impute_symmetric(knn_imputer, observed, similarity)
        r2_knn = r2_score(similarity[mask], knn_filled[mask])

        log.info(f"\nBaselines: Median R²={r2_median:.4f}, KNN R²={r2_knn:.4f}")

        # Analyze convergence
        analyze_convergence(similarity, observed, mask, k, seed, frac)

        # Test pre-imputation
        test_pre_imputation_initialization(similarity, observed, mask, k, seed)

        # Test alternatives
        alt_results = test_alternative_approaches(similarity, observed, mask, k, seed)

    log.info("\n\nAnalysis complete!")


if __name__ == "__main__":
    main()
