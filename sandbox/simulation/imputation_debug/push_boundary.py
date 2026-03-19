"""Push the boundary: Can we make SRF work at 5% retention?

Try various strategies:
1. Very low rho with more iterations
2. Adaptive rho scheduling
3. Different initialization strategies
4. Rank reduction
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from pysrf import SRF
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tools.metrics import compute_similarity
from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def generate_dataset(n: int, k: int, kernel: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, rng=rng, alpha=1.0)
    similarity = compute_similarity(w, w, kernel)
    return similarity


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


def impute_and_score(srf, observed, similarity, mask):
    """Fit SRF and return R² score."""
    srf.fit(observed)
    recon = srf.reconstruct()

    # Symmetrize
    n = recon.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    recon[triu_i, triu_j] = (recon[triu_i, triu_j] + recon[triu_j, triu_i]) / 2
    recon[triu_j, triu_i] = recon[triu_i, triu_j]
    np.fill_diagonal(recon, np.diag(similarity))

    return r2_score(similarity[mask], recon[mask])


def test_strategies(similarity, observed, mask, true_rank, seed):
    """Test various strategies to improve low-retention performance."""
    results = []

    # Baseline: Median
    median_imp = SimpleImputer(strategy="median")
    median_filled = median_imp.fit_transform(observed)
    median_filled = 0.5 * (median_filled + median_filled.T)
    r2_median = r2_score(similarity[mask], median_filled[mask])
    results.append({"strategy": "Median (baseline)", "r2": r2_median})
    log.info(f"Median (baseline): R² = {r2_median:.4f}")

    # Strategy 1: Default SRF
    log.info("\n--- Strategy 1: Default SRF (rho=3.0) ---")
    srf_default = SRF(
        rank=true_rank, random_state=seed, max_outer=100, max_inner=500,
        tol=0.0, rho=3.0, bounds=(0.0, 1.0), missing_values=np.nan
    )
    r2 = impute_and_score(srf_default, observed, similarity, mask)
    results.append({"strategy": "SRF default (rho=3.0)", "r2": r2})
    log.info(f"R² = {r2:.4f}")

    # Strategy 2: Very low rho
    log.info("\n--- Strategy 2: Very low rho ---")
    for rho in [0.001, 0.005, 0.01, 0.05]:
        srf = SRF(
            rank=true_rank, random_state=seed, max_outer=200, max_inner=500,
            tol=1e-5, rho=rho, bounds=(0.0, 1.0), missing_values=np.nan
        )
        r2 = impute_and_score(srf, observed, similarity, mask)
        results.append({"strategy": f"SRF rho={rho}", "r2": r2})
        log.info(f"rho={rho}: R² = {r2:.4f}, iters={srf.n_iter_}")

    # Strategy 3: Lower rank (underfitting to avoid noise)
    log.info("\n--- Strategy 3: Lower rank ---")
    for rank in [2, 3, 4]:
        srf = SRF(
            rank=rank, random_state=seed, max_outer=200, max_inner=500,
            tol=1e-4, rho=0.01, bounds=(0.0, 1.0), missing_values=np.nan
        )
        r2 = impute_and_score(srf, observed, similarity, mask)
        results.append({"strategy": f"SRF rank={rank}", "r2": r2})
        log.info(f"rank={rank}: R² = {r2:.4f}")

    # Strategy 4: More iterations with very low rho
    log.info("\n--- Strategy 4: Many iterations + low rho ---")
    srf = SRF(
        rank=true_rank, random_state=seed, max_outer=500, max_inner=1000,
        tol=1e-6, rho=0.01, bounds=(0.0, 1.0), missing_values=np.nan
    )
    r2 = impute_and_score(srf, observed, similarity, mask)
    results.append({"strategy": "SRF 500 iters, rho=0.01", "r2": r2})
    log.info(f"R² = {r2:.4f}, iters={srf.n_iter_}")

    # Strategy 5: No bounds constraint
    log.info("\n--- Strategy 5: No bounds ---")
    srf = SRF(
        rank=true_rank, random_state=seed, max_outer=200, max_inner=500,
        tol=1e-4, rho=0.01, bounds=None, missing_values=np.nan
    )
    r2 = impute_and_score(srf, observed, similarity, mask)
    results.append({"strategy": "SRF no bounds", "r2": r2})
    log.info(f"R² = {r2:.4f}")

    # Strategy 6: Two-stage with warm start from median
    log.info("\n--- Strategy 6: Warm start from median ---")
    # Pre-fill with median
    prefilled = median_imp.fit_transform(observed)
    prefilled = 0.5 * (prefilled + prefilled.T)

    # Run SRF on prefilled (no missing)
    srf = SRF(
        rank=true_rank, random_state=seed, max_outer=200, max_inner=500,
        tol=1e-4, rho=0.1, bounds=(0.0, 1.0), missing_values=None
    )
    srf.fit(prefilled)
    recon = srf.reconstruct()
    np.fill_diagonal(recon, np.diag(similarity))
    r2 = r2_score(similarity[mask], recon[mask])
    results.append({"strategy": "Two-stage (median+SRF)", "r2": r2})
    log.info(f"R² = {r2:.4f}")

    # Strategy 7: Random restarts
    log.info("\n--- Strategy 7: Best of 5 random restarts ---")
    best_r2 = -np.inf
    for restart in range(5):
        srf = SRF(
            rank=true_rank, random_state=seed + restart * 100,
            max_outer=200, max_inner=500, tol=1e-4, rho=0.01,
            bounds=(0.0, 1.0), missing_values=np.nan
        )
        r2 = impute_and_score(srf, observed, similarity, mask)
        if r2 > best_r2:
            best_r2 = r2
    results.append({"strategy": "SRF best of 5 restarts", "r2": best_r2})
    log.info(f"Best R² = {best_r2:.4f}")

    return pd.DataFrame(results)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = 200
    k = 5
    kernel = "gaussian_kernel"
    seed = 42

    log.info("=" * 60)
    log.info("PUSHING THE BOUNDARY: Can SRF work at very low retention?")
    log.info("=" * 60)

    # Generate dataset
    similarity = generate_dataset(n, k, kernel, seed)

    # Test at different challenging retention rates
    all_results = []

    for frac in [0.05, 0.07, 0.10]:
        log.info(f"\n{'#'*60}")
        log.info(f"RETENTION: {frac:.0%}")
        log.info(f"{'#'*60}")

        mask_rng = np.random.default_rng(123)
        observed, mask = mask_similarity(similarity, frac, mask_rng)

        n_observed = (~np.isnan(observed)).sum() // 2 - n  # exclude diagonal
        log.info(f"Observed entries: {n_observed}")

        df = test_strategies(similarity, observed, mask, k, seed)
        df["fraction_retained"] = frac
        all_results.append(df)

    # Combine and save
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(OUTPUT_DIR / "push_boundary_results.csv", index=False)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY: Best strategies by retention rate")
    log.info("=" * 60)

    for frac in [0.05, 0.07, 0.10]:
        frac_df = results_df[results_df["fraction_retained"] == frac]
        best = frac_df.loc[frac_df["r2"].idxmax()]
        median_r2 = frac_df[frac_df["strategy"] == "Median (baseline)"]["r2"].values[0]
        log.info(f"\n{frac:.0%} retention:")
        log.info(f"  Median baseline: {median_r2:.4f}")
        log.info(f"  Best strategy: {best['strategy']} (R² = {best['r2']:.4f})")
        log.info(f"  Improvement: {best['r2'] - median_r2:+.4f}")


if __name__ == "__main__":
    main()
