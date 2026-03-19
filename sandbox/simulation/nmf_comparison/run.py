"""Compare SRF vs NMF on different similarity kernels.

Tests whether SRF better recovers ground truth factors when the similarity
matrix is computed with different kernels (linear, RBF with median heuristic).

Usage:
    poetry run python sandbox/nmf_comparison/run.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

from utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def _normalize_factors(factors: np.ndarray) -> np.ndarray:
    """Normalize factor matrix so rows sum to 1."""
    return factors / (factors.sum(axis=1, keepdims=True) + 1e-10)


def _align_factors(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target factors to reference using Hungarian algorithm."""
    ref_norm = _normalize_factors(reference)
    target_norm = _normalize_factors(target)
    similarity = ref_norm.T @ target_norm
    _, col_ind = linear_sum_assignment(-similarity)
    return target[:, col_ind]


def _factor_correlation(true_factors: np.ndarray, learned: np.ndarray) -> float:
    """Compute mean correlation between aligned factors."""
    aligned = _align_factors(true_factors, learned)
    correlations = []
    for i in range(true_factors.shape[1]):
        corr = np.corrcoef(true_factors[:, i], aligned[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


def _rsm_correlation(true_rsm: np.ndarray, recon: np.ndarray) -> float:
    """Compute correlation between true RSM and reconstructed RSM."""
    n = true_rsm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    true_flat = true_rsm[triu_idx]
    recon_flat = recon[triu_idx]
    return float(np.corrcoef(true_flat, recon_flat)[0, 1])


def _symmetry_error(matrix: np.ndarray) -> float:
    """Compute symmetry error: ||S - S.T|| / ||S||."""
    diff = matrix - matrix.T
    return float(np.linalg.norm(diff) / (np.linalg.norm(matrix) + 1e-10))


def _compute_rbf_similarity(factors: np.ndarray) -> np.ndarray:
    """Compute RBF kernel similarity with median heuristic."""
    dist = pairwise_distances(factors, metric="euclidean")
    median_dist = np.median(dist[dist > 0])
    gamma = 1 / (2 * median_dist**2)
    return rbf_kernel(factors, gamma=gamma)


def _run_single(
    n: int, k: int, alpha: float, kernel: str, seed: int
) -> dict:
    """Run single comparison between SRF and NMF."""
    rng = np.random.default_rng(seed)

    # Generate ground truth factors
    true_factors = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)

    # Compute similarity matrix with specified kernel
    if kernel == "linear":
        similarity = true_factors @ true_factors.T
    elif kernel == "rbf":
        similarity = _compute_rbf_similarity(true_factors)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Ensure symmetric and non-negative
    similarity = (similarity + similarity.T) / 2
    similarity = np.clip(similarity, 0, None)

    # === SRF ===
    srf = SRF(rank=k, random_state=seed, max_outer=200, tol=1e-6)
    srf.fit(similarity)
    srf_recon = srf.reconstruct()
    srf_factors = srf.w_

    # === NMF ===
    nmf = NMF(n_components=k, random_state=seed, max_iter=500, tol=1e-6)
    nmf_W = nmf.fit_transform(similarity)
    nmf_H = nmf.components_
    nmf_recon = nmf_W @ nmf_H
    nmf_recon_sym = (nmf_recon + nmf_recon.T) / 2

    result = {
        "n": n,
        "k": k,
        "alpha": alpha,
        "kernel": kernel,
        "seed": seed,
        # RSM recovery (correlation with input similarity)
        "srf_rsm_corr": _rsm_correlation(similarity, srf_recon),
        "nmf_rsm_corr": _rsm_correlation(similarity, nmf_recon_sym),
        # Factor recovery (correlation with true factors)
        "srf_factor_corr": _factor_correlation(true_factors, srf_factors),
        "nmf_factor_corr": _factor_correlation(true_factors, nmf_W),
        # Symmetry error
        "srf_symmetry_error": _symmetry_error(srf_recon),
        "nmf_symmetry_error": _symmetry_error(nmf_recon),
    }

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters
    n = 64
    k = 8
    alphas = [0.3, 0.5, 1.0, 2.0]
    kernels = ["linear", "rbf"]
    n_replicates = 30

    log.info(f"Running SRF vs NMF kernel comparison: n={n}, k={k}")
    log.info(f"Alphas: {alphas}")
    log.info(f"Kernels: {kernels}")
    log.info(f"Replicates: {n_replicates}")

    # Generate tasks
    tasks = []
    for alpha in alphas:
        for kernel in kernels:
            for rep in range(n_replicates):
                seed = hash((alpha, kernel, rep)) % (2**31)
                tasks.append((n, k, alpha, kernel, seed))

    log.info(f"Running {len(tasks)} comparisons...")
    results = Parallel(n_jobs=-1)(
        delayed(_run_single)(n, k, alpha, kernel, seed)
        for n, k, alpha, kernel, seed in tasks
    )

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved to {csv_path}")

    # Print summary by kernel
    log.info("\n=== Summary by kernel ===")
    for kernel in kernels:
        df_k = df[df["kernel"] == kernel]
        srf_factor = df_k["srf_factor_corr"].mean()
        nmf_factor = df_k["nmf_factor_corr"].mean()
        srf_rsm = df_k["srf_rsm_corr"].mean()
        nmf_rsm = df_k["nmf_rsm_corr"].mean()
        winner_factor = "SRF" if srf_factor > nmf_factor else "NMF"
        winner_rsm = "SRF" if srf_rsm > nmf_rsm else "NMF"
        log.info(f"{kernel.upper()}:")
        log.info(f"  Factor recovery: SRF={srf_factor:.3f}, NMF={nmf_factor:.3f} ({winner_factor})")
        log.info(f"  RSM recovery:    SRF={srf_rsm:.3f}, NMF={nmf_rsm:.3f} ({winner_rsm})")


if __name__ == "__main__":
    main()
