"""RBF bandwidth selection via harmonic mean of factorization stability and explained variance.

Given feature matrix X, selects the optimal bandwidth multiplier alpha* for
    sigma = alpha * median(pairwise_distances(X))
by estimating rank via kappa at each bandwidth, running multiple SRF fits,
and selecting the alpha that maximizes H(stability, R^2).

The harmonic mean naturally balances the trade-off: stability decreases monotonically
with alpha (sharper kernels are easier to factorize stably) while explained variance
increases (smoother kernels are better approximated by low-rank reconstructions).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, pairwise_kernels

from pysrf import SRF
from src.coherence import (
    _estimate_kappa_hat,
    _smooth_median,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)

log = logging.getLogger(__name__)

DEFAULT_ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def select_rbf_bandwidth(
    features: np.ndarray,
    alpha_grid: list[float] | None = None,
    n_runs: int = 5,
    k_max: int = 100,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    """Select optimal RBF bandwidth multiplier for SRF factorization.

    For each alpha in alpha_grid:
      1. Build S = RBF(features, sigma = alpha * median_dist)
      2. Estimate rank k* via kappa changepoint
      3. Run n_runs SRF fits at rank k*
      4. Compute stability (mean pairwise aligned correlation) and R^2

    Returns the alpha that maximizes H(stability, R^2).

    Parameters
    ----------
    features : (n, d) array
        Feature matrix (e.g. fMRI betas, DNN activations).
    alpha_grid : list of float, optional
        Bandwidth multipliers to evaluate. Default [0.2, 0.4, 0.6, 0.8, 1.0].
    n_runs : int
        Number of SRF fits per bandwidth for stability assessment.
    k_max : int
        Maximum rank for kappa estimation.
    n_jobs : int
        Parallelism for SRF fits and coherence computation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        alpha_star : float
        k_star : int
        sigma_star : float
        h_star : float
        results : DataFrame with full results per alpha
        median_dist : float
    """
    if alpha_grid is None:
        alpha_grid = list(DEFAULT_ALPHA_GRID)

    log.info(f"Computing pairwise distances for {features.shape[0]} samples...")
    dist = pairwise_distances(features, metric="euclidean")
    median_dist = float(np.median(dist[np.triu_indices(len(dist), k=1)]))
    log.info(f"Median pairwise distance: {median_dist:.2f}")

    records = []
    for alpha in alpha_grid:
        sigma = alpha * median_dist
        gamma = 1.0 / (2.0 * sigma ** 2)
        s = pairwise_kernels(features, metric="rbf", gamma=gamma)

        log.info(f"alpha={alpha}, sigma={sigma:.2f}, "
                 f"sim_mean={s.mean():.4f}, sim_min={s.min():.4f}")

        rank = estimate_rank_kappa(s, k_max=k_max, n_jobs=n_jobs,
                                   random_state=random_state)
        log.info(f"  kappa rank: k*={rank}")

        quality = factorization_quality(s, rank, n_runs=n_runs, n_jobs=n_jobs,
                                        random_state=random_state)
        log.info(f"  stability={quality['stability_mean']:.3f}, "
                 f"R2={quality['r2_mean']:.3f}, H={quality['h_mean']:.3f}")

        records.append({
            "alpha": alpha,
            "sigma": sigma,
            "k_star": rank,
            "sim_mean": float(s.mean()),
            "sim_min": float(s.min()),
            **quality,
        })

    df = pd.DataFrame(records)
    best_idx = int(df["h_mean"].idxmax())

    result = {
        "alpha_star": float(df.loc[best_idx, "alpha"]),
        "k_star": int(df.loc[best_idx, "k_star"]),
        "sigma_star": float(df.loc[best_idx, "sigma"]),
        "h_star": float(df.loc[best_idx, "h_mean"]),
        "results": df,
        "median_dist": median_dist,
    }

    log.info(f"Selected: alpha*={result['alpha_star']}, k*={result['k_star']}, "
             f"H={result['h_star']:.3f}")
    return result


def estimate_rank_kappa(
    s: np.ndarray,
    k_max: int = 100,
    n_p: int = 20,
    B: int = 20,
    B_null: int = 15,
    hi_band_quantile: float = 0.85,
    smooth_window: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> int:
    """Estimate factorization rank via kappa changepoint on eigenspace coherence.

    Uses reduced bootstrap parameters for speed (suitable for bandwidth selection
    where approximate rank is sufficient).

    Parameters
    ----------
    s : (n, n) symmetric similarity matrix
    k_max : int
        Maximum rank to consider.
    n_p : int
        Number of masking rates.
    B, B_null : int
        Bootstrap replicates for coherence and null distribution.
    hi_band_quantile : float
        Quantile for high-p band in kappa estimation.
    smooth_window : int
        Median filter window for kappa smoothing.
    n_jobs : int
        Parallelism for eigendecompositions.
    random_state : int
        Random seed.

    Returns
    -------
    int : estimated rank k*
    """
    n = s.shape[0]
    effective_k_max = min(k_max, n - 1)
    k_list = list(range(1, effective_k_max + 1))
    p_list = np.linspace(0.05, 0.95, n_p)

    n_cpus = n_jobs if n_jobs > 0 else os.cpu_count() - 1

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list=k_list,
        p_list=p_list,
        B=B,
        random_state=random_state,
        compute_null=True,
        B_null=B_null,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=n_cpus,
        show_progress=False,
        visualize=False,
    )

    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_arr = result["k_list"]
    p_arr = result["p"]

    kappa_hat, _ = _estimate_kappa_hat(x_median, p_arr, hi_band_quantile)
    k_star, _ = kappa_changepoint(kappa_hat, k_arr, smooth_window=smooth_window)
    return int(k_star)


def factorization_quality(
    s: np.ndarray,
    rank: int,
    n_runs: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    """Assess SRF factorization quality: stability, explained variance, harmonic mean.

    Fits n_runs independent SRF models at the given rank. Stability is the mean
    pairwise correlation of Hungarian-aligned embedding columns (Fisher-z averaged).
    Explained variance is the mean R^2 of WW^T reconstructions (upper triangle).
    The harmonic mean H = 2*stab*R^2/(stab+R^2) balances both.

    Parameters
    ----------
    s : (n, n) symmetric similarity matrix
    rank : int
        Factorization rank.
    n_runs : int
        Number of independent SRF fits.
    n_jobs : int
        Parallelism for SRF fits.
    random_state : int
        Base random seed (each run uses random_state + i).

    Returns
    -------
    dict with keys: stability_mean, stability_min, reliability_per_dim,
                    r2_mean, r2_std, h_mean
    """
    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(_fit_srf)(s, rank, random_state + i) for i in range(n_runs)
    )

    # Explained variance: R^2 on upper triangle (avoids diagonal and double-counting)
    triu_idx = np.triu_indices(s.shape[0], k=1)
    s_upper = s[triu_idx]
    s_mean = s_upper.mean()
    ss_tot = float(np.sum((s_upper - s_mean) ** 2))

    r2_values = np.zeros(n_runs)
    for i, w in enumerate(embeddings):
        recon = w @ w.T
        ss_res = float(np.sum((s_upper - recon[triu_idx]) ** 2))
        r2_values[i] = 1.0 - ss_res / ss_tot

    r2_mean = float(np.mean(r2_values))

    # Stability: pairwise inter-run correlations via Hungarian alignment
    all_per_dim = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            per_dim = _align_and_correlate(embeddings[i], embeddings[j])
            all_per_dim.append(per_dim)

    all_per_dim = np.array(all_per_dim)  # (n_pairs, rank)

    z_scores = np.arctanh(np.clip(all_per_dim, -0.999, 0.999))
    mean_z = np.mean(z_scores, axis=0)
    reliability_per_dim = np.tanh(mean_z)
    stability_mean = float(np.mean(reliability_per_dim))

    # Harmonic mean (clamp R^2 at 0 -- negative R^2 means worse than mean predictor)
    r2_clamped = max(r2_mean, 0.0)
    if stability_mean > 0 and r2_clamped > 0:
        h_mean = float(2.0 * stability_mean * r2_clamped / (stability_mean + r2_clamped))
    else:
        h_mean = 0.0

    return {
        "stability_mean": stability_mean,
        "stability_min": float(np.min(reliability_per_dim)),
        "reliability_per_dim": reliability_per_dim,
        "r2_mean": r2_mean,
        "r2_std": float(np.std(r2_values)),
        "h_mean": h_mean,
    }


def _fit_srf(s: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Fit a single SRF model and return the embedding W."""
    model = SRF(rank=rank, random_state=seed)
    model.fit(s)
    return model.w_


def _align_and_correlate(w_a: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """Align two embeddings via Hungarian matching on absolute column correlations.

    Returns per-dimension absolute correlation after optimal assignment.
    """
    k = w_a.shape[1]
    # Vectorized correlation matrix
    a_centered = w_a - w_a.mean(axis=0, keepdims=True)
    b_centered = w_b - w_b.mean(axis=0, keepdims=True)
    a_norm = np.sqrt(np.sum(a_centered ** 2, axis=0, keepdims=True))
    b_norm = np.sqrt(np.sum(b_centered ** 2, axis=0, keepdims=True))
    a_normed = a_centered / np.maximum(a_norm, 1e-12)
    b_normed = b_centered / np.maximum(b_norm, 1e-12)
    corr_matrix = a_normed.T @ b_normed  # (k, k)

    row_idx, col_idx = linear_sum_assignment(-np.abs(corr_matrix))

    per_dim_corr = np.abs(corr_matrix[row_idx, col_idx])
    return per_dim_corr
