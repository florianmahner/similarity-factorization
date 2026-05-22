"""Does kappa k* change with RBF bandwidth?

Sweep sigma multipliers on VGG16 features and run kappa at each.
If k* is stable across bandwidths, we can decouple rank estimation
from bandwidth selection. If not, we have a circularity problem.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

from src.coherence import (
    _estimate_kappa_hat,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

SIGMA_MULTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
K_MAX = 100
B = 30
B_NULL = 20
N_P = 20


def _build_rbf(features, sigma_mult):
    dist = pairwise_distances(features, metric="euclidean")
    median_dist = np.median(dist[np.triu_indices(len(dist), k=1)])
    sigma = median_dist * sigma_mult
    gamma = 1 / (2 * sigma**2)
    s = pairwise_kernels(features, metric="rbf", gamma=gamma)
    return s, sigma, median_dist


def _run_kappa(similarity, k_max=K_MAX, n_jobs=16):
    n = similarity.shape[0]
    k_list = list(range(1, min(k_max, n - 1) + 1))
    p_list = np.linspace(0.05, 0.95, N_P)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        similarity,
        k_list=k_list,
        p_list=p_list,
        B=B,
        random_state=42,
        compute_null=True,
        B_null=B_NULL,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=n_jobs,
        show_progress=False,
        visualize=False,
    )

    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_arr = result["k_list"]
    p_arr = result["p"]

    kappa_hat, kappa_info = _estimate_kappa_hat(x_median, p_arr, hi_band_quantile=0.85)
    k_star, _ = kappa_changepoint(kappa_hat, k_arr)
    return int(k_star)


def _run_one(features, sigma_mult):
    s, sigma, median_dist = _build_rbf(features, sigma_mult)
    sim_mean = float(s.mean())
    sim_min = float(s.min())
    print(f"  sigma_mult={sigma_mult}, sigma={sigma:.1f}, sim_mean={sim_mean:.4f}, sim_min={sim_min:.4f}")

    k_star = _run_kappa(s)
    print(f"  -> k*={k_star}")
    return {
        "sigma_mult": sigma_mult,
        "sigma": sigma,
        "median_dist": median_dist,
        "sim_mean": sim_mean,
        "sim_min": sim_min,
        "k_star": k_star,
    }


def main():
    features = np.load("/LOCAL/fmahner/similarity-factorization/data/features/vgg16/vgg16_features.npy")
    print(f"VGG16 features: {features.shape}")

    # Build all similarity matrices first (fast)
    conditions = []
    for mult in SIGMA_MULTS:
        s, sigma, median_dist = _build_rbf(features, mult)
        conditions.append(("rbf", mult, sigma, median_dist, s))
        print(f"  RBF sigma_mult={mult}, sim_mean={s.mean():.4f}, sim_min={s.min():.4f}")

    s_linear = features @ features.T
    s_linear = s_linear / np.max(s_linear)
    conditions.append(("linear", None, None, None, s_linear))
    print(f"  Linear, sim_mean={s_linear.mean():.4f}, sim_min={s_linear.min():.4f}")

    # Run sequentially, each using half the cores
    print(f"\nRunning kappa for {len(conditions)} conditions sequentially...")
    k_stars = [_run_kappa(s, n_jobs=70) for _, _, _, _, s in conditions]

    rows = []
    for (kernel, mult, sigma, median_dist, s), k_star in zip(conditions, k_stars):
        rows.append({
            "kernel": kernel,
            "sigma_mult": mult,
            "sigma": sigma,
            "median_dist": median_dist,
            "sim_mean": float(s.mean()),
            "sim_min": float(s.min()),
            "k_star": k_star,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\n{'='*50}")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
