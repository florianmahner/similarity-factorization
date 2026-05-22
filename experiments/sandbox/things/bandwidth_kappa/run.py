"""Test effect of RBF bandwidth on kappa rank estimation.

Compare kappa k* for DINOv3 and NSD at different sigma multipliers,
with and without centering.

Answers: is the rank estimate driven by bandwidth choice or centering?
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels

from src.coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def _build_rbf(features: np.ndarray, sigma_mult: float) -> np.ndarray:
    dist = pairwise_distances(features, metric="euclidean")
    sigma = np.median(dist[np.triu_indices(len(dist), k=1)]) * sigma_mult
    gamma = 1 / (2 * sigma**2)
    s = pairwise_kernels(features, metric="rbf", gamma=gamma)
    return s, sigma


def _run_kappa(s: np.ndarray, center: bool, k_max: int = 100) -> dict:
    n = s.shape[0]
    if center:
        h = np.eye(n) - np.ones((n, n)) / n
        s = h @ s @ h

    k_list = list(range(1, min(k_max, n - 1) + 1))
    p_list = np.linspace(0.05, 0.95, 25)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s, k_list, p_list, B=30, random_state=42, n_jobs=-1,
    )

    kappa_result = kappa_changepoint(
        result["iproj"], result["iproj_null"],
        k_list, p_list, alpha_tau=0.95, ci_level=0.95,
    )
    return {
        "k_star": int(kappa_result["k_star"]),
        "p_star": float(kappa_result.get("p_star", 0)),
    }


def main():
    import pandas as pd

    sigma_mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []

    # DINOv3
    print("Loading DINOv3 features...")
    feat_dino = np.load("data/features/dinov3/dinov3_features.npy")
    print(f"  shape: {feat_dino.shape}")

    for mult in sigma_mults:
        s, sigma = _build_rbf(feat_dino, mult)
        sim_range = s.min(), s.max(), s.mean()
        print(f"\nDINOv3 sigma={sigma:.1f} ({mult}x median), sim=[{sim_range[0]:.3f}, {sim_range[1]:.3f}], mean={sim_range[2]:.3f}")

        for center in [False, True]:
            label = "centered" if center else "raw"
            print(f"  {label}: computing kappa...")
            try:
                res = _run_kappa(s, center=center, k_max=100)
                print(f"    k* = {res['k_star']}")
                rows.append({
                    "dataset": "dinov3", "sigma_mult": mult, "sigma": sigma,
                    "center": center, "k_star": res["k_star"],
                    "sim_min": sim_range[0], "sim_max": sim_range[1], "sim_mean": sim_range[2],
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                rows.append({
                    "dataset": "dinov3", "sigma_mult": mult, "sigma": sigma,
                    "center": center, "k_star": -1,
                    "sim_min": sim_range[0], "sim_max": sim_range[1], "sim_mean": sim_range[2],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
