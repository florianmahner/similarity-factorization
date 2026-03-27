"""Joint bandwidth-rank selection via kappa sharpness and harmonic mean criterion.

For each bandwidth multiplier alpha, estimates rank via kappa, then runs 5 SRF
fits to measure factorization stability and explained variance. Selects alpha*
that maximizes H(stability, R^2) = 2 * stability * R^2 / (stability + R^2).
Also tests whether kappa changepoint SNR predicts the harmonic mean ranking.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, pairwise_kernels

from pysrf import SRF
from src.coherence import (
    _estimate_kappa_hat,
    _mad,
    _smooth_median,
    compute_incremental_coherence_multi_k_eig_anisotropic,
    kappa_changepoint,
)
from src.utils import get_output_dir

log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

FEATURES_PATH = Path("experiments/sandbox/things/dino_extract/outputs/latest/dinov3_features.npy")
SIGMA_MULTS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Coherence parameters (reduced for speed)
B = 20
B_NULL = 15
N_P = 20
K_MAX = 100
HI_BAND_QUANTILE = 0.85
SMOOTH_WINDOW = 5

# Stability parameters
N_SRF_RUNS = 5


def _build_rbf(features: np.ndarray, sigma_mult: float) -> tuple[np.ndarray, float]:
    """Build RBF similarity matrix with given multiplier of the median heuristic."""
    dist = pairwise_distances(features, metric="euclidean")
    median_dist = float(np.median(dist[np.triu_indices(len(dist), k=1)]))
    sigma = median_dist * sigma_mult
    gamma = 1 / (2 * sigma**2)
    s = pairwise_kernels(features, metric="rbf", gamma=gamma)
    return s, sigma


def _compute_kappa_snr(s: np.ndarray) -> dict:
    """Compute kappa curve and changepoint SNR for a similarity matrix.

    Returns dict with k_star, kappa_hat, snr, and diagnostics.
    """
    n = s.shape[0]
    k_list = list(range(1, min(K_MAX, n - 1) + 1))
    p_list = np.linspace(0.05, 0.95, N_P)

    result = compute_incremental_coherence_multi_k_eig_anisotropic(
        s,
        k_list=k_list,
        p_list=p_list,
        B=B,
        random_state=42,
        compute_null=True,
        B_null=B_NULL,
        alpha_tau=0.95,
        ci_level=0.95,
        use_baseline_correction=False,
        n_jobs=os.cpu_count() - 1,
        show_progress=False,
        visualize=False,
    )

    diag = result["diagnostics"]
    x_median = diag["x_median"]
    k_arr = result["k_list"]
    p_arr = result["p"]

    kappa_hat, kappa_info = _estimate_kappa_hat(x_median, p_arr, HI_BAND_QUANTILE)
    k_star, cp_info = kappa_changepoint(kappa_hat, k_arr, smooth_window=SMOOTH_WINDOW)

    # Changepoint SNR: max |delta_kappa_smooth| / MAD(kappa_hat)
    kappa_smooth = _smooth_median(kappa_hat, SMOOTH_WINDOW)
    delta = np.diff(kappa_smooth)
    mad_val = _mad(kappa_hat)
    snr = float(np.max(np.abs(delta)) / mad_val)

    return {
        "k_star": int(k_star),
        "kappa_hat": kappa_hat,
        "kappa_smooth": kappa_smooth,
        "snr": snr,
        "mad": mad_val,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "k_arr": k_arr,
    }


def run_stage1(features: np.ndarray) -> tuple[list[dict], dict]:
    """Stage 1: Compute kappa changepoint SNR for each bandwidth.

    Returns list of record dicts and dict of kappa curves keyed by sigma_mult.
    """
    records = []
    kappa_curves = {}

    for mult in SIGMA_MULTS:
        log.info(f"Stage 1: sigma_mult={mult}")
        s, sigma = _build_rbf(features, mult)
        sim_mean = float(s.mean())
        sim_min = float(s.min())
        log.info(f"  sigma={sigma:.2f}, sim_mean={sim_mean:.4f}, sim_min={sim_min:.4f}")

        kappa_result = _compute_kappa_snr(s)
        log.info(f"  k*={kappa_result['k_star']}, SNR={kappa_result['snr']:.3f}")

        records.append({
            "sigma_mult": mult,
            "sigma": sigma,
            "k_star": kappa_result["k_star"],
            "snr": kappa_result["snr"],
            "mad": kappa_result["mad"],
            "max_abs_delta": kappa_result["max_abs_delta"],
            "sim_mean": sim_mean,
            "sim_min": sim_min,
        })

        kappa_curves[str(mult)] = kappa_result["kappa_hat"]

    return records, kappa_curves


def _fit_one_srf(s: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Fit a single SRF model and return the embedding."""
    model = SRF(rank=rank, random_state=seed)
    model.fit(s)
    return model.w_


def _align_and_correlate(w_a: np.ndarray, w_b: np.ndarray) -> np.ndarray:
    """Align two embedding matrices via Hungarian matching on absolute correlation.

    Returns per-dimension correlation after optimal alignment.
    """
    k = w_a.shape[1]
    corr_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            r = np.corrcoef(w_a[:, i], w_b[:, j])[0, 1]
            corr_matrix[i, j] = r if np.isfinite(r) else 0.0

    # Hungarian on negative absolute correlation (minimize cost = maximize matching)
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr_matrix))

    per_dim_corr = np.zeros(k)
    for i, j in zip(row_idx, col_idx):
        per_dim_corr[i] = abs(corr_matrix[i, j])

    return per_dim_corr


def _explained_variance(s: np.ndarray, w: np.ndarray) -> float:
    """R^2 of the low-rank reconstruction WW^T relative to S."""
    reconstruction = w @ w.T
    ss_res = np.sum((s - reconstruction) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _compute_stability_and_variance(s: np.ndarray, rank: int) -> dict:
    """Run N_SRF_RUNS independent SRF fits, compute stability and explained variance.

    Stability: mean pairwise correlation of Hungarian-aligned embeddings (Fisher-z averaged).
    Explained variance: mean R^2 of WW^T reconstructions across runs.
    """
    embeddings = Parallel(n_jobs=-1)(
        delayed(_fit_one_srf)(s, rank, seed) for seed in range(N_SRF_RUNS)
    )

    # Explained variance: mean R^2 across all runs
    r2_values = np.array([_explained_variance(s, w) for w in embeddings])
    r2_mean = float(np.mean(r2_values))

    # Pairwise stability (C(5,2) = 10 pairs)
    n_runs = len(embeddings)
    all_per_dim = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            per_dim = _align_and_correlate(embeddings[i], embeddings[j])
            all_per_dim.append(per_dim)

    all_per_dim = np.array(all_per_dim)  # (n_pairs, k)

    # Average in Fisher-z space
    z_scores = np.arctanh(np.clip(all_per_dim, -0.999, 0.999))
    mean_z = np.mean(z_scores, axis=0)
    reliability_per_dim = np.tanh(mean_z)
    stability_mean = float(np.mean(reliability_per_dim))

    # Harmonic mean of stability and R^2
    if stability_mean > 0 and r2_mean > 0:
        h_mean = float(2 * stability_mean * r2_mean / (stability_mean + r2_mean))
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


def run_stage2(features: np.ndarray, stage1_records: list[dict]) -> list[dict]:
    """Stage 2: Compute stability, R^2, and harmonic mean for each bandwidth."""
    stability_records = []

    for rec in stage1_records:
        mult = rec["sigma_mult"]
        k_star = rec["k_star"]
        log.info(f"Stage 2: sigma_mult={mult}, rank={k_star}")

        s, _ = _build_rbf(features, mult)
        result = _compute_stability_and_variance(s, k_star)
        log.info(f"  stability={result['stability_mean']:.3f}, "
                 f"R2={result['r2_mean']:.3f}, H={result['h_mean']:.3f}")

        stability_records.append({
            "sigma_mult": mult,
            "k_star": k_star,
            **result,
        })

    return stability_records


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info(f"Loading features from {FEATURES_PATH}")
    features = np.load(FEATURES_PATH)
    log.info(f"Features shape: {features.shape}")

    # Stage 1: Kappa changepoint sharpness
    log.info("\n=== Stage 1: Kappa changepoint sharpness ===")
    stage1_records, kappa_curves = run_stage1(features)

    # Stage 2: Profile stability
    log.info("\n=== Stage 2: Profile stability ===")
    stage2_records = run_stage2(features, stage1_records)

    # Merge records
    rows = []
    stability_per_dim = {}
    for s1, s2 in zip(stage1_records, stage2_records):
        rows.append({
            "sigma_mult": s1["sigma_mult"],
            "sigma": s1["sigma"],
            "k_star": s1["k_star"],
            "snr": s1["snr"],
            "sim_mean": s1["sim_mean"],
            "stability_mean": s2["stability_mean"],
            "stability_min": s2["stability_min"],
            "r2_mean": s2["r2_mean"],
            "r2_std": s2["r2_std"],
            "h_mean": s2["h_mean"],
        })
        stability_per_dim[str(s1["sigma_mult"])] = s2["reliability_per_dim"]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    log.info(f"\nSaved results to {OUTPUT_DIR / 'results.csv'}")

    # Save kappa curves and per-dim stability for diagnostics
    np.savez(OUTPUT_DIR / "kappa_curves.npz", **kappa_curves)
    np.savez(OUTPUT_DIR / "stability_per_dim.npz", **stability_per_dim)

    # Stage 3: Validation -- compare rankings
    log.info("\n=== Stage 3: Validation ===")
    from scipy.stats import spearmanr

    h_best_idx = int(df["h_mean"].idxmax())
    h_best_mult = df.loc[h_best_idx, "sigma_mult"]
    snr_best_idx = int(df["snr"].idxmax())
    snr_best_mult = df.loc[snr_best_idx, "sigma_mult"]

    rho_snr_h, p_snr_h = spearmanr(df["snr"].values, df["h_mean"].values)
    rho_snr_stab, p_snr_stab = spearmanr(df["snr"].values, df["stability_mean"].values)

    log.info(f"Best by H(stab, R2): sigma_mult={h_best_mult} "
             f"(stab={df.loc[h_best_idx, 'stability_mean']:.3f}, "
             f"R2={df.loc[h_best_idx, 'r2_mean']:.3f}, "
             f"H={df.loc[h_best_idx, 'h_mean']:.3f})")
    log.info(f"Best by SNR:         sigma_mult={snr_best_mult}")
    log.info(f"Spearman rho(SNR, H) = {rho_snr_h:.3f} (p={p_snr_h:.4f})")
    log.info(f"Spearman rho(SNR, stability) = {rho_snr_stab:.3f} (p={p_snr_stab:.4f})")

    log.info(f"\n=== Summary table ===")
    log.info(df[["sigma_mult", "k_star", "snr", "stability_mean", "r2_mean",
                 "h_mean", "sim_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
