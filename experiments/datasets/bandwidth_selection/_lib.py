"""Shared library for bandwidth (sigma_scale alpha) selection per dataset.

Protocol (from the paper):
  - sigma = alpha * median(pairwise euclidean distance over features)
  - alpha grid: [0.2, 0.4, 0.6, 0.8, 1.0]
  - per alpha: B=5 SRF fits at fixed rank
  - stability = mean of pysrf.AlignedConsensus.agreement_scores_
  - explained variance = 1 - var(S - W W^T) / var(S) on the centroid run
  - harmonic mean of (stability, explained_var) -> select alpha*
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_kernels

from pysrf import SRF
from pysrf.consensus import AlignedConsensus, EnsembleFit
from src.colors import CYCLE, INDIGO


def median_pairwise_distance(features: np.ndarray) -> float:
    """median of off-diagonal pairwise Euclidean distance."""
    d = pairwise_distances(features, metric="euclidean")
    triu = np.triu_indices_from(d, k=1)
    return float(np.median(d[triu]))


def build_rbf(features: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel with explicit sigma."""
    gamma = 1.0 / (2.0 * sigma**2)
    return pairwise_kernels(features, metric="rbf", gamma=gamma)


def explained_variance(similarity: np.ndarray, w: np.ndarray) -> float:
    """1 - var(S - W W^T) / var(S) on off-diagonal entries."""
    recon = w @ w.T
    mask = ~np.eye(similarity.shape[0], dtype=bool) & np.isfinite(similarity)
    s_vals = similarity[mask]
    r_vals = recon[mask]
    s_var = float(np.var(s_vals))
    if s_var <= 0:
        return float("nan")
    res_var = float(np.var(s_vals - r_vals))
    return 1.0 - res_var / s_var


def harmonic_mean(a: float, b: float) -> float:
    if a + b <= 0:
        return float("nan")
    return 2.0 * a * b / (a + b)


def run_bandwidth_selection(
    *,
    dataset_name: str,
    features: np.ndarray,
    rank: int,
    alphas: list[float],
    n_runs: int = 5,
    n_jobs: int = 8,
    output_dir: Path,
    srf_max_outer: int = 50,
    log_prefix: str = "",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {log_prefix}{msg}", flush=True)

    log(f"features shape: {features.shape}")
    log("computing median pairwise distance (this can take a minute on large n)...")
    t0 = time.time()
    med = median_pairwise_distance(features)
    log(f"  median pairwise distance = {med:.4f}  (computed in {time.time()-t0:.1f}s)")

    srf_kwargs = dict(rho=3.0, max_inner=30, max_outer=srf_max_outer,
                      tol=0.0, check_input=False)

    rows = []
    for alpha in alphas:
        sigma = alpha * med
        log(f"=== alpha={alpha} (sigma={sigma:.4f}) ===")
        t0 = time.time()
        sim = build_rbf(features, sigma=sigma)
        log(f"  built RBF sim in {time.time()-t0:.1f}s   "
            f"off-diag mean={sim[~np.eye(sim.shape[0], dtype=bool)].mean():.4f}  "
            f"min={sim[~np.eye(sim.shape[0], dtype=bool)].min():.4f}")

        t0 = time.time()
        ens = EnsembleFit(
            SRF(rank=rank, random_state=0, missing_values=np.nan, **srf_kwargs),
            n_runs=n_runs, random_state=0, n_jobs=n_jobs,
        )
        con = AlignedConsensus(rank=rank)
        ens.fit(sim)
        con.fit(ens.embeddings_)
        elapsed = time.time() - t0
        log(f"  ensemble+consensus done in {elapsed:.1f}s  (n_runs={n_runs})")

        stability = float(np.mean(con.agreement_scores_))
        # Explained variance averaged across all runs.
        # AlignedConsensus.aligned_embeddings_ is (n_runs, n, rank).
        aligned = con.aligned_embeddings_
        evs = [explained_variance(sim, aligned[i]) for i in range(aligned.shape[0])]
        ev_mean = float(np.mean(evs))
        ev_std = float(np.std(evs))
        hmean = harmonic_mean(stability, ev_mean)

        rows.append({
            "alpha": alpha, "sigma": sigma, "rank": rank, "n_runs": n_runs,
            "stability_mean": stability,
            "explained_var_mean": ev_mean, "explained_var_std": ev_std,
            "harmonic_mean": hmean,
            "elapsed_sec": elapsed,
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log(f"  stability={stability:.4f}  EV={ev_mean:.4f}±{ev_std:.4f}  H={hmean:.4f}")

    df = pd.DataFrame(rows)
    star_idx = int(df["harmonic_mean"].idxmax())
    alpha_star = float(df.loc[star_idx, "alpha"])
    log(f"alpha* (max harmonic mean) = {alpha_star}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["alpha"], df["stability_mean"], marker="o", color=CYCLE[0],
            label="stability (agreement)", linewidth=1.5)
    ax.plot(df["alpha"], df["explained_var_mean"], marker="s", color=CYCLE[1],
            label="explained variance", linewidth=1.5)
    ax.plot(df["alpha"], df["harmonic_mean"], marker="D", color=INDIGO,
            label="harmonic mean", linewidth=2.0)
    ax.axvline(alpha_star, color=INDIGO, linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("alpha (sigma_scale)")
    ax.set_ylabel("score")
    ax.set_title(f"{dataset_name}: bandwidth selection (rank={rank})  alpha* = {alpha_star}")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {csv_path} + alpha_curves.png")

    # Write a tiny json with the selected alpha for downstream use
    (output_dir / "selected.json").write_text(json.dumps({
        "dataset": dataset_name, "rank": rank, "alpha_star": alpha_star,
        "harmonic_mean_star": float(df.loc[star_idx, "harmonic_mean"]),
        "stability_star": float(df.loc[star_idx, "stability_mean"]),
        "ev_star": float(df.loc[star_idx, "explained_var_mean"]),
    }, indent=2))
    return df
