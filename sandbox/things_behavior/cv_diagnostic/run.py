"""Deep dive: why doesn't things_behavior CV have a sharp U-shape?

Five diagnostics, each independently informative, all on the cached
things_behavior similarity matrix:

  A. Effective-rank check
       For nominal rank in {30, 50, 80, 120, 200}, fit SRF 20 times with
       different seeds and count nonzero columns of W per fit. If effective
       rank << nominal at high rank, SRF's non-negativity is "shutting
       down" the extra dims -> flat plateau is explained.

  B. Fine CV scan
       Rank 28-50 step 2, n_repeats=10, full 5fold CV. Sharper resolution
       than the step-10 grid; might reveal a real minimum the coarser
       grid missed.

  C. Convergence-vs-iterations test (rank=50 specifically)
       At the suspected-local-minima rank=50, vary max_outer in
       {50, 100, 200, 500}. If longer training drops val_mse there,
       the bump is a convergence artifact.

  D. PCA scree
       Top-200 eigenvalues of the cached similarity matrix. Compare the
       eigenvalue elbow to the SRF CV argmin.

  E. Train vs Val MSE
       At rank in {20, 30, 50, 80, 120}, compute both train and val MSE
       over 5 folds. If train_mse keeps falling but val_mse stays flat,
       overfitting *is* happening but small. If train_mse also flattens,
       SRF cannot fit more structure (data-intrinsic flat plateau).

Each diagnostic writes its CSV immediately so progress is visible. PNG
plots saved as each phase completes.

Run:
    poetry run python sandbox/things_behavior/cv_diagnostic/run.py
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
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from pysrf import SRF, cross_val_score
from src.colors import GRAY_LIGHT, INDIGO, ROSE, TEAL
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
SIM_PATH = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/cache/things_behavior.npy")
CV_JSON = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json")

SRF_KWARGS = dict(rho=3.0, max_inner=30, max_outer=50, tol=0.0, check_input=False)
N_JOBS = 8


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (OUTPUT_DIR / "log.txt").write_text(
        ((OUTPUT_DIR / "log.txt").read_text() if (OUTPUT_DIR / "log.txt").exists() else "")
        + line + "\n"
    )


def fit_srf_once(sim: np.ndarray, rank: int, seed: int, srf_kwargs: dict) -> dict:
    t0 = time.time()
    est = SRF(rank=rank, random_state=seed, missing_values=np.nan, **srf_kwargs)
    est.fit(sim.astype(np.float64))
    w = est.w_
    elapsed = time.time() - t0
    col_norms = np.linalg.norm(w, axis=0)
    col_max = w.max(axis=0)
    nnz_cols = int((col_max > 1e-8).sum())
    # train MSE on observed (non-nan) entries
    mask = np.isfinite(sim)
    np.fill_diagonal(mask, False)
    recon = w @ w.T
    train_mse = float(np.mean((sim[mask] - recon[mask]) ** 2))
    return {
        "rank": rank, "seed": seed, "fit_sec": elapsed,
        "nnz_cols_w": nnz_cols,
        "col_norm_max": float(col_norms.max()),
        "col_norm_min": float(col_norms.min()),
        "train_mse": train_mse,
    }


# ============================================================================
# A. Effective rank
# ============================================================================
def phase_a(sim: np.ndarray) -> None:
    log("=== Phase A: effective rank check ===")
    ranks = [30, 50, 80, 120, 200]
    seeds = list(range(20))
    csv_path = OUTPUT_DIR / "a_effective_rank.csv"

    rows = []
    for rank in ranks:
        log(f"  rank={rank}: fitting {len(seeds)} SRFs with different seeds (n_jobs={N_JOBS})")
        t0 = time.time()
        results = Parallel(n_jobs=N_JOBS)(
            delayed(fit_srf_once)(sim, rank, seed, SRF_KWARGS) for seed in seeds
        )
        rows.extend(results)
        pd.DataFrame(rows).to_csv(csv_path, index=False)  # incremental save
        nnz = [r["nnz_cols_w"] for r in results if r["rank"] == rank]
        log(f"    nnz cols (mean ± std): {np.mean(nnz):.1f} ± {np.std(nnz):.1f}  "
            f"of nominal {rank}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    summary = df.groupby("rank").agg(eff_mean=("nnz_cols_w", "mean"),
                                      eff_std=("nnz_cols_w", "std"))
    ax.errorbar(summary.index, summary["eff_mean"], yerr=summary["eff_std"],
                marker="o", color=INDIGO, capsize=3, lw=1.5)
    ax.plot(summary.index, summary.index, "--", color=GRAY_LIGHT, label="y=x (nominal)")
    ax.set_xlabel("nominal SRF rank")
    ax.set_ylabel("# nonzero columns in W")
    ax.set_title("A. Effective rank: does SRF shut down extra dims?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "a_effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved a_effective_rank.csv + .png")


# ============================================================================
# B. Fine CV scan
# ============================================================================
def phase_b(sim: np.ndarray, sampling_fraction: float) -> None:
    log("=== Phase B: fine CV scan (rank 28-50 step 2) ===")
    ranks = list(range(28, 52, 2))
    csv_path = OUTPUT_DIR / "b_fine_cv.csv"
    t0 = time.time()
    curve = cross_val_score(
        sim, ranks=ranks, sampling_fraction=sampling_fraction,
        n_folds=5, n_repeats=10, random_state=42, n_jobs=N_JOBS,
        srf_kwargs=SRF_KWARGS,
    )
    log(f"  fine CV done in {time.time()-t0:.1f}s")
    summary = curve.groupby("rank")["val_mse"].agg(["mean", "std", "count"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary.to_csv(csv_path)
    for r, row in summary.iterrows():
        log(f"  rank={r:>3}: {row['mean']:.6e} ± {row['sem']:.2e}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = summary.index.to_numpy()
    y = summary["mean"].to_numpy()
    sem = summary["sem"].to_numpy()
    ax.plot(x, y, marker="o", color=INDIGO, lw=1.5)
    ax.fill_between(x, y - 2*sem, y + 2*sem, color=INDIGO, alpha=0.25)
    ax.errorbar(x, y, yerr=2*sem, fmt="none", ecolor=INDIGO, alpha=0.6, capsize=2)
    ax.set_xlabel("rank")
    ax.set_ylabel("Validation MSE")
    ax.set_title("B. Fine CV (rank 28-50 step 2, n_repeats=10)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "b_fine_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# C. Convergence test at rank=50
# ============================================================================
def phase_c(sim: np.ndarray, sampling_fraction: float) -> None:
    log("=== Phase C: convergence at rank=50 varying max_outer ===")
    csv_path = OUTPUT_DIR / "c_convergence.csv"
    rows = []
    for max_outer in [50, 100, 200, 500]:
        srf_kw = {**SRF_KWARGS, "max_outer": max_outer}
        t0 = time.time()
        curve = cross_val_score(
            sim, ranks=[50], sampling_fraction=sampling_fraction,
            n_folds=5, n_repeats=10, random_state=42, n_jobs=N_JOBS,
            srf_kwargs=srf_kw,
        )
        val_mse = curve["val_mse"].mean()
        val_sem = curve["val_mse"].std(ddof=1) / np.sqrt(len(curve))
        elapsed = time.time() - t0
        rows.append({"max_outer": max_outer, "val_mse": val_mse,
                      "val_sem": val_sem, "elapsed_sec": elapsed})
        log(f"  max_outer={max_outer}: val_mse={val_mse:.6e} ± {val_sem:.2e}  ({elapsed:.1f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(df["max_outer"], df["val_mse"], yerr=2*df["val_sem"],
                marker="o", color=INDIGO, capsize=3, lw=1.5)
    ax.set_xlabel("max_outer (SRF iters)")
    ax.set_ylabel("Validation MSE at rank=50")
    ax.set_title("C. Does training longer at rank=50 reduce the bump?")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "c_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# D. PCA scree of the similarity matrix
# ============================================================================
def phase_d(sim: np.ndarray) -> None:
    log("=== Phase D: PCA scree of similarity ===")
    sim_clean = np.nan_to_num(sim, nan=0.0)
    sim_clean = 0.5 * (sim_clean + sim_clean.T)
    eigvals = np.linalg.eigvalsh(sim_clean)[::-1][:200]
    pd.DataFrame({"index": np.arange(1, len(eigvals)+1), "eigenvalue": eigvals}).to_csv(
        OUTPUT_DIR / "d_eigenvalues.csv", index=False)
    log(f"  top-5 eigenvalues: {eigvals[:5]}")
    log(f"  eigenvalue at rank 30: {eigvals[29]:.4f}")
    log(f"  eigenvalue at rank 50: {eigvals[49]:.4f}")
    log(f"  eigenvalue at rank 100: {eigvals[99]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(np.arange(1, len(eigvals)+1), eigvals, marker=".", ms=3,
                 color=INDIGO)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("eigenvalue (linear)")
    axes[0].set_title("D. PCA scree (linear)")
    axes[1].semilogy(np.arange(1, len(eigvals)+1), np.clip(eigvals, 1e-12, None),
                     marker=".", ms=3, color=INDIGO)
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("eigenvalue (log)")
    axes[1].set_title("D. PCA scree (log)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d_pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# E. Train vs val MSE
# ============================================================================
def phase_e(sim: np.ndarray) -> None:
    log("=== Phase E: train vs val MSE ===")
    ranks = [20, 30, 50, 80, 120]
    n_seeds = 5  # one fit per (rank, seed) on the whole sim (no fold)
    csv_path = OUTPUT_DIR / "e_train_val.csv"
    rows = []
    for rank in ranks:
        results = Parallel(n_jobs=N_JOBS)(
            delayed(fit_srf_once)(sim, rank, seed, SRF_KWARGS) for seed in range(n_seeds)
        )
        for r in results:
            rows.append(r)
        train_mean = np.mean([r["train_mse"] for r in results])
        log(f"  rank={rank}: train_mse_mean={train_mean:.6e}")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    summary = df.groupby("rank").agg(train_mean=("train_mse", "mean"),
                                      train_std=("train_mse", "std"))
    ax.errorbar(summary.index, summary["train_mean"],
                yerr=summary["train_std"], marker="o", color=INDIGO,
                label="train MSE (full-fit)", capsize=3, lw=1.5)
    ax.set_xlabel("rank")
    ax.set_ylabel("Train MSE (on observed entries, no CV mask)")
    ax.set_title("E. Train MSE — does SRF actually fit more with more rank?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "e_train_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = np.load(SIM_PATH).astype(np.float64)
    log(f"loaded similarity n={sim.shape[0]}")
    cv = json.loads(CV_JSON.read_text())
    sampling_fraction = float(cv["validations"]["5fold"]["params"]["sampling_fraction"])
    log(f"sampling_fraction (p*) = {sampling_fraction:.4f}")

    phase_d(sim)          # quickest, no fits
    phase_a(sim)          # SRF fits, ~10 min
    phase_e(sim)          # SRF fits, ~5 min
    phase_c(sim, sampling_fraction)  # CV, ~15 min
    phase_b(sim, sampling_fraction)  # CV, ~30 min  (do last; largest)
    log("=== ALL PHASES DONE ===")


if __name__ == "__main__":
    main()
