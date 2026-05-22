"""vgg16 with RBF (Gaussian) kernel — does the CV curve bottom out?

Pipeline:
  1. Build gaussian-kernel similarity from raw vgg16 features (median-heuristic sigma).
  2. estimate_rank -> k_cut, p*.
  3. cross_val_score at p* on a coarse rank grid (50, 100, 150, 200) plus k_cut.
  4. Save the CV table + a quick plot.

Run:
    poetry run python sandbox/dimensionality/vgg16_rbf_cv/run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from pysrf import cross_val_score, estimate_rank
from tools.metrics import gaussian_kernel_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = PROJECT_ROOT / "data/features/vgg16/vgg16_features.npy"
COARSE_RANKS = [50, 100, 150, 200]
N_FOLDS = 5
N_REPEATS = 1
N_JOBS = 64
SRF_KWARGS = {"rho": 3.0, "max_inner": 30, "tol": 0.0, "max_outer": 50,
              "check_input": False, "verbose": 1}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"loading features from {FEATURES}")
    x = np.load(FEATURES)
    log(f"  features shape={x.shape}  dtype={x.dtype}")

    log("building gaussian-kernel similarity (median-heuristic sigma)...")
    t0 = time.time()
    s = gaussian_kernel_similarity(x, x).astype(np.float64)
    log(f"  built in {time.time()-t0:.1f}s  shape={s.shape}  "
        f"range=[{s.min():.4f}, {s.max():.4f}]  median_off={np.median(s[np.triu_indices(s.shape[0],1)]):.4f}")
    np.save(OUT / "vgg16_rbf.npy", s)

    log("estimate_rank (recovery_tolerance=0.10) ...")
    t0 = time.time()
    est = estimate_rank(
        s,
        recovery_tolerance=0.10,
        max_rank=200,
        sampling_grid=np.linspace(0.05, 0.95, 20),
        n_bootstrap=20,
        random_state=0,
        n_jobs=N_JOBS,
    )
    log(f"  k_cut={est.rank}  p*={est.sampling_fraction:.4f}  "
        f"floor={est.detectability_floor:.4f}  ({time.time()-t0:.1f}s)")

    ranks = sorted(set(COARSE_RANKS) | {int(est.rank)})
    log(f"CV at p*={est.sampling_fraction:.4f}  n_folds={N_FOLDS}  ranks={ranks}")
    t0 = time.time()
    cv = cross_val_score(
        s, ranks=ranks,
        sampling_fraction=float(est.sampling_fraction),
        n_folds=N_FOLDS, n_repeats=N_REPEATS,
        random_state=0, n_jobs=N_JOBS,
        srf_kwargs=SRF_KWARGS,
    )
    elapsed = time.time() - t0
    log(f"  CV done in {elapsed:.1f}s")

    cv.to_csv(OUT / "cv_rbf.csv", index=False)
    summary = cv.groupby("rank")["val_mse"].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(np.maximum(summary["count"], 1))
    summary = summary.sort_values("rank")
    summary.to_csv(OUT / "cv_rbf_summary.csv", index=False)

    log("")
    log(f"{'rank':>5}  {'mean':>10}  {'sem':>9}")
    argmin_rank = int(summary.loc[summary["mean"].idxmin(), "rank"])
    for _, row in summary.iterrows():
        marker = "  <- argmin" if int(row["rank"]) == argmin_rank else ""
        marker += "  <- k_cut" if int(row["rank"]) == int(est.rank) and int(row["rank"]) != argmin_rank else ""
        log(f"{int(row['rank']):>5}  {row['mean']:>10.4f}  {row['sem']:>9.4f}{marker}")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar(summary["rank"], summary["mean"], yerr=summary["sem"],
                marker="o", ms=5, lw=1.6, color="C0")
    ax.scatter([argmin_rank],
               [float(summary.loc[summary["rank"] == argmin_rank, "mean"].iloc[0])],
               marker="*", s=180, color="C0",
               edgecolors="white", linewidths=1.2, zorder=10)
    ax.axvline(est.rank, color="gray", ls="--", lw=0.9, label=f"k_cut={est.rank}")
    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"vgg16 RBF kernel  p*={est.sampling_fraction:.3f}  argmin={argmin_rank}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "cv_curve_rbf.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "kernel": "gaussian (median-heuristic sigma)",
        "n": int(s.shape[0]),
        "k_cut": int(est.rank),
        "p_star": float(est.sampling_fraction),
        "detectability_floor": float(est.detectability_floor),
        "ranks_tested": ranks,
        "argmin_rank": argmin_rank,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "srf_kwargs": SRF_KWARGS,
        "estimate_seconds": None,
        "cv_seconds": round(elapsed, 1),
    }, indent=2) + "\n")
    log(f"summary -> {OUT / 'summary.json'}")
    log(f"plot   -> {OUT / 'cv_curve_rbf.png'}")


if __name__ == "__main__":
    main()
