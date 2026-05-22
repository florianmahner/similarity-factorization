"""CLIP RN50 on THINGS+ 1854 — LINEAR kernel + p* + coarse CV.

Test whether a lower-magnitude linear kernel (CLIP RN50 row norm median ~2 vs
vgg16's ~50) gives a U-shaped CV curve. Same pipeline as vgg16_rbf_cv but
linear kernel instead of RBF.

Caveat: CLIP features have negative entries -> linear kernel can have negative
off-diagonals (unlike vgg16's post-ReLU). pysrf handles via `bounds`.

Run:
    poetry run python sandbox/dimensionality/clip_rn50_linear/run.py
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
from tools.metrics import dot_similarity

FEAT_22K = Path("/data/labshare/_stachelschwein/SSD/projects/deep-similarity/data/features/things22k/clip_rn50.npy")
META = Path("/data/labshare/_stachelschwein/SSD/projects/deep-similarity/data/meta/things22k/image_info_22k.csv")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COARSE_RANKS = [50, 100, 150, 200]
N_FOLDS = 5
N_REPEATS = 1
N_JOBS = 64
SRF_KWARGS = dict(rho=3.0, max_inner=30, tol=0.0, max_outer=50, check_input=False, verbose=1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_plus_rows(meta: pd.DataFrame) -> pd.DataFrame:
    mask = meta["filename"].str.match(r".+_01[bsn]\.jpg$")
    plus = meta.loc[mask].copy()
    assert len(plus) == 1854 and plus["category"].is_unique
    return plus.sort_values("category").reset_index(drop=False).rename(columns={"index": "orig_row"})


def main() -> None:
    log(f"loading features from {FEAT_22K}")
    x_22k = np.load(FEAT_22K)
    meta = pd.read_csv(META)
    plus = select_plus_rows(meta)
    rows = plus["orig_row"].to_numpy()
    x = x_22k[rows].astype(np.float64)
    nrm = np.linalg.norm(x, axis=1)
    log(f"  features filtered to plus subset: shape={x.shape}  "
        f"row_norms median={np.median(nrm):.2f}  max={nrm.max():.2f}")

    log("building LINEAR kernel similarity (x @ x.T) ...")
    t0 = time.time()
    s = dot_similarity(x, x).astype(np.float64)
    off = s[np.triu_indices(s.shape[0], k=1)]
    log(f"  built in {time.time()-t0:.1f}s  shape={s.shape}  "
        f"range=[{s.min():.4f}, {s.max():.4f}]  median_off={np.median(off):.4f}  "
        f"frac_negative_off={(off < 0).mean():.4f}")
    np.save(OUT / "s_linear.npy", s)

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
    est_elapsed = time.time() - t0
    log(f"  k_cut={est.rank}  p*={est.sampling_fraction:.4f}  "
        f"floor={est.detectability_floor:.4f}  ({est_elapsed:.1f}s)")

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
    cv_elapsed = time.time() - t0
    log(f"  CV done in {cv_elapsed:.1f}s")

    cv.to_csv(OUT / "cv.csv", index=False)
    summary = cv.groupby("rank")["val_mse"].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(np.maximum(summary["count"], 1))
    summary = summary.sort_values("rank")
    summary.to_csv(OUT / "cv_summary.csv", index=False)

    argmin_rank = int(summary.loc[summary["mean"].idxmin(), "rank"])
    log("")
    log(f"{'rank':>5}  {'mean':>12}  {'sem':>10}")
    for _, row in summary.iterrows():
        marker = "  <- argmin" if int(row["rank"]) == argmin_rank else ""
        if int(row["rank"]) == int(est.rank) and int(row["rank"]) != argmin_rank:
            marker += "  <- k_cut"
        log(f"{int(row['rank']):>5}  {row['mean']:>12.6f}  {row['sem']:>10.6f}{marker}")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.errorbar(summary["rank"], summary["mean"], yerr=summary["sem"],
                marker="o", ms=5, lw=1.6, color="C2")
    ax.scatter([argmin_rank],
               [float(summary.loc[summary["rank"] == argmin_rank, "mean"].iloc[0])],
               marker="*", s=180, color="C2", edgecolors="white", linewidths=1.2, zorder=10)
    ax.axvline(est.rank, color="gray", ls="--", lw=0.9, label=f"k_cut={est.rank}")
    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"CLIP RN50 LINEAR  n=1854  p*={est.sampling_fraction:.3f}  argmin={argmin_rank}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "cv_curve.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "features_source": str(FEAT_22K),
        "kernel": "linear (x @ x.T, NO normalization)",
        "n_categories": int(s.shape[0]),
        "row_norm_median": float(np.median(nrm)),
        "frac_negative_off": float((off < 0).mean()),
        "k_cut": int(est.rank),
        "p_star": float(est.sampling_fraction),
        "detectability_floor": float(est.detectability_floor),
        "ranks_tested": ranks,
        "argmin_rank": argmin_rank,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "srf_kwargs": SRF_KWARGS,
        "estimate_seconds": round(est_elapsed, 1),
        "cv_seconds": round(cv_elapsed, 1),
    }, indent=2) + "\n")
    log(f"plot -> {OUT / 'cv_curve.png'}")


if __name__ == "__main__":
    main()
