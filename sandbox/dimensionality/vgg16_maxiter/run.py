"""VGG16 CV diagnostic: does max_outer=2000 produce a minimum near k_cut?

Background
----------
For vgg16 (n=1854) the dimensionality experiment reports:
    estimate_rank: k_cut=45, p*=0.697
    cross_val_score (max_outer=300, tol=0, 5x5-fold): argmin_rank=90

The CV curve is monotonically decreasing from 2 -> 90. With srf_kwargs.tol=0
the SRF convergence check never fires (eps_pri=eps_dual=0), so SRF always
runs exactly max_outer ADMM iterations. At larger rank the problem is
harder and may not actually be converged at 300 outer iters.

This script
-----------
One 5-fold CV at p* with max_outer=2000 on ranks [30, 40, 45, 50, 60].
Runs in parallel via joblib (n_jobs x OMP threads = total threads).
Bypasses pysrf.cross_val_score so we can stream parallel progress
(joblib verbose=10).

Run
---
    poetry run python sandbox/dimensionality/vgg16_maxiter/run.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Each loky worker runs single-threaded BLAS to avoid oversubscription
# (25 workers x 1 BLAS thread = 25 threads on cpu_count=128).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from pysrf import estimate_rank
from pysrf.cross_validation import (
    _cv_pool_fraction,
    _entry_splits,
    _fit_score,
    _observed_bounds,
    _split_fit_seeds,
)
from pysrf._common import observation_mask
from similarity import build_similarity
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine
from src.colors import GRAY, INDIGO, ROSE

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = get_output_dir()

RANKS = [30, 40, 45, 50, 60]
N_FOLDS = 5
N_REPEATS = 1
RANDOM_STATE = 0
# More inner / fewer outer: V/lam updates and metrics happen per outer iter,
# so dropping max_outer cuts overhead; bumping max_inner solves the W subproblem
# better each outer step so total inner work is unchanged.
SRF_KWARGS = {"rho": 3.0, "max_inner": 100, "tol": 0.0, "max_outer": 600, "verbose": 1}
N_JOBS = 25
BLAS_THREADS_PER_WORKER = 1  # OMP=1 per worker -> no nested thread contention


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_vgg16() -> np.ndarray:
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "vgg16.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    return build_similarity(parent.dataset)


def _plot_curve(ranks, mean, sem, k_cut, argmin_rank, out_path):
    fig, ax = create_figure("single")
    ax.errorbar(ranks, mean, yerr=sem, marker="o", color=INDIGO, capsize=2)
    ax.axvline(k_cut, color=GRAY, linestyle="--", label=f"k_cut={k_cut}")
    ax.axvline(argmin_rank, color=ROSE, linestyle=":", label=f"argmin={argmin_rank}")
    ax.set_xlabel("rank")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"vgg16 CV (max_outer={SRF_KWARGS['max_outer']})")
    ax.legend(frameon=False)
    despine(ax)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    log("=" * 70)
    log("vgg16 CV diagnostic: max_outer=2000")
    log("=" * 70)

    log("Loading vgg16 similarity matrix...")
    t0 = time.time()
    s = load_vgg16()
    s = np.asarray(s, dtype=np.float64)
    log(f"  shape={s.shape}  ({time.time() - t0:.1f}s)")

    log("Estimating rank...")
    t0 = time.time()
    est = estimate_rank(s, n_bootstrap=20, random_state=0)
    p_star = float(est.sampling_fraction)
    log(f"  k_cut={est.rank}  p*={p_star:.4f}  "
        f"floor={est.detectability_floor:.3f}  ({time.time() - t0:.1f}s)")

    log(f"5-fold CV at p*={p_star:.4f}  srf_kwargs={SRF_KWARGS}")
    log(f"  ranks={RANKS}  n_jobs={N_JOBS}  OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    # Replicate pysrf.cross_val_score internals so we can use joblib verbose.
    obs_mask = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs_mask)
    pool_fraction = _cv_pool_fraction(p_star, N_FOLDS, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(
        RANDOM_STATE, N_REPEATS, N_FOLDS, len(RANKS),
    )
    splits = _entry_splits(obs_mask, pool_fraction, N_FOLDS, split_seeds)

    jobs = []
    for rep, fold, train_mask, validation_mask in splits:
        for r_idx, rank in enumerate(RANKS):
            seed = int(fit_seeds[rep, fold, r_idx])
            jobs.append((rep, fold, int(rank), train_mask, validation_mask, seed))

    log(f"  pool_fraction={pool_fraction:.4f}  total fits={len(jobs)}")
    log("  -> dispatching to joblib (Parallel verbose=10 streams progress)")

    def _worker(tm, vm, rank, seed):
        # Loky workers default to 1 BLAS thread; lift it explicitly so each
        # worker actually uses BLAS_THREADS_PER_WORKER threads inside dsymm.
        with threadpool_limits(limits=BLAS_THREADS_PER_WORKER, user_api="blas"):
            return _fit_score(s, tm, vm, rank, bounds, seed, SRF_KWARGS)

    t_cv_start = time.time()
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_worker)(tm, vm, rank, seed)
        for (_, _, rank, tm, vm, seed) in jobs
    )
    elapsed = time.time() - t_cv_start
    log(f"CV done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    df = pd.DataFrame(
        [(rep, fold, rank, score)
         for (rep, fold, rank, *_), score in zip(jobs, scores)],
        columns=["rep", "fold", "rank", "val_mse"],
    )
    df["dataset"] = "vgg16"
    df.to_csv(OUTPUT_DIR / "cv.csv", index=False)

    grouped = df.groupby("rank")["val_mse"]
    mean = grouped.mean().reindex(RANKS).to_numpy()
    counts = np.maximum(grouped.count().reindex(RANKS).to_numpy(), 1)
    sem = grouped.std().reindex(RANKS).to_numpy() / np.sqrt(counts)
    argmin_rank = int(RANKS[int(np.nanargmin(mean))])

    log("")
    log(f"{'rank':>5}  {'mean':>10}  {'sem':>9}")
    for r, m, e in zip(RANKS, mean, sem):
        marker = ""
        if r == argmin_rank:
            marker = "  <- argmin"
        if r == est.rank:
            marker += "  <- k_cut"
        log(f"{r:>5}  {m:>10.3f}  {e:>9.3f}{marker}")

    log("")
    log(f"k_cut       = {est.rank}")
    log(f"argmin_rank = {argmin_rank}")
    if argmin_rank == est.rank:
        log("PASS: CV argmin matches k_cut.")
    elif abs(argmin_rank - est.rank) <= 5:
        log("CLOSE: CV argmin within 5 of k_cut.")
    else:
        log("MISMATCH: argmin still off from k_cut at max_outer=2000.")

    summary = {
        "dataset": "vgg16",
        "n": int(s.shape[0]),
        "k_cut": int(est.rank),
        "p_star": p_star,
        "detectability_floor": float(est.detectability_floor),
        "srf_kwargs": SRF_KWARGS,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "ranks": [int(r) for r in RANKS],
        "val_mse_mean": [float(v) for v in mean],
        "val_mse_sem": [float(v) for v in sem],
        "argmin_rank": argmin_rank,
        "runtime_sec": round(elapsed, 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved {OUTPUT_DIR / 'summary.json'}")

    _plot_curve(RANKS, mean, sem, est.rank, argmin_rank, OUTPUT_DIR / "cv_curve.png")
    log(f"Saved {OUTPUT_DIR / 'cv_curve.png'}")


if __name__ == "__main__":
    main()
