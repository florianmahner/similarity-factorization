"""Phase 1 convergence anchor for macaque22k SRF CV.

Question: does the monotonically decreasing CV curve reflect a genuinely
high-rank dataset, or undertrained SRF fits (the same pathology that hid
the U-shape in things_behavior at max_outer=50)?

Test: at ONE rank (k=50, the middle of the existing CV grid), sweep
max_outer in {10, 50, 100, 150, 200, 250} across 5 entry-prekfold folds.
If val_mse drops with max_outer, training is the bottleneck. If it
plateaus by max_outer=100, training is fine.

Threading: bypasses ``cross_val_score`` so we can use multi-threaded BLAS
(pysrf hardcodes ``threadpool_limits(limits=1)`` inside ``_fit_score``).

Layout: 25 parallel fits × 4 BLAS threads = 100 cores on prefrontal.
30 fits total (6 max_outer × 5 folds × 1 repeat). Longest-first
submission so the slow tail starts immediately.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from pysrf import SRF
from pysrf._common import observation_mask
from pysrf.cross_validation import (
    _cv_pool_fraction,
    _entry_splits,
    _observed_bounds,
    _split_fit_seeds,
)

from src.utils import get_output_dir

ROOT = Path("/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
CACHE_PATH = ROOT / "experiments/datasets/dimensionality/outputs/cache/things_macaque22k.npy"
CV_JSON = ROOT / "experiments/datasets/dimensionality/outputs/things_macaque22k/cross_validation.json"

OUTPUT_DIR = get_output_dir()
CSV_PATH = OUTPUT_DIR / "convergence.csv"
SUMMARY_PATH = OUTPUT_DIR / "convergence_summary.csv"

RANK = 50
MAX_OUTERS = [10, 50, 100, 150, 200, 250]
N_FOLDS = 5
N_REPEATS = 1
RANDOM_STATE = 0
BLAS_THREADS = 6
PARALLEL_FITS = 20  # 20 × 6 = 120 cores; ~40 GB / worker × 20 = ~800 GB peak (out of 944)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fit_one(
    cache_path: Path,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    bounds: tuple[float, float],
    rank: int,
    max_outer: int,
    seed: int,
    srf_kwargs_base: dict,
    blas_threads: int,
    label: str,
) -> dict:
    t0 = time.time()
    # mmap so 20 workers share the OS page cache for s instead of each holding 4 GB.
    s_mm = np.load(cache_path, mmap_mode="r")
    with threadpool_limits(limits=blas_threads):
        # Build train matrix once; fold s values into the observed positions.
        train = np.full(s_mm.shape, np.nan, dtype=np.float64)
        train[train_mask] = np.asarray(s_mm[train_mask])
        kwargs = {**srf_kwargs_base, "max_outer": max_outer}
        est = SRF(
            rank=rank,
            bounds=bounds,
            missing_values=np.nan,
            random_state=seed,
            verbose=1,
            **kwargs,
        )
        est.fit(train)
        del train
        s_hat = est.reconstruct()

        # Compute val_mse without allocating triu_indices (which would be ~4 GB of int64).
        ii, jj = np.where(val_mask)
        upper = ii < jj
        ii = ii[upper]
        jj = jj[upper]
        s_vals = np.asarray(s_mm[ii, jj], dtype=np.float64)
        s_hat_vals = s_hat[ii, jj]
        finite = np.isfinite(s_vals) & np.isfinite(s_hat_vals)
        n_entries = int(finite.sum())
        if n_entries == 0:
            val_mse = float("nan")
        else:
            residual = s_vals[finite] - s_hat_vals[finite]
            val_mse = float(np.mean(residual * residual))
    return {
        "label": label,
        "max_outer": max_outer,
        "val_mse": val_mse,
        "n_entries": n_entries,
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(CV_JSON.read_text())
    sampling_fraction = float(payload["sampling_fraction"])
    log(f"loaded existing CV: sampling_fraction={sampling_fraction:.4f}")

    srf_kwargs_base = {"rho": 3.0, "max_inner": 30, "tol": 0.0, "check_input": False}

    log(f"loading similarity from {CACHE_PATH}")
    s = np.load(CACHE_PATH)
    log(f"  n={s.shape[0]}  dtype={s.dtype}  min={s.min():.4f}  max={s.max():.4f}")
    obs = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs)
    pool_fraction = _cv_pool_fraction(sampling_fraction, N_FOLDS, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(RANDOM_STATE, N_REPEATS, N_FOLDS, 1)
    splits = _entry_splits(obs, pool_fraction, N_FOLDS, split_seeds)
    log(f"prepared {len(splits)} splits (p_pool={pool_fraction:.4f}, bounds={bounds})")

    # free the in-process copy of s — workers will reload via np.load
    del s

    jobs = []
    # Longest-first so the slow tail starts immediately
    for max_outer in sorted(MAX_OUTERS, reverse=True):
        for rep, fold, train_mask, val_mask in splits:
            seed = int(fit_seeds[rep, fold, 0])
            label = f"mo={max_outer:>3d}_fold={fold}_seed={seed}"
            jobs.append({
                "label": label,
                "max_outer": max_outer,
                "fold": fold,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "seed": seed,
            })

    log(f"=== {len(jobs)} fits queued ===")
    log(f"  rank={RANK}  max_outers={MAX_OUTERS}  n_folds={N_FOLDS}  n_repeats={N_REPEATS}")
    log(f"  parallel_fits={PARALLEL_FITS}  blas_threads={BLAS_THREADS}  total_cores={PARALLEL_FITS * BLAS_THREADS}")
    log(f"  output: {OUTPUT_DIR}")

    started = time.time()
    rows: list[dict] = []

    with Parallel(n_jobs=PARALLEL_FITS, return_as="generator", verbose=10, backend="loky") as parallel:
        results = parallel(
            delayed(_fit_one)(
                CACHE_PATH, j["train_mask"], j["val_mask"], bounds, RANK,
                j["max_outer"], j["seed"], srf_kwargs_base, BLAS_THREADS, j["label"],
            )
            for j in jobs
        )
        for result in results:
            rows.append(result)
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
            log(f"  done [{len(rows):>2d}/{len(jobs)}]  {result['label']}: "
                f"val_mse={result['val_mse']:.6e}  ({result['elapsed_sec']:.0f}s)")

    elapsed = time.time() - started
    log(f"=== all {len(rows)} fits done in {elapsed:.0f}s = {elapsed/3600:.2f}h ===")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("max_outer")["val_mse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "val_mse_mean"})
    )
    summary["val_mse_sem"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.sort_values("max_outer").reset_index(drop=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    log("=== summary ===")
    for _, row in summary.iterrows():
        log(f"  max_outer={int(row['max_outer']):>3d}: "
            f"val_mse={row['val_mse_mean']:.6e} ± {row['val_mse_sem']:.2e}  (n={int(row['count'])})")
    log(f"saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
