"""Macaque rank sweep at max_outer=150 — proper persistence + resume.

Question: with longer training (max_outer=150, vs the old 50) AND tol=1e-4
early-stopping, does the macaque CV curve show a U-shape minimum or keep
dropping? Sweep 4 ranks {30, 50, 80, 120} × 5 folds.

This experiment supersedes the convergence_anchor sweep (which I built
based on a misread of the user's request).

Persistence (critical so we never lose finished work to a crash):
- ``outputs/results/<label>.json`` — written atomically the moment each
  fit completes. Holds val_mse, n_iter, fit_time, per-iter history, and
  metadata.
- ``outputs/estimators/<label>.joblib`` — full fitted SRF estimator, so
  W and history can be reloaded later for analysis / warm-start.
- Resume: at startup, any label whose result JSON exists is skipped.

Threading: 20 fits in parallel × 6 BLAS threads each = 120 cores on
prefrontal. Bypasses ``cross_val_score`` to allow multi-threaded BLAS
(pysrf hardcodes ``threadpool_limits(limits=1)`` inside ``_fit_score``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
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
RESULTS_DIR = OUTPUT_DIR / "results"
ESTIMATORS_DIR = OUTPUT_DIR / "estimators"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

RANKS = [30, 50, 80, 120]
MAX_OUTER = 150
MAX_INNER = 30
TOL = 1e-4               # converges early when possible (deep-sim-neurips precedent)
RHO = 3.0
N_FOLDS = 5
N_REPEATS = 1
RANDOM_STATE = 0
BLAS_THREADS = 6
PARALLEL_FITS = 20       # 20 × 6 = 120 cores  ;  ~30 GB/worker × 20 = ~600 GB peak


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _label(rank: int, fold: int, seed: int) -> str:
    return f"rank{rank:03d}_fold{fold}_seed{seed}"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _fit_one(
    cache_path: Path,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    bounds: tuple[float, float],
    rank: int,
    seed: int,
    fold: int,
    srf_kwargs: dict,
    blas_threads: int,
    results_dir: Path,
    estimators_dir: Path,
) -> dict:
    label = _label(rank, fold, seed)
    result_path = results_dir / f"{label}.json"

    # Resume: if we already have a result JSON for this label, just return it
    if result_path.exists():
        return {"label": label, **json.loads(result_path.read_text()), "resumed": True}

    t0 = time.time()
    s_mm = np.load(cache_path, mmap_mode="r")
    with threadpool_limits(limits=blas_threads):
        train = np.full(s_mm.shape, np.nan, dtype=np.float64)
        train[train_mask] = np.asarray(s_mm[train_mask])

        est = SRF(
            rank=rank,
            bounds=bounds,
            missing_values=np.nan,
            random_state=seed,
            verbose=1,
            **srf_kwargs,
        )
        est.fit(train)
        n_iter = int(est.n_iter_) if hasattr(est, "n_iter_") else None
        del train
        s_hat = est.reconstruct()

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

    elapsed = time.time() - t0

    # Strip the big _observed_mask before saving to keep estimator files small;
    # it's trivially reconstructed from train_mask if needed.
    if hasattr(est, "_observed_mask"):
        est._observed_mask = None

    estimators_dir.mkdir(parents=True, exist_ok=True)
    estimator_path = estimators_dir / f"{label}.joblib"
    dump(est, estimator_path, compress=3)

    history = {}
    if hasattr(est, "history_") and est.history_:
        history = {k: [float(v) for v in vs] for k, vs in est.history_.items()}

    result = {
        "label": label,
        "rank": rank,
        "fold": fold,
        "seed": seed,
        "val_mse": val_mse,
        "n_entries_val": n_entries,
        "n_iter": n_iter,
        "fit_time_s": elapsed,
        "max_outer": srf_kwargs["max_outer"],
        "max_inner": srf_kwargs["max_inner"],
        "tol": srf_kwargs["tol"],
        "rho": srf_kwargs["rho"],
        "bounds": list(bounds),
        "estimator_path": str(estimator_path.relative_to(estimator_path.parents[2])),
        "history": history,
    }
    _atomic_write_json(result_path, result)
    return {**result, "resumed": False}


def _write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("rank")["val_mse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "val_mse_mean"})
    )
    summary["val_mse_sem"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.sort_values("rank").reset_index(drop=True)
    summary.to_csv(path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(CV_JSON.read_text())
    sampling_fraction = float(payload["sampling_fraction"])
    log(f"sampling_fraction p*={sampling_fraction:.4f}")

    srf_kwargs = {"rho": RHO, "max_inner": MAX_INNER, "tol": TOL,
                  "max_outer": MAX_OUTER, "check_input": False}
    log(f"srf_kwargs={srf_kwargs}")

    log(f"loading similarity from {CACHE_PATH}")
    s = np.load(CACHE_PATH)
    log(f"  n={s.shape[0]}  dtype={s.dtype}")
    obs = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs)
    pool_fraction = _cv_pool_fraction(sampling_fraction, N_FOLDS, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(RANDOM_STATE, N_REPEATS, N_FOLDS, len(RANKS))
    splits = _entry_splits(obs, pool_fraction, N_FOLDS, split_seeds)
    log(f"prepared {len(splits)} entry-prekfold splits (p_pool={pool_fraction:.4f}, bounds={bounds})")
    del s

    # Build the job queue, skip anything already on disk
    jobs = []
    skipped = 0
    # Submit largest rank first (longest fits) so they start immediately
    for rank_idx, rank in sorted(enumerate(RANKS), key=lambda x: -x[1]):
        for rep, fold, train_mask, val_mask in splits:
            seed = int(fit_seeds[rep, fold, rank_idx])
            label = _label(rank, fold, seed)
            if (RESULTS_DIR / f"{label}.json").exists():
                skipped += 1
                continue
            jobs.append({
                "rank": rank, "fold": fold, "seed": seed, "rep": rep,
                "train_mask": train_mask, "val_mask": val_mask,
            })

    total = len(splits) * len(RANKS)
    log(f"=== {len(jobs)} new fits queued  ({skipped} already on disk, {total} total) ===")
    log(f"  ranks={RANKS}  max_outer={MAX_OUTER}  tol={TOL}  n_folds={N_FOLDS}")
    log(f"  parallel_fits={PARALLEL_FITS}  blas_threads={BLAS_THREADS}  cores={PARALLEL_FITS * BLAS_THREADS}")
    log(f"  output: {OUTPUT_DIR}")

    if not jobs:
        log("all fits already present on disk; nothing to do.")
        rows = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
        _write_summary(rows, SUMMARY_PATH)
        log(f"wrote summary: {SUMMARY_PATH}")
        return

    started = time.time()
    rows: list[dict] = []
    # Seed with already-completed results so the running summary stays correct
    for p in RESULTS_DIR.glob("*.json"):
        rows.append(json.loads(p.read_text()))

    with Parallel(n_jobs=PARALLEL_FITS, return_as="generator", verbose=10, backend="loky") as parallel:
        results = parallel(
            delayed(_fit_one)(
                CACHE_PATH, j["train_mask"], j["val_mask"], bounds,
                j["rank"], j["seed"], j["fold"],
                srf_kwargs, BLAS_THREADS, RESULTS_DIR, ESTIMATORS_DIR,
            )
            for j in jobs
        )
        for result in results:
            rows.append(result)
            _write_summary(rows, SUMMARY_PATH)
            tag = "resumed" if result.get("resumed") else "fit"
            log(f"  [{len(rows):>2d}/{total}] {tag}  rank={result['rank']:>3d}  fold={result['fold']}  "
                f"n_iter={result.get('n_iter')}  val_mse={result['val_mse']:.6e}  ({result['fit_time_s']:.0f}s)")

    elapsed = time.time() - started
    log(f"=== all {len(rows)} fits done in {elapsed:.0f}s = {elapsed/3600:.2f}h ===")

    _write_summary(rows, SUMMARY_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    log("=== per-rank summary ===")
    for _, row in summary.iterrows():
        log(f"  rank={int(row['rank']):>3d}: "
            f"val_mse={row['val_mse_mean']:.6e} ± {row['val_mse_sem']:.2e}  (n={int(row['count'])})")
    log(f"saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
