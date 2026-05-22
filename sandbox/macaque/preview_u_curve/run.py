"""Quick 2-rank macaque CV peek to test U-curve emergence early.

Designed to run *alongside* the full rank sweep — gets us a result on two
ranks (40 and 70) bracketing the suspected k_est=43 region before the
larger sweep completes.

Single fold per rank, max_outer=150, tol=1e-4 (early-stop allowed).
Same persistence pattern as the rank sweep:
- ``outputs/results/<label>.json`` — atomic per-fit JSON with metrics + history
- ``outputs/estimators/<label>.joblib`` — full SRF estimator
- resume on disk

Cores budget: only 8 cores free on prefrontal (rank sweep using 120).
So 2 fits × 4 BLAS threads = 8 cores, both fits parallel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
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

RANKS = [40, 70]
MAX_OUTER = 10     # quick preview — short outer
MAX_INNER = 150    # but fully-solve W each outer iter
TOL = 1e-4
RHO = 3.0
N_FOLDS = 5            # generate full 5-fold split structure
USE_FOLD = 0           # but only run fold=0
N_REPEATS = 1
RANDOM_STATE = 0
BLAS_THREADS = 8
PARALLEL_FITS = 2      # 2 × 8 = 16 cores (load avg has plenty of headroom — rank sweep is mem-bw-bound)


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
    if result_path.exists():
        return {"label": label, **json.loads(result_path.read_text()), "resumed": True}

    t0 = time.time()
    s_mm = np.load(cache_path, mmap_mode="r")
    with threadpool_limits(limits=blas_threads):
        train = np.full(s_mm.shape, np.nan, dtype=np.float64)
        train[train_mask] = np.asarray(s_mm[train_mask])
        est = SRF(
            rank=rank, bounds=bounds, missing_values=np.nan,
            random_state=seed, verbose=1, **srf_kwargs,
        )
        est.fit(train)
        n_iter = int(est.n_iter_) if hasattr(est, "n_iter_") else None
        del train
        s_hat = est.reconstruct()

        ii, jj = np.where(val_mask)
        upper = ii < jj
        ii, jj = ii[upper], jj[upper]
        s_vals = np.asarray(s_mm[ii, jj], dtype=np.float64)
        s_hat_vals = s_hat[ii, jj]
        finite = np.isfinite(s_vals) & np.isfinite(s_hat_vals)
        n_entries = int(finite.sum())
        residual = s_vals[finite] - s_hat_vals[finite]
        val_mse = float(np.mean(residual * residual)) if n_entries else float("nan")

    elapsed = time.time() - t0
    if hasattr(est, "_observed_mask"):
        est._observed_mask = None
    estimators_dir.mkdir(parents=True, exist_ok=True)
    estimator_path = estimators_dir / f"{label}.joblib"
    dump(est, estimator_path, compress=3)

    history = {}
    if hasattr(est, "history_") and est.history_:
        history = {k: [float(v) for v in vs] for k, vs in est.history_.items()}

    result = {
        "label": label, "rank": rank, "fold": fold, "seed": seed,
        "val_mse": val_mse, "n_entries_val": n_entries,
        "n_iter": n_iter, "fit_time_s": elapsed,
        "max_outer": srf_kwargs["max_outer"], "max_inner": srf_kwargs["max_inner"],
        "tol": srf_kwargs["tol"], "rho": srf_kwargs["rho"],
        "bounds": list(bounds),
        "estimator_path": str(estimator_path),
        "history": history,
    }
    _atomic_write_json(result_path, result)
    return {**result, "resumed": False}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ESTIMATORS_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(CV_JSON.read_text())
    sampling_fraction = float(payload["sampling_fraction"])
    log(f"sampling_fraction p*={sampling_fraction:.4f}")

    srf_kwargs = {"rho": RHO, "max_inner": MAX_INNER, "tol": TOL,
                  "max_outer": MAX_OUTER, "check_input": False}
    log(f"srf_kwargs={srf_kwargs}")

    log(f"loading {CACHE_PATH}")
    s = np.load(CACHE_PATH)
    obs = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs)
    pool_fraction = _cv_pool_fraction(sampling_fraction, N_FOLDS, s.shape[0])
    split_seeds, fit_seeds = _split_fit_seeds(RANDOM_STATE, N_REPEATS, N_FOLDS, len(RANKS))
    splits = _entry_splits(obs, pool_fraction, N_FOLDS, split_seeds)
    log(f"got {len(splits)} splits; using fold={USE_FOLD} only")
    del s

    # Pick only the one fold
    rep, fold, train_mask, val_mask = splits[USE_FOLD]
    assert fold == USE_FOLD

    jobs = []
    for rank_idx, rank in sorted(enumerate(RANKS), key=lambda x: -x[1]):
        seed = int(fit_seeds[rep, fold, rank_idx])
        label = _label(rank, fold, seed)
        if (RESULTS_DIR / f"{label}.json").exists():
            log(f"  skip {label}: already on disk")
            continue
        jobs.append({"rank": rank, "fold": fold, "seed": seed,
                     "train_mask": train_mask, "val_mask": val_mask})

    log(f"=== {len(jobs)} fits queued, parallel={PARALLEL_FITS}, blas={BLAS_THREADS} ===")
    if not jobs:
        return

    started = time.time()
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
            tag = "resumed" if result.get("resumed") else "fit"
            log(f"  {tag}  rank={result['rank']:>3}  fold={result['fold']}  "
                f"n_iter={result.get('n_iter')}  val_mse={result['val_mse']:.6e}  "
                f"({result['fit_time_s']:.0f}s)")

    elapsed = time.time() - started
    log(f"=== done in {elapsed:.0f}s = {elapsed/3600:.2f}h ===")

    # Compare to old CV val_mse at adjacent ranks
    old = json.loads(CV_JSON.read_text())["validations"]["5fold"]
    print()
    log("=== comparison vs old CV (max_outer=50, n_repeats=1) ===")
    for rank_str_int in [40, 50, 70, 80]:
        if rank_str_int in old["ranks"]:
            i = old["ranks"].index(rank_str_int)
            old_mse = old["val_mse_mean"][i]
            log(f"  old rank={rank_str_int}: val_mse={old_mse:.6e}")


if __name__ == "__main__":
    main()
