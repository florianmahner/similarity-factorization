"""Resumable batched CV: per-fit JSON dumps + skip-existing logic.

Each (rank, fold, repeat) is its own job, persisted as
``outputs/<dataset>/cv_per_fit/rank{r:03d}_fold{f}_repeat{rep}_seed{s}.json``
the moment it completes (atomic via tmp+rename). On re-launch, any label
whose JSON already exists is skipped; only new combos run.

After all fits land, ``cross_validation.json`` is rebuilt from the FULL
set of per-fit JSONs using the same ``_record_from_curve`` aggregator as
``batched_validate.py``, so its schema is identical and downstream tools
(``plot_cv.py``, ``plot_dimensionality_cv/plot.py``) need no changes.

Three workflows this enables cleanly:

* **First run**:
  ``cv_resumable --dataset X --ranks 10,20,30,40 ...``  → 4×n_folds×n_repeats fits

* **Extend grid** (e.g. min looks edge-y):
  ``cv_resumable --dataset X --ranks 10,20,30,40,50,60``  → only ranks 50, 60 fit;
  the cross_validation.json is rebuilt with all 6 ranks.

* **Refine around minimum**:
  ``cv_resumable --dataset X --ranks 15,20,25,30,35``  → only 15, 25, 35 fit
  (20, 30 already exist); CV JSON rebuilt with all 5 ranks.

Caveats
-------
Seeds diverge from ``cross_val_score``: this script uses
hash(``random_state, rank, fold, repeat``) so that adding/removing other
ranks from the grid does not shift the seeds of existing combos.
``batched_validate.py`` (which calls ``cross_val_score`` directly) does
not share these seeds — do not mix the two scripts' outputs in the same
``cv_per_fit/`` directory.

All per-fit JSONs in a directory must share the same SRF kwargs. If you
change ``max_outer`` / ``max_inner`` / ``tol`` mid-experiment, clear or
relocate ``cv_per_fit/`` first; otherwise the aggregator emits a warning
and the cross_validation.json's params block reflects only the LAST run's
kwargs (which would misrepresent the older entries).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from pysrf import SRF
from pysrf._common import observation_mask
from pysrf.cross_validation import (
    _cv_pool_fraction,
    _entry_splits,
    _observed_bounds,
    _split_fit_seeds,
)

from ._validate import _record_from_curve

log = logging.getLogger("cv_resumable")


def _fit_seed(random_state: int, rank: int, fold: int, repeat: int) -> int:
    """Stable per-(rank, fold, repeat) seed, independent of which other ranks
    are in the active --ranks list. Adding a new rank to the grid does not
    shift seeds of existing combos."""
    ss = np.random.SeedSequence([int(random_state), int(rank), int(fold), int(repeat)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _label(rank: int, fold: int, repeat: int, seed: int) -> str:
    return f"rank{rank:03d}_fold{fold}_repeat{repeat}_seed{seed}"


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
    fold: int,
    repeat: int,
    seed: int,
    srf_kwargs: dict,
    blas_threads: int,
    per_fit_dir: Path,
    save_estimators: bool,
    estimator_dir: Path | None,
) -> dict:
    label = _label(rank, fold, repeat, seed)
    result_path = per_fit_dir / f"{label}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

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
        if n_entries == 0:
            val_mse = float("nan")
        else:
            residual = s_vals[finite] - s_hat_vals[finite]
            val_mse = float(np.mean(residual * residual))

    history = {}
    if hasattr(est, "history_") and est.history_:
        history = {k: [float(v) for v in vs] for k, vs in est.history_.items()}

    result = {
        "label": label,
        "rank": int(rank), "fold": int(fold), "repeat": int(repeat), "seed": int(seed),
        "val_mse": val_mse,
        "n_entries_val": n_entries,
        "n_iter": n_iter,
        "fit_time_s": time.time() - t0,
        "srf_kwargs": srf_kwargs,
        "bounds": list(bounds),
        "history": history,
    }
    if save_estimators:
        if estimator_dir is None:
            raise ValueError("estimator_dir is required when save_estimators=True")
        estimator_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(est, "_observed_mask"):
            est._observed_mask = None
        estimator_path = estimator_dir / f"{label}.joblib"
        dump(est, estimator_path, compress=3)
        result["estimator_path"] = str(estimator_path)

    _atomic_write_json(result_path, result)
    return result


def _aggregate_block(
    per_fit_dir: Path,
    ranks: list[int],
    n_folds: int,
    n_repeats: int,
    sampling_fraction: float,
    srf_kwargs: dict,
    random_state: int,
    variant_name: str,
    n_jobs: int,
    started: float,
) -> dict | None:
    """Build a cross_validation block from per-fit JSONs (same schema as
    ``batched_validate._record_from_curve``)."""
    rows = []
    kwargs_seen = set()
    for p in sorted(per_fit_dir.glob("*.json")):
        d = json.loads(p.read_text())
        rows.append({
            "rep": int(d["repeat"]),
            "fold": int(d["fold"]),
            "rank": int(d["rank"]),
            "val_mse": float(d["val_mse"]),
        })
        kwargs_seen.add(json.dumps(d.get("srf_kwargs", {}), sort_keys=True))
    if not rows:
        return None
    if len(kwargs_seen) > 1:
        log.warning(
            f"Per-fit JSONs in {per_fit_dir} mix {len(kwargs_seen)} different "
            f"srf_kwargs configs; the rebuilt cross_validation.json's params "
            f"block will only reflect the current run's kwargs.")

    curve = pd.DataFrame(rows)
    variant = {
        "name": variant_name, "n_folds": n_folds, "n_repeats": n_repeats,
        "random_state": random_state,
    }
    params = OmegaConf.create({"srf_kwargs": srf_kwargs, "strategy": "fixed"})
    return _record_from_curve(
        curve=curve, target_ranks=ranks,
        sampling_fraction=sampling_fraction,
        variant=variant, params=params, n_jobs=n_jobs,
        started=started, resumed_from_json=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ranks", required=True,
                        help="Comma-separated rank list (e.g. '10,20,30,40,50')")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--variant", default="5fold")
    parser.add_argument("--max-outer", type=int, default=200)
    parser.add_argument("--max-inner", type=int, default=30)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs",
    )
    parser.add_argument(
        "--cache-dataset",
        default=None,
        help="Dataset/cache name for outputs/cache/<name>.npy. Defaults to --dataset.",
    )
    parser.add_argument(
        "--per-fit-dir",
        type=Path,
        default=None,
        help="Directory for atomic per-fit JSONs. Defaults to <dataset>/cv_per_fit.",
    )
    parser.add_argument(
        "--save-estimators",
        action="store_true",
        help="Persist each completed SRF estimator as a joblib file for inspection.",
    )
    parser.add_argument(
        "--estimator-dir",
        type=Path,
        default=None,
        help="Directory for saved estimators. Defaults to <dataset>/cv_estimators_<variant>.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, str(args.blas_threads))

    output_dir = Path(args.output_dir).resolve()
    ds_dir = output_dir / args.dataset
    cache_dataset = args.cache_dataset or args.dataset
    per_fit_dir = args.per_fit_dir.resolve() if args.per_fit_dir else ds_dir / "cv_per_fit"
    estimator_dir = (
        args.estimator_dir.resolve()
        if args.estimator_dir
        else ds_dir / f"cv_estimators_{args.variant}"
    )
    cache_path = output_dir / "cache" / f"{cache_dataset}.npy"
    estimate_path = ds_dir / "coherence_estimate.json"
    cv_path = ds_dir / "cross_validation.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached similarity at {cache_path} — run estimate first.")
    if not estimate_path.exists():
        raise FileNotFoundError(f"No coherence_estimate.json at {estimate_path} — run estimate first.")

    estimate = json.loads(estimate_path.read_text())["estimate"]
    sampling_fraction = float(estimate["sampling_fraction"])
    ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()})

    srf_kwargs = {
        "rho": float(args.rho), "max_inner": int(args.max_inner),
        "tol": float(args.tol), "max_outer": int(args.max_outer),
        "check_input": False,
    }

    log.info(f"dataset={args.dataset}  ranks={ranks}")
    log.info(f"n_folds={args.n_folds} n_repeats={args.n_repeats} random_state={args.random_state}")
    log.info(f"srf_kwargs={srf_kwargs}")
    log.info(f"cache_path={cache_path}")
    log.info(f"per_fit_dir={per_fit_dir}")
    if args.save_estimators:
        log.info(f"estimator_dir={estimator_dir}")

    # Load full S once to compute observation mask + bounds; mmap inside workers.
    s = np.load(cache_path)
    obs = observation_mask(s, np.nan)
    bounds = _observed_bounds(s, obs)
    pool_fraction = _cv_pool_fraction(sampling_fraction, args.n_folds, s.shape[0])
    n_items = int(s.shape[0])
    del s

    # split_seeds depend only on n_repeats (stable across rank list changes).
    # We discard the rank-indexed fit_seeds and use stable hash-based ones instead.
    split_seeds, _ = _split_fit_seeds(args.random_state, args.n_repeats, args.n_folds, 1)
    splits = _entry_splits(obs, pool_fraction, args.n_folds, split_seeds)
    log.info(f"prepared {len(splits)} (rep, fold) splits  p_pool={pool_fraction:.4f}  bounds={bounds}")

    # Build job list, longest-first, skip-if-exists.
    per_fit_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    skipped = 0
    for rank in sorted(ranks, reverse=True):
        for rep, fold, train_mask, val_mask in splits:
            seed = _fit_seed(args.random_state, rank, fold, rep)
            label = _label(rank, fold, rep, seed)
            if (per_fit_dir / f"{label}.json").exists():
                skipped += 1
                continue
            jobs.append({
                "rank": rank, "fold": fold, "repeat": rep, "seed": seed,
                "train_mask": train_mask, "val_mask": val_mask,
            })

    total = len(splits) * len(ranks)
    log.info(f"=== {len(jobs)} new fits queued  ({skipped} already on disk, {total} total target) ===")
    log.info(f"parallel_fits={args.n_jobs}  blas_threads={args.blas_threads}  cores~={args.n_jobs * args.blas_threads}")

    started = time.time()
    if jobs:
        Parallel(n_jobs=args.n_jobs, verbose=10, backend="loky")(
            delayed(_fit_one)(
                cache_path, j["train_mask"], j["val_mask"], bounds,
                j["rank"], j["fold"], j["repeat"], j["seed"],
                srf_kwargs, args.blas_threads, per_fit_dir,
                args.save_estimators, estimator_dir,
            )
            for j in jobs
        )
    elapsed = time.time() - started
    log.info(f"=== all fits done in {elapsed:.0f}s = {elapsed/3600:.2f}h ===")

    # Rebuild cross_validation.json from the FULL set of per-fit JSONs.
    record = _aggregate_block(
        per_fit_dir, ranks, args.n_folds, args.n_repeats, sampling_fraction,
        srf_kwargs, args.random_state, args.variant, args.n_jobs, started,
    )
    if record is None:
        log.error("No per-fit JSONs found; cross_validation.json NOT written.")
        return

    payload = json.loads(cv_path.read_text()) if cv_path.exists() else {
        "dataset": args.dataset,
        "n": n_items,
        "rank_estimate": int(estimate.get("rank", 0)),
        "sampling_fraction": sampling_fraction,
        "validations": {},
        "primary_variant": args.variant,
    }
    payload.setdefault("validations", {})[args.variant] = record
    payload["updated_at_unix"] = time.time()
    _atomic_write_json(cv_path, payload)
    log.info(
        f"wrote {cv_path}  argmin_rank={record['argmin_rank']}  "
        f"one_se_rank={record['one_se_rank']}  status={record['status']}  "
        f"completed_ranks={record['completed_ranks']}"
    )


if __name__ == "__main__":
    main()
