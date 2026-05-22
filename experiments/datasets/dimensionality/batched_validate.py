"""Run 5fold CV for a dataset on an explicit rank grid in ONE batched call.

Unlike the main validate pipeline (which loops one rank at a time, dispatching
only n_folds * n_repeats fits per call), this batches all (rank, fold, repeat)
combinations into a single `cross_val_score` call so joblib can saturate the
pool. Critical when n_folds * n_repeats is small (e.g. 5) and you want >5
workers actually busy.

Usage
-----
    python -m experiments.datasets.dimensionality.batched_validate \
        --dataset things_macaque22k \
        --ranks 10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
        --n-jobs 32 \
        --n-folds 5 \
        --n-repeats 1
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
from omegaconf import OmegaConf
from pysrf import cross_val_score

from ._validate import CV_PROTOCOL_VERSION, _record_from_curve

log = logging.getLogger("batched_validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

DEFAULT_SRF_KWARGS = {
    "rho": 3.0,
    "max_inner": 30,
    "tol": 0.0,
    "max_outer": 200,
    "check_input": False,
}


def main() -> None:
    args = _parse_args()
    _limit_thread_env()

    output_dir = Path(args.output_dir).resolve()
    cache_path = output_dir / "cache" / f"{args.dataset}.npy"
    estimate_path = output_dir / args.dataset / "coherence_estimate.json"
    cv_path = output_dir / args.dataset / "cross_validation.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached similarity at {cache_path}. Run estimate first.")
    if not estimate_path.exists():
        raise FileNotFoundError(f"No coherence_estimate.json at {estimate_path}. Run estimate first.")

    similarity = np.load(cache_path)
    estimate = json.loads(estimate_path.read_text())["estimate"]
    sampling_fraction = float(estimate["sampling_fraction"])
    log.info(f"loaded n={similarity.shape[0]} similarity for {args.dataset}, p*={sampling_fraction:.3f}")

    ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()})
    total_fits = len(ranks) * args.n_folds * args.n_repeats
    parallel = min(total_fits, args.n_jobs)
    log.info(
        f"ranks={ranks}  n_folds={args.n_folds}  n_repeats={args.n_repeats}  "
        f"total_fits={total_fits}  parallel_cap={parallel}"
    )

    variant = {
        "name": args.variant,
        "n_folds": args.n_folds,
        "n_repeats": args.n_repeats,
        "random_state": args.random_state,
    }
    params = OmegaConf.create({"srf_kwargs": DEFAULT_SRF_KWARGS, "strategy": "fixed"})

    started = time.time()
    log.info(f"dispatching {total_fits} fits to joblib (n_jobs={args.n_jobs})")
    multi_curve = cross_val_score(
        similarity,
        ranks=ranks,
        sampling_fraction=sampling_fraction,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        srf_kwargs=DEFAULT_SRF_KWARGS,
    )
    elapsed = time.time() - started
    log.info(f"all fits complete in {elapsed:.1f}s")

    multi_curve = multi_curve.assign(dataset=args.dataset, cv_variant=variant["name"])
    record = _record_from_curve(
        curve=multi_curve,
        target_ranks=ranks,
        sampling_fraction=sampling_fraction,
        variant=variant,
        params=params,
        n_jobs=args.n_jobs,
        started=started,
        resumed_from_json=False,
    )

    payload = json.loads(cv_path.read_text()) if cv_path.exists() else {
        "dataset": args.dataset,
        "n": int(similarity.shape[0]),
        "rank_estimate": int(estimate["rank"]),
        "sampling_fraction": sampling_fraction,
        "validations": {},
        "primary_variant": variant["name"],
    }
    payload.setdefault("validations", {})[variant["name"]] = record
    payload["updated_at_unix"] = time.time()
    _write_json_atomic(cv_path, payload)
    log.info(f"wrote {cv_path}  argmin_rank={record['argmin_rank']}  one_se_rank={record['one_se_rank']}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _limit_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ranks", required=True, help="Comma-separated rank list (e.g. 10,15,20,...,100).")
    parser.add_argument("--variant", default="5fold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
