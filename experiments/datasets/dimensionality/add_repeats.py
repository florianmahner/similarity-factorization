"""Add MORE CV repeats to an existing 5fold/10fold block without redoing the existing ones.

Reads the existing block from outputs/<dataset>/cross_validation.json, keeps all
existing per-repeat-per-fold val_mse rows, runs N additional repeats with a fresh
random_state (so the new fold splits are independent), then writes the merged
result back, updating n_repeats and recomputing argmin / one_se.

Usage:
    python -m experiments.datasets.dimensionality.add_repeats \
        --dataset things_behavior \
        --variant 5fold \
        --add-repeats 15 \
        --new-seed 1000 \
        --n-jobs 16
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

from ._validate import _curve_from_block, _record_from_curve

log = logging.getLogger("add_repeats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main() -> None:
    args = _parse_args()
    _limit_thread_env()

    output_dir = Path(args.output_dir).resolve()
    cv_path = output_dir / args.dataset / "cross_validation.json"
    cache_path = output_dir / "cache" / f"{args.dataset}.npy"

    if not cv_path.exists():
        raise FileNotFoundError(f"Missing CV: {cv_path}")
    payload = json.loads(cv_path.read_text())
    block = payload["validations"][args.variant]

    n_folds = int(block["params"]["n_folds"])
    n_repeats_old = int(block["params"]["n_repeats"])
    sampling_fraction = float(block["params"]["sampling_fraction"])
    srf_kwargs = dict(block["params"]["srf_kwargs"])
    ranks = [int(r) for r in block["ranks"]]

    log.info(
        f"dataset={args.dataset} variant={args.variant} ranks={ranks} "
        f"n_folds={n_folds} n_repeats_old={n_repeats_old}  "
        f"adding {args.add_repeats} new repeats (seed={args.new_seed})"
    )

    similarity = np.load(cache_path)
    log.info(f"loaded similarity n={similarity.shape[0]}")

    existing_curve = _curve_from_block(block, expected_per_rank=n_folds * n_repeats_old)
    log.info(f"existing rows: {len(existing_curve)}  unique reps: {sorted(existing_curve['rep'].unique())}")

    t0 = time.time()
    new_curve = cross_val_score(
        similarity,
        ranks=ranks,
        sampling_fraction=sampling_fraction,
        n_folds=n_folds,
        n_repeats=args.add_repeats,
        random_state=args.new_seed,
        n_jobs=args.n_jobs,
        srf_kwargs=srf_kwargs,
    )
    elapsed = time.time() - t0
    log.info(f"new {args.add_repeats} repeats done in {elapsed:.1f}s  rows={len(new_curve)}")

    # Renumber the new reps to start AFTER existing ones (5..5+add-1 if old was 0..4)
    new_curve = new_curve.copy()
    new_curve["rep"] = new_curve["rep"].astype(int) + n_repeats_old
    new_curve["dataset"] = args.dataset
    new_curve["cv_variant"] = args.variant

    merged = pd.concat([existing_curve, new_curve], ignore_index=True)
    n_repeats_new = n_repeats_old + args.add_repeats
    log.info(f"merged rows: {len(merged)}  reps: {sorted(merged['rep'].unique())}")

    variant_dict = {
        "name": args.variant,
        "n_folds": n_folds,
        "n_repeats": n_repeats_new,
        "random_state": int(block["params"]["random_state"]),
    }
    params = OmegaConf.create({"srf_kwargs": srf_kwargs, "strategy": "fixed"})

    record = _record_from_curve(
        curve=merged,
        target_ranks=ranks,
        sampling_fraction=sampling_fraction,
        variant=variant_dict,
        params=params,
        n_jobs=args.n_jobs,
        started=t0,
        resumed_from_json=True,
    )
    payload["validations"][args.variant] = record
    payload["updated_at_unix"] = time.time()
    _write_json_atomic(cv_path, payload)
    log.info(
        f"wrote {cv_path}  argmin={record['argmin_rank']}  one_se={record['one_se_rank']}  "
        f"n_repeats={n_repeats_new}"
    )


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
    parser.add_argument("--variant", default="5fold")
    parser.add_argument("--add-repeats", type=int, default=15)
    parser.add_argument("--new-seed", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
