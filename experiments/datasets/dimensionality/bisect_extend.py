"""Extend an existing 5fold CV curve with bisection refinement.

Reuses the same `cross_val_score` machinery as the main validate pipeline,
so the new ranks are directly comparable to the existing curve.

Workflow per pass:
  1. Evaluate any initial "new" ranks the user wants (e.g. 250,300,350,400).
  2. Find current argmin in the merged curve.
  3. Bisect: midpoint(prev_neighbor, argmin) and midpoint(argmin, next_neighbor).
  4. Stop when both adjacent gaps <= --min-gap.

Writes results back to outputs/<dataset>/cross_validation.json incrementally.

Usage
-----
    python -m experiments.datasets.dimensionality.bisect_extend \
        --dataset swow \
        --new-seeds 250,300,350,400 \
        --min-gap 5 \
        --max-rounds 8
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

from . import io as _io
from ._loader import load_similarity
from ._validate import (
    CV_PROTOCOL_VERSION,
    _curve_from_block,
    _record_from_curve,
)

log = logging.getLogger("bisect_extend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main() -> None:
    args = _parse_args()
    _limit_thread_env()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cv_path = output_dir / args.dataset / "cross_validation.json"
    payload = json.loads(cv_path.read_text())
    block = payload["validations"][args.variant]

    n_folds = int(block["params"]["n_folds"])
    n_repeats = int(block["params"]["n_repeats"])
    sampling_fraction = float(block["params"]["sampling_fraction"])
    random_state = int(block["params"]["random_state"])
    srf_kwargs = dict(block["params"]["srf_kwargs"])
    variant = {"name": args.variant, "n_folds": n_folds, "n_repeats": n_repeats, "random_state": random_state}
    params = OmegaConf.create({"srf_kwargs": srf_kwargs, "strategy": "fixed"})

    similarity = _load_similarity_for(args.dataset, project_root, output_dir)
    log.info(f"loaded n={similarity.shape[0]} similarity for {args.dataset}")

    curve = _curve_from_block(block, n_folds * n_repeats)
    log.info(f"existing curve: {sorted(curve['rank'].astype(int).unique().tolist())}")

    new_seeds = sorted({int(r) for r in args.new_seeds.split(",") if r.strip()})
    already = set(curve["rank"].astype(int).unique().tolist())
    to_eval = [r for r in new_seeds if r not in already]
    target_ranks = sorted(set(already).union(new_seeds))

    started = time.time()
    if to_eval:
        log.info(f"round 0: evaluating new seeds {to_eval}")
        curve = _evaluate_and_checkpoint(
            curve=curve, to_eval=to_eval, target_ranks=target_ranks,
            dataset=args.dataset, similarity=similarity,
            sampling_fraction=sampling_fraction, variant=variant, params=params,
            n_jobs=args.n_jobs, started=started, cv_path=cv_path, payload=payload,
        )
    else:
        log.info("round 0: all initial seeds already evaluated")

    for round_idx in range(1, args.max_rounds + 1):
        evaluated = sorted(curve["rank"].astype(int).unique().tolist())
        bisect_pair = _bisect_around_argmin(curve, evaluated, args.min_gap)
        if bisect_pair is None:
            log.info(f"converged: both adjacent gaps <= {args.min_gap}")
            break
        to_eval = sorted(set(bisect_pair) - set(evaluated))
        if not to_eval:
            log.info("nothing new to evaluate (rounding collision)")
            break
        log.info(f"round {round_idx}: bisecting around argmin -> {to_eval}")
        target_ranks = sorted(set(evaluated).union(to_eval))
        curve = _evaluate_and_checkpoint(
            curve=curve, to_eval=to_eval, target_ranks=target_ranks,
            dataset=args.dataset, similarity=similarity,
            sampling_fraction=sampling_fraction, variant=variant, params=params,
            n_jobs=args.n_jobs, started=started, cv_path=cv_path, payload=payload,
        )

    final_argmin, final_one_se = _argmin_and_one_se(curve)
    log.info(f"done: argmin={final_argmin}, one_se={final_one_se}, total ranks={len(curve['rank'].unique())}")


def _evaluate_and_checkpoint(
    *, curve, to_eval, target_ranks, dataset, similarity, sampling_fraction,
    variant, params, n_jobs, started, cv_path, payload,
):
    """Batch all ranks in this round into ONE cross_val_score call.

    Joblib then dispatches len(to_eval) * n_folds * n_repeats fits in parallel,
    instead of the loop-per-rank approach which only ever has n_folds*n_repeats
    fits in flight at once.
    """
    n_folds = int(variant["n_folds"])
    n_repeats = int(variant["n_repeats"])
    expected_parallel = len(to_eval) * n_folds * n_repeats
    log.info(
        f"  batched: ranks={to_eval}  fits={expected_parallel}  n_jobs={n_jobs}  "
        f"(parallel cap = min(fits, n_jobs))"
    )
    multi_curve = cross_val_score(
        similarity,
        ranks=list(to_eval),
        sampling_fraction=sampling_fraction,
        n_folds=n_folds,
        n_repeats=n_repeats,
        random_state=int(variant["random_state"]),
        n_jobs=n_jobs,
        srf_kwargs=dict(params.srf_kwargs),
    )
    multi_curve = multi_curve.assign(dataset=dataset, cv_variant=variant["name"])
    keep = curve[~curve["rank"].astype(int).isin([int(r) for r in to_eval])]
    curve = pd.concat([keep, multi_curve], ignore_index=True)

    record = _record_from_curve(
        curve=curve,
        target_ranks=target_ranks,
        sampling_fraction=sampling_fraction,
        variant=variant,
        params=params,
        n_jobs=n_jobs,
        started=started,
        resumed_from_json=True,
    )
    payload["validations"][variant["name"]] = record
    payload["updated_at_unix"] = time.time()
    _write_json_atomic(cv_path, payload)
    return curve


def _bisect_around_argmin(curve: pd.DataFrame, evaluated: list[int], min_gap: int) -> tuple[int, int] | None:
    stats = curve.groupby("rank")["val_mse"].mean().sort_index()
    argmin_rank = int(stats.idxmin())
    sorted_ranks = sorted(evaluated)
    i = sorted_ranks.index(argmin_rank)
    lower = sorted_ranks[i - 1] if i > 0 else None
    upper = sorted_ranks[i + 1] if i < len(sorted_ranks) - 1 else None

    new_ranks: list[int] = []
    if lower is not None and (argmin_rank - lower) > min_gap:
        new_ranks.append((lower + argmin_rank) // 2)
    if upper is not None and (upper - argmin_rank) > min_gap:
        new_ranks.append((argmin_rank + upper) // 2)
    if not new_ranks:
        return None
    return tuple(new_ranks)  # type: ignore[return-value]


def _argmin_and_one_se(curve: pd.DataFrame) -> tuple[int, int | None]:
    stats = curve.groupby("rank")["val_mse"].agg(["mean", "std", "count"]).sort_index()
    mean = stats["mean"].to_numpy()
    sem = stats["std"].to_numpy() / np.sqrt(np.maximum(stats["count"].to_numpy(), 1))
    argmin_idx = int(np.nanargmin(mean))
    argmin_rank = int(stats.index[argmin_idx])
    threshold = mean[argmin_idx] + sem[argmin_idx]
    candidates = [int(r) for r, m in zip(stats.index, mean) if m <= threshold]
    one_se = min(candidates) if candidates else None
    return argmin_rank, one_se


def _load_similarity_for(dataset: str, project_root: Path, output_dir: Path) -> np.ndarray:
    # Reuse loader expectations: needs a DictConfig-like dataset entry. We fake it
    # by reading the cached similarity directly since it's always written by previous runs.
    cache_path = output_dir / "cache" / f"{dataset}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    raise FileNotFoundError(f"No cached similarity at {cache_path}. Run estimate/validate first to populate.")


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
    parser.add_argument("--new-seeds", default="", help="Comma-separated ranks to evaluate first (e.g. 250,300,350,400).")
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--project-root", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization")
    parser.add_argument("--output-dir", default="/data/labshare/_stachelschwein/LOCAL/fmahner/similarity-factorization/experiments/datasets/dimensionality/outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main()
