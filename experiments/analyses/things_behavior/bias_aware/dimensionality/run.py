"""Rank selection on the bias-aware THINGS-behavior matrix.

Cross-validation with the same protocol as the normal-RSM CV
(``experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json``):

* 5-fold x 10-repeat entry CV (pysrf.cross_val_score)
* matched sampling_fraction
* srf_kwargs = {rho: 3.0, max_outer: 200, max_inner: 30, tol: 0.0}

Writes cross_validation.json with the argmin / one-SE ranks.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from pysrf import cross_val_score

from experiments.analyses.things_behavior.bias_aware._matrix import (
    ALPHA,
    PRIOR_STRENGTH,
    WEIGHT_MODE,
    build_bias_aware_things_matrix,
    load_things_train_triplets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# Match the normal-RSM CV (experiments/datasets/dimensionality/outputs/things_behavior/cross_validation.json).
SRF_KWARGS = {
    "rho": 3.0,
    "max_outer": 200,
    "max_inner": 30,
    "tol": 0.0,
}
# Default rank grid (matches what the main CV expanded to).
RANK_GRID = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 150]
# Same sampling fraction as the main CV picked (Bs-style entry CV uses this).
SAMPLING_FRACTION = 0.6493759738544267
N_FOLDS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=str, default=",".join(str(r) for r in RANK_GRID))
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--sampling-fraction", type=float, default=SAMPLING_FRACTION)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--out-name",
        type=str,
        default="cross_validation",
        help="Basename for the output JSON. When sharding across hosts, set "
             "different names per host (e.g. partial_<hostname>) to avoid races.",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=PRIOR_STRENGTH,
        help=f"Bias-aware lambda (default {PRIOR_STRENGTH}). Smaller = milder.",
    )
    return parser.parse_args()


def _one_se_rank(ranks: list[int], mean: np.ndarray, sem: np.ndarray, argmin_idx: int) -> int:
    threshold = float(mean[argmin_idx] + sem[argmin_idx])
    for r, m in zip(ranks, mean):
        if np.isfinite(m) and m <= threshold:
            return int(r)
    return int(ranks[argmin_idx])


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()})

    log.info("=== CV on bias-aware matrix ===")
    log.info("ranks=%s  n_folds=%d  n_repeats=%d  sampling_fraction=%.6f",
             ranks, args.n_folds, args.n_repeats, args.sampling_fraction)
    log.info("srf_kwargs=%s", SRF_KWARGS)
    log.info("matrix: alpha=%s, weight_mode=%s, lambda=%s", ALPHA, WEIGHT_MODE, args.prior_strength)

    log.info("Loading triplets ...")
    triplets = load_things_train_triplets()
    log.info("  n_triplets=%d", len(triplets))

    log.info("Building bias-aware matrix ...")
    M = build_bias_aware_things_matrix(
        triplets, n_objects=1854, prior_strength=args.prior_strength, verbose=True,
    )
    similarity = M.similarity

    rank_means = {}
    rank_sems = {}
    rank_scores = {}
    started = time.time()

    for rank in ranks:
        t0 = time.time()
        log.info("rank=%d  (%d of %d) ...", rank, ranks.index(rank) + 1, len(ranks))
        curve = cross_val_score(
            similarity,
            ranks=[rank],
            sampling_fraction=args.sampling_fraction,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=args.n_jobs,
            srf_kwargs=SRF_KWARGS,
        )
        # cross_val_score returns a DataFrame with 'val_mse' rows
        vals = curve["val_mse"].to_numpy(dtype=float)
        rank_means[rank] = float(np.mean(vals))
        rank_sems[rank] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rank_scores[rank] = vals.tolist()
        elapsed = time.time() - t0
        log.info("  rank=%d  val_mse=%.4e +/- %.2e  (%.1fs, total=%.1fs)",
                 rank, rank_means[rank], rank_sems[rank], elapsed, time.time() - started)

        # Checkpoint every rank
        means_arr = np.array([rank_means[r] for r in ranks if r in rank_means])
        sems_arr = np.array([rank_sems[r] for r in ranks if r in rank_means])
        completed = [r for r in ranks if r in rank_means]
        argmin_idx = int(np.nanargmin(means_arr))
        payload = {
            "matrix": {
                "kind": "bias_aware_fisher",
                "alpha": ALPHA,
                "weight_mode": WEIGHT_MODE,
                "prior_strength": float(args.prior_strength),
                "n_objects": 1854,
            },
            "params": {
                "n_folds": args.n_folds,
                "n_repeats": args.n_repeats,
                "sampling_fraction": float(args.sampling_fraction),
                "srf_kwargs": SRF_KWARGS,
                "random_state": RANDOM_STATE,
            },
            "ranks": completed,
            "val_mse_mean": means_arr.tolist(),
            "val_mse_sem": sems_arr.tolist(),
            "scores": {str(r): rank_scores[r] for r in completed},
            "argmin_rank": int(completed[argmin_idx]),
            "one_se_rank": _one_se_rank(completed, means_arr, sems_arr, argmin_idx),
            "status": "complete" if set(completed) == set(ranks) else "partial",
            "runtime_sec": time.time() - started,
        }
        (OUTPUT_DIR / f"{args.out_name}.json").write_text(json.dumps(payload, indent=2))

    log.info("=== DONE in %.1fs ===", time.time() - started)
    log.info("argmin_rank=%d  one_se_rank=%d",
             payload["argmin_rank"], payload["one_se_rank"])
    log.info("Wrote %s", OUTPUT_DIR / f"{args.out_name}.json")


if __name__ == "__main__":
    main()
