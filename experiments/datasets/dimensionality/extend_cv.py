"""Extend an existing cross_validation.json with additional ranks.

Loads the cached similarity matrix at
``outputs/cache/<dataset>.npy``, runs ``pysrf.cross_val_score`` at the
requested new ranks using the same protocol as the existing run, then
merges the results back into ``outputs/<dataset>/cross_validation.json``
preserving the original schema.

Usage:
    poetry run python -m experiments.datasets.dimensionality.extend_cv \\
        --dataset clip_vit_l14_sigma0.5 \\
        --ranks 110,120,140,160,200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pysrf import cross_val_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="e.g. clip_vit_l14_sigma0.5")
    p.add_argument("--ranks", required=True, help="comma-separated new ranks")
    p.add_argument("--variant", default="5fold", help="which variant block to extend")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def _one_se_rank(ranks: list[int], mean: np.ndarray, sem: np.ndarray, argmin_idx: int) -> int:
    threshold = float(mean[argmin_idx] + sem[argmin_idx])
    for r, m in zip(ranks, mean):
        if np.isfinite(m) and m <= threshold:
            return int(r)
    return int(ranks[argmin_idx])


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    new_ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()})

    cv_path = OUTPUT_ROOT / dataset / "cross_validation.json"
    cache_path = OUTPUT_ROOT / "cache" / f"{dataset}.npy"

    log.info("=== extend_cv ===")
    log.info("dataset=%s  variant=%s  new_ranks=%s", dataset, args.variant, new_ranks)
    log.info("cv_path=%s  cache=%s", cv_path, cache_path)

    payload = json.loads(cv_path.read_text())
    variant_block = payload["validations"][args.variant]
    existing_ranks = list(variant_block["ranks"])
    existing_mean = list(variant_block["val_mse_mean"])
    existing_sem = list(variant_block["val_mse_sem"])
    existing_count = list(variant_block["val_mse_count"])
    existing_scores = dict(variant_block["scores"])
    srf_kwargs = variant_block["params"]["srf_kwargs"]
    sampling_fraction = float(variant_block["params"]["sampling_fraction"])
    n_folds = int(variant_block["params"]["n_folds"])
    n_repeats = int(variant_block["params"]["n_repeats"])
    random_state = int(variant_block["params"]["random_state"])

    overlap = set(new_ranks) & set(existing_ranks)
    if overlap:
        log.warning("Dropping already-completed ranks: %s", sorted(overlap))
        new_ranks = [r for r in new_ranks if r not in overlap]
    if not new_ranks:
        log.info("Nothing to do.")
        return

    log.info("Loading cached similarity ...")
    similarity = np.load(cache_path)
    log.info("  shape=%s", similarity.shape)
    log.info("srf_kwargs=%s  sampling_fraction=%.6f  n_folds=%d  n_repeats=%d",
             srf_kwargs, sampling_fraction, n_folds, n_repeats)

    started = time.time()
    for rank in new_ranks:
        t0 = time.time()
        log.info("rank=%d ...", rank)
        curve = cross_val_score(
            similarity,
            ranks=[rank],
            sampling_fraction=sampling_fraction,
            n_folds=n_folds,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=args.n_jobs,
            srf_kwargs=srf_kwargs,
        )
        vals = curve["val_mse"].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        log.info("  rank=%d  val_mse=%.4e +/- %.2e  (%.1fs, total=%.1fs)",
                 rank, mean, sem, time.time() - t0, time.time() - started)

        existing_ranks.append(int(rank))
        existing_mean.append(mean)
        existing_sem.append(sem)
        existing_count.append(int(len(vals)))
        existing_scores[str(rank)] = vals.tolist()

        # Sort everything by rank for clean JSON ordering
        order = sorted(range(len(existing_ranks)), key=lambda i: existing_ranks[i])
        existing_ranks = [existing_ranks[i] for i in order]
        existing_mean = [existing_mean[i] for i in order]
        existing_sem = [existing_sem[i] for i in order]
        existing_count = [existing_count[i] for i in order]

        mean_arr = np.array(existing_mean)
        sem_arr = np.array(existing_sem)
        argmin_idx = int(np.nanargmin(mean_arr))
        variant_block["ranks"] = existing_ranks
        variant_block["completed_ranks"] = existing_ranks
        variant_block["target_ranks"] = sorted(set(variant_block.get("target_ranks", [])) | {int(rank)})
        variant_block["val_mse_mean"] = existing_mean
        variant_block["val_mse_sem"] = existing_sem
        variant_block["val_mse_count"] = existing_count
        variant_block["scores"] = existing_scores
        variant_block["argmin_rank"] = int(existing_ranks[argmin_idx])
        variant_block["one_se_rank"] = _one_se_rank(existing_ranks, mean_arr, sem_arr, argmin_idx)

        cv_path.write_text(json.dumps(payload, indent=2))
        log.info("  checkpoint saved (now argmin=%d, one_se=%d)",
                 variant_block["argmin_rank"], variant_block["one_se_rank"])

    log.info("=== DONE in %.1fs ===", time.time() - started)
    log.info("Final argmin=%d  one_se=%d", variant_block["argmin_rank"], variant_block["one_se_rank"])


if __name__ == "__main__":
    main()
