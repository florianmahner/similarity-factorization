"""Direct (non-Hydra) runner for similarity_48 at the current configured srf_rank.

Builds the config in-process so it can be launched without ./scripts/submit
(which has been removed). Reads srf_rank/dims from similarity_48/config.yaml
and writes accuracy_comparison.csv (one row per model x seed) to outputs/rank{srf_rank}/.
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
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from src.tools.rsa import correlate_rsms, reconstruct_rsm
from src.utils.io import load_shared_data, load_triplets
from src.utils.helpers import compute_similarity_matrix_from_triplets
from experiments.analyses.things_behavior.common import (
    compute_triplet_prediction_accuracy,
    fit_srf_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated list of seeds (default matches the rank=66 baseline).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _one_seed(
    similarity: np.ndarray,
    spose_embedding: np.ndarray,
    vice_embedding: np.ndarray,
    indices_48: np.ndarray,
    rsm_48_true: np.ndarray,
    validation_triplets: np.ndarray,
    srf_rank: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    srf_embedding = fit_srf_model(similarity, rank=srf_rank, seed=seed)
    rsm_48_spose = reconstruct_rsm(spose_embedding[indices_48])
    rsm_48_vice = reconstruct_rsm(vice_embedding[indices_48])
    rsm_srf = reconstruct_rsm(srf_embedding)
    rsm_48_srf = rsm_srf[np.ix_(indices_48, indices_48)]

    corr_srf = correlate_rsms(rsm_48_srf, rsm_48_true)
    corr_spose = correlate_rsms(rsm_48_spose, rsm_48_true)
    corr_vice = correlate_rsms(rsm_48_vice, rsm_48_true)

    acc_srf = compute_triplet_prediction_accuracy(srf_embedding, validation_triplets)
    acc_spose = compute_triplet_prediction_accuracy(spose_embedding, validation_triplets)
    acc_vice = compute_triplet_prediction_accuracy(vice_embedding, validation_triplets)

    elapsed = time.time() - t0
    log.info(
        "  seed=%d  srf=%.4f/%.2f%%  vice=%.4f/%.2f%%  spose=%.4f/%.2f%%  (%.1fs)",
        seed,
        corr_srf, acc_srf * 100,
        corr_vice, acc_vice * 100,
        corr_spose, acc_spose * 100,
        elapsed,
    )
    summary_rows = [
        {"model": "SRF", "correlation": float(corr_srf), "accuracy": float(acc_srf), "seed": int(seed)},
        {"model": "VICE", "correlation": float(corr_vice), "accuracy": float(acc_vice), "seed": int(seed)},
        {"model": "SPoSE", "correlation": float(corr_spose), "accuracy": float(acc_spose), "seed": int(seed)},
    ]
    # Per-pair upper-triangle entries for the panel-b scatter.
    iu = np.triu_indices(len(indices_48), k=1)
    true_vec = rsm_48_true[iu]
    per_pair_rows = []
    for name, rsm in (("SRF", rsm_48_srf), ("SPoSE", rsm_48_spose), ("VICE", rsm_48_vice)):
        pred_vec = rsm[iu]
        for k, (t, p) in enumerate(zip(true_vec, pred_vec)):
            per_pair_rows.append({
                "model": name, "seed": int(seed), "pair_idx": int(k),
                "true_similarity": float(t), "predicted_similarity": float(p),
            })
    return summary_rows, per_pair_rows


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg_path = HERE / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dataset_path = str(PROJECT_ROOT / "data" / "things")
    cfg.images_path = "/data/labshare/_stachelschwein/SSD/datasets/things/behav1854"
    cfg.vice_embedding_path = str(PROJECT_ROOT / "data" / "things" / "vice_embedding_66d.txt")

    srf_rank = int(cfg.get("srf_rank", cfg.dims))
    out_dir = HERE / "outputs" / f"rank{srf_rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== similarity_48 multi-seed direct run ===")
    log.info("srf_rank=%d  spose_dims=%d  seeds=%s", srf_rank, int(cfg.dims), seeds)
    log.info("out_dir=%s", out_dir)
    log.info("n_jobs=%d", args.n_jobs)

    log.info("Loading shared data ...")
    spose_embedding, indices_48, rsm_48_true = load_shared_data(
        Path(cfg.dataset_path), Path(cfg.images_path), num_dims=int(cfg.dims)
    )
    vice_embedding = np.maximum(np.loadtxt(cfg.vice_embedding_path), 0)
    train_triplets, validation_triplets = load_triplets(
        Path(cfg.dataset_path), number=cfg.triplet_version
    )

    log.info("Building training similarity matrix from triplets (n=%d) ...", int(cfg.n_items))
    similarity = compute_similarity_matrix_from_triplets(int(cfg.n_items), train_triplets)

    log.info("Fitting SRF rank=%d for %d seeds ...", srf_rank, len(seeds))
    t_all = time.time()
    seed_outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_one_seed)(
            similarity,
            spose_embedding,
            vice_embedding,
            np.asarray(indices_48),
            rsm_48_true,
            validation_triplets,
            srf_rank,
            seed,
        )
        for seed in seeds
    )
    log.info("All %d seeds done in %.1fs", len(seeds), time.time() - t_all)

    flat_summary = [r for (s, _) in seed_outputs for r in s]
    flat_pairs = [r for (_, pp) in seed_outputs for r in pp]

    df = pd.DataFrame(flat_summary)
    csv_path = out_dir / "accuracy_comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s (%d rows)", csv_path, len(df))

    pair_df = pd.DataFrame(flat_pairs)
    pair_csv = out_dir / "48_performance.csv"
    pair_df.to_csv(pair_csv, index=False)
    log.info("Wrote %s (%d rows)", pair_csv, len(pair_df))

    log.info("\n=== summary (rank=%d, n_seeds=%d) ===", srf_rank, len(seeds))
    for model in ["SRF", "VICE", "SPoSE"]:
        sub = df[df["model"] == model]
        log.info(
            "  %-5s  corr_48 %.4f +/- %.4f   acc %.2f%% +/- %.2f%%",
            model,
            sub["correlation"].mean(), sub["correlation"].std(ddof=1) if len(sub) > 1 else 0.0,
            sub["accuracy"].mean() * 100, sub["accuracy"].std(ddof=1) * 100 if len(sub) > 1 else 0.0,
        )


if __name__ == "__main__":
    main()
