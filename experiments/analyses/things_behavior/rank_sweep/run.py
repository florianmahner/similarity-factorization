"""Triplet prediction accuracy as a function of SRF rank.

Each seed uses a different random 90/10 split of trainset.txt:
- 90% for RSM construction
- 10% held out (not used here, reserved for model selection)
- Evaluation on canonical validationset.txt

SPoSE and VICE baselines evaluated once on the same validation set.

Outputs:
    results.csv                          -- model, alpha, rank, seed, val_acc
    embeddings/srf_alpha{a}_k{k}_s{s}.npz -- embedding array per fit
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF

from ..common import (
    compute_similarity_matrix_from_triplets,
    compute_triplet_prediction_accuracy,
)
from ..resources import load_resources

log = logging.getLogger(__name__)


def _make_split(triplets: np.ndarray, seed: int, frac: float = 0.9):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(triplets))
    split = int(frac * len(triplets))
    return triplets[idx[:split]]


def _run_single(
    train: np.ndarray,
    val: np.ndarray,
    n_items: int,
    alpha: int,
    rank: int,
    seed: int,
    max_outer: int = 50,
    max_inner: int = 200,
) -> tuple[dict, np.ndarray]:
    rsm = compute_similarity_matrix_from_triplets(n_items, train, alpha=alpha)
    model = SRF(
        rank=rank, random_state=seed,
        max_outer=max_outer, max_inner=max_inner,
        tol=1e-4, verbose=0,
    )
    embedding = model.fit_transform(rsm)
    acc = compute_triplet_prediction_accuracy(embedding, val)
    record = {
        "model": "SRF",
        "alpha": alpha,
        "rank": rank,
        "seed": seed,
        "val_acc": acc,
    }
    return record, embedding


def run(cfg: DictConfig) -> None:
    output_dir = Path.cwd()
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    resources = load_resources(cfg)
    val = resources.validation_triplets

    train_all = resources.train_triplets
    log.info("Total training triplets: %d", len(train_all))

    splits = {}
    for seed in cfg.seeds:
        splits[seed] = _make_split(train_all, seed)
    log.info("Created %d random 90/10 splits", len(splits))

    tasks = [
        (splits[seed], val, cfg.n_items, alpha, rank, seed)
        for alpha in cfg.alphas
        for rank in cfg.ranks
        for seed in cfg.seeds
    ]
    log.info(
        "Running %d SRF fits (%d alphas x %d ranks x %d seeds)...",
        len(tasks), len(cfg.alphas), len(cfg.ranks), len(cfg.seeds),
    )

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_single)(tr, v, n, a, r, s)
        for tr, v, n, a, r, s in tasks
    )

    records = []
    for record, embedding in results:
        records.append(record)
        a, k, s = record["alpha"], record["rank"], record["seed"]
        np.savez_compressed(
            emb_dir / f"srf_alpha{a}_k{k}_s{s}.npz",
            embedding=embedding,
        )

    acc_spose = compute_triplet_prediction_accuracy(
        resources.spose_embedding, val
    )
    acc_vice = compute_triplet_prediction_accuracy(
        resources.vice_embedding, val
    )
    records.append({
        "model": "SPoSE",
        "alpha": np.nan,
        "rank": resources.spose_embedding.shape[1],
        "seed": 0,
        "val_acc": acc_spose,
    })
    records.append({
        "model": "VICE",
        "alpha": np.nan,
        "rank": resources.vice_embedding.shape[1],
        "seed": 0,
        "val_acc": acc_vice,
    })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "results.csv", index=False)

    log.info("SPoSE (k=%d): %.4f", resources.spose_embedding.shape[1], acc_spose)
    log.info("VICE  (k=%d): %.4f", resources.vice_embedding.shape[1], acc_vice)

    for alpha in cfg.alphas:
        sub = df[(df["model"] == "SRF") & (df["alpha"] == alpha)]
        summary = sub.groupby("rank")["val_acc"].agg(["mean", "std"])
        log.info("\nalpha=%d:", alpha)
        for rank, row in summary.iterrows():
            log.info("  k=%3d: %.4f +/- %.4f", rank, row["mean"], row["std"])

    log.info("Saved %d rows + %d embeddings", len(df), len(results))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
