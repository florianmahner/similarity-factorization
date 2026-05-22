#!/usr/bin/env python3
"""
Test different SRF parameters to see if we can beat SPoSE.
"""
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_triplet_prediction_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    acc = 0
    for i, j, k in triplets:
        similarities = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
        probas = np.exp(similarities) / np.sum(np.exp(similarities))
        acc += np.argmax(probas) == 0
    return acc / len(triplets)


def rsm_standard(n: int, triplets: np.ndarray) -> np.ndarray:
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            if a != b:
                shown[a, b] += 1
                shown[b, a] += 1
        if i != j:
            counts[i, j] += 1
            counts[j, i] += 1
    similarity = np.divide(
        counts, shown,
        out=0.5 * np.ones_like(counts),
        where=shown != 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    # Baselines
    log.info(f"\nBaselines:")
    log.info(f"SPoSE (66d): {compute_triplet_prediction_accuracy(spose, val_triplets):.4f}")

    # Build RSM
    rsm = rsm_standard(n, train_triplets)
    log.info(f"RSM built, shape: {rsm.shape}")

    from pysrf import SRF

    # Test different ranks
    log.info("\n=== Varying Rank ===")
    for rank in [20, 40, 66, 100, 150, 200, 300]:
        model = SRF(rank=rank, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm)
        acc = compute_triplet_prediction_accuracy(w, val_triplets)
        log.info(f"Rank {rank}: {acc:.4f}")

    # Test different rho values
    log.info("\n=== Varying rho (rank=66) ===")
    for rho in [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]:
        model = SRF(rank=66, rho=rho, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm)
        acc = compute_triplet_prediction_accuracy(w, val_triplets)
        log.info(f"rho={rho}: {acc:.4f}")

    # Test different loss functions
    log.info("\n=== Varying Loss (rank=66) ===")
    for loss in ["frobenius", "kullback-leibler"]:
        try:
            model = SRF(rank=66, loss=loss, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
            w = model.fit_transform(rsm)
            acc = compute_triplet_prediction_accuracy(w, val_triplets)
            log.info(f"loss={loss}: {acc:.4f}")
        except Exception as e:
            log.info(f"loss={loss}: ERROR - {e}")

    # Best rank with more iterations
    log.info("\n=== Best rank with more iterations ===")
    best_rank = 66
    for max_outer in [2000, 5000]:
        model = SRF(rank=best_rank, random_state=args.seed, max_outer=max_outer, max_inner=100, tol=1e-5, verbose=0)
        w = model.fit_transform(rsm)
        acc = compute_triplet_prediction_accuracy(w, val_triplets)
        log.info(f"max_outer={max_outer}: {acc:.4f}")


if __name__ == "__main__":
    main()
