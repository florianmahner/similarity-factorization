#!/usr/bin/env python3
"""
Test KL divergence loss function in SRF for triplet prediction.
"""
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    acc = 0
    for i, j, k in triplets:
        sims = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
        probas = np.exp(sims - sims.max())
        probas /= probas.sum()
        acc += np.argmax(probas) == 0
    return acc / len(triplets)


def build_rsm(n: int, triplets: np.ndarray):
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    rsm = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main():
    from pysrf import SRF

    log.info("Loading 4.7M dataset...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    rsm = build_rsm(n, train_triplets)

    spose_acc = compute_triplet_accuracy(spose, val_triplets)
    log.info(f"\nSPoSE baseline: {spose_acc:.4f}")
    log.info(f"Noise ceiling: 0.6667")

    log.info(f"\n{'='*50}")
    log.info("Testing Different Loss Functions")
    log.info(f"{'='*50}")

    losses = ["frobenius", "kullback-leibler", "bce"]

    for loss in losses:
        log.info(f"\n{loss}:")
        try:
            model = SRF(
                rank=66,
                loss=loss,
                random_state=0,
                max_outer=2000,
                max_inner=50,
                tol=1e-4,
                verbose=1,
            )
            emb = model.fit_transform(rsm)
            acc = compute_triplet_accuracy(emb, val_triplets)
            log.info(f"  Accuracy: {acc:.4f}")
            log.info(f"  Final loss: {model.loss_:.6f}")
            log.info(f"  Iterations: {model.n_iter_}")
        except Exception as e:
            log.info(f"  Error: {e}")

    log.info(f"\n{'='*50}")
    log.info("Testing KL with different rho values")
    log.info(f"{'='*50}")

    for rho in [0.01, 0.1, 0.5, 1.0, 2.0]:
        log.info(f"\nKL (rho={rho}):")
        try:
            model = SRF(
                rank=66,
                loss="kullback-leibler",
                rho=rho,
                random_state=0,
                max_outer=2000,
                max_inner=50,
                tol=1e-4,
                verbose=0,
            )
            emb = model.fit_transform(rsm)
            acc = compute_triplet_accuracy(emb, val_triplets)
            log.info(f"  Accuracy: {acc:.4f}")
        except Exception as e:
            log.info(f"  Error: {e}")


if __name__ == "__main__":
    main()
