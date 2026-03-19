#!/usr/bin/env python3
"""
Full comparison of RSM vs SPoSE on complete data.
"""
import logging
from pathlib import Path
import numpy as np
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_count_rsm(n: int, triplets: np.ndarray) -> np.ndarray:
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        rsm = counts / shown
    rsm[~np.isfinite(rsm)] = 0.5
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_triplet_accuracy(s: np.ndarray, triplets: np.ndarray, fallback: float = 0.5) -> float:
    correct = 0
    unknown = 0
    for i, j, k in triplets:
        sims = np.array([s[i,j], s[i,k], s[j,k]])
        if np.all(np.isfinite(sims)):
            if np.argmax(sims) == 0:
                correct += 1
        else:
            unknown += 1
            if np.random.random() < fallback:  # Random guess when unknown
                correct += 1/3
    return correct / len(triplets)


def embedding_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


def main():
    # Load all data
    log.info("Loading full dataset...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things/triplets_47")
    train_triplets = np.loadtxt(data_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "validationset.txt").astype(int)
    n = 1854

    log.info(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

    # Build RSM from train
    log.info("\nBuilding RSM from train triplets...")
    rsm = compute_count_rsm(n, train_triplets)

    # Check coverage
    n_pairs = n * (n - 1) // 2
    shown = rsm != 0.5
    np.fill_diagonal(shown, False)
    coverage = shown.sum() // 2
    log.info(f"RSM coverage: {coverage}/{n_pairs} pairs ({100*coverage/n_pairs:.1f}%)")

    # Evaluate RSM
    log.info("\n=== RSM Evaluation ===")
    train_acc = rsm_triplet_accuracy(rsm, train_triplets)
    val_acc = rsm_triplet_accuracy(rsm, val_triplets)
    log.info(f"RSM Train: {train_acc:.4f}")
    log.info(f"RSM Val:   {val_acc:.4f}")

    # Evaluate SPoSE
    log.info("\n=== SPoSE Evaluation ===")
    spose = np.maximum(np.loadtxt(Path("/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt")), 0)
    train_acc = embedding_accuracy(spose, train_triplets)
    val_acc = embedding_accuracy(spose, val_triplets)
    log.info(f"SPoSE Train: {train_acc:.4f}")
    log.info(f"SPoSE Val:   {val_acc:.4f}")

    # Check overlap between train and val triplets
    log.info("\n=== Train/Val overlap analysis ===")
    train_set = set(map(tuple, train_triplets))
    val_set = set(map(tuple, val_triplets))
    overlap = len(train_set & val_set)
    log.info(f"Exact triplet overlap: {overlap}")

    # Check pair coverage in validation
    val_pairs = set()
    for i, j, k in val_triplets:
        val_pairs.add((min(i,j), max(i,j)))
        val_pairs.add((min(i,k), max(i,k)))
        val_pairs.add((min(j,k), max(j,k)))

    train_pairs = set()
    for i, j, k in train_triplets:
        train_pairs.add((min(i,j), max(i,j)))
        train_pairs.add((min(i,k), max(i,k)))
        train_pairs.add((min(j,k), max(j,k)))

    pair_coverage = len(val_pairs & train_pairs) / len(val_pairs)
    log.info(f"Val pairs seen in train: {100*pair_coverage:.1f}%")

    # Break down RSM performance by how many times pairs were seen
    log.info("\n=== RSM performance by observation count ===")
    shown_counts = np.zeros((n, n))
    for i, j, k in train_triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown_counts[a, b] += 1
            shown_counts[b, a] += 1

    # Bin validation triplets by minimum observation count
    bins = [(0, 0), (1, 5), (6, 20), (21, 100), (101, float('inf'))]
    for lo, hi in bins:
        subset = []
        for i, j, k in val_triplets:
            min_count = min(shown_counts[i,j], shown_counts[i,k], shown_counts[j,k])
            if lo <= min_count <= hi:
                subset.append((i, j, k))
        if subset:
            subset = np.array(subset)
            acc = rsm_triplet_accuracy(rsm, subset)
            log.info(f"  Obs {lo}-{hi}: {len(subset)} triplets, acc={acc:.4f}")


if __name__ == "__main__":
    main()
