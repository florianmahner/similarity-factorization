#!/usr/bin/env python3
"""
Test different RSM construction methods.
Evaluate using the same pipeline as experiments/things_behavior.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_triplet_prediction_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    """Same as experiments/things_behavior/common.py"""
    acc = 0
    for i, j, k in triplets:
        similarities = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
        probas = np.exp(similarities) / np.sum(np.exp(similarities))
        acc += np.argmax(probas) == 0
    return acc / len(triplets)


# ============================================================================
# RSM Construction Methods
# ============================================================================

def rsm_standard(n: int, triplets: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Current method from utils/helpers.py"""
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
        counts + alpha,
        shown + 2 * alpha,
        out=np.nan * np.ones_like(counts),
        where=shown != 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return similarity


def rsm_laplace(n: int, triplets: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplace smoothing with alpha > 0"""
    return rsm_standard(n, triplets, alpha=alpha)


def rsm_log_transform(n: int, triplets: np.ndarray) -> np.ndarray:
    """Log-odds transform of probabilities"""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # Laplace smoothing then log-odds
    prob = (counts + 1) / (shown + 3)
    log_odds = np.log(prob / (1 - prob))

    # Scale to [0, 1] with diagonal = 1
    np.fill_diagonal(log_odds, np.nan)
    log_odds = (log_odds - np.nanmin(log_odds)) / (np.nanmax(log_odds) - np.nanmin(log_odds))
    np.fill_diagonal(log_odds, 1.0)

    # Handle any remaining NaNs (unseen pairs)
    log_odds[np.isnan(log_odds)] = 0.5
    return log_odds


def rsm_softmax_inverse(n: int, triplets: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Attempt to invert the softmax. If P = softmax(s), then s = log(P) + const.
    """
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # Get probability with smoothing
    prob = (counts + 0.5) / (shown + 1.5)

    # Inverse softmax: s = temp * log(P)
    s = temp * np.log(prob + 1e-10)

    # Normalize
    np.fill_diagonal(s, np.nan)
    s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s))
    np.fill_diagonal(s, 1.0)
    s[np.isnan(s)] = 0.5
    return s


def rsm_sqrt_var_stabilize(n: int, triplets: np.ndarray) -> np.ndarray:
    """Variance stabilizing transform: arcsin(sqrt(p))"""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    prob = (counts + 0.5) / (shown + 1)
    # Arcsin-sqrt transform (variance stabilizing for binomial)
    transformed = np.arcsin(np.sqrt(prob))

    np.fill_diagonal(transformed, np.nan)
    transformed = (transformed - np.nanmin(transformed)) / (np.nanmax(transformed) - np.nanmin(transformed))
    np.fill_diagonal(transformed, 1.0)
    transformed[np.isnan(transformed)] = 0.5
    return transformed


def rsm_weighted_by_confidence(n: int, triplets: np.ndarray) -> np.ndarray:
    """
    Weight observations by confidence (more observations = more weight).
    Use Wilson score interval center.
    """
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # Wilson score center (better than raw proportion for small n)
    z = 1.96  # 95% confidence
    z2 = z * z
    n_obs = shown + 1e-10
    p_hat = counts / n_obs

    # Wilson score center
    center = (p_hat + z2 / (2 * n_obs)) / (1 + z2 / n_obs)

    np.fill_diagonal(center, 1.0)
    center[shown == 0] = 0.5
    return center


def rsm_net_wins(n: int, triplets: np.ndarray) -> np.ndarray:
    """
    Net wins: how many times (i,j) was chosen minus how many times it lost.
    """
    wins = np.zeros((n, n))
    losses = np.zeros((n, n))
    for i, j, k in triplets:
        wins[i, j] += 1
        wins[j, i] += 1
        losses[i, k] += 1
        losses[k, i] += 1
        losses[j, k] += 1
        losses[k, j] += 1

    net = wins - losses
    total = wins + losses + 1e-10

    # Normalize by total appearances
    rsm = net / total

    np.fill_diagonal(rsm, np.nan)
    rsm = (rsm - np.nanmin(rsm)) / (np.nanmax(rsm) - np.nanmin(rsm))
    np.fill_diagonal(rsm, 1.0)
    rsm[np.isnan(rsm)] = 0.5
    return rsm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=66)
    args = parser.parse_args()

    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    vice = np.loadtxt(data_dir / "vice_embedding_66d.txt")
    n = 1854

    log.info(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

    # Baselines
    log.info("\n=== Baselines ===")
    log.info(f"SPoSE: {compute_triplet_prediction_accuracy(spose, val_triplets):.4f}")
    log.info(f"VICE:  {compute_triplet_prediction_accuracy(vice, val_triplets):.4f}")
    log.info(f"Noise ceiling: 0.6667")

    # RSM methods
    methods = {
        "standard (alpha=0)": lambda: rsm_standard(n, train_triplets, alpha=0.0),
        "laplace (alpha=0.5)": lambda: rsm_laplace(n, train_triplets, alpha=0.5),
        "laplace (alpha=1)": lambda: rsm_laplace(n, train_triplets, alpha=1.0),
        "laplace (alpha=2)": lambda: rsm_laplace(n, train_triplets, alpha=2.0),
        "log_transform": lambda: rsm_log_transform(n, train_triplets),
        "softmax_inv (t=0.5)": lambda: rsm_softmax_inverse(n, train_triplets, temp=0.5),
        "softmax_inv (t=1)": lambda: rsm_softmax_inverse(n, train_triplets, temp=1.0),
        "softmax_inv (t=2)": lambda: rsm_softmax_inverse(n, train_triplets, temp=2.0),
        "sqrt_var_stabilize": lambda: rsm_sqrt_var_stabilize(n, train_triplets),
        "wilson_score": lambda: rsm_weighted_by_confidence(n, train_triplets),
        "net_wins": lambda: rsm_net_wins(n, train_triplets),
    }

    from pysrf import SRF

    log.info("\n=== RSM Construction Methods ===")
    results = []
    for name, rsm_fn in methods.items():
        rsm = rsm_fn()

        # Handle NaNs
        nan_count = np.isnan(rsm).sum()
        rsm[np.isnan(rsm)] = 0.5

        # Fit SRF
        model = SRF(rank=args.rank, random_state=args.seed, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        embedding = model.fit_transform(rsm)

        acc = compute_triplet_prediction_accuracy(embedding, val_triplets)
        log.info(f"{name}: {acc:.4f} (NaNs: {nan_count})")
        results.append((name, acc))

    # Summary
    log.info("\n=== Summary (sorted) ===")
    log.info(f"Noise ceiling: 0.6667")
    log.info(f"VICE:  0.6422")
    log.info(f"SPoSE: 0.6412")
    for name, acc in sorted(results, key=lambda x: -x[1]):
        delta = (acc - 0.6412) * 100
        log.info(f"{name}: {acc:.4f} ({delta:+.2f}% vs SPoSE)")


if __name__ == "__main__":
    main()
