#!/usr/bin/env python3
"""
Compare different RSM construction methods from triplets.
Goal: Find a construction that preserves signal after SRF factorization.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def embedding_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    """Triplet accuracy from embedding (argmax of dot products)."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


# ============================================================================
# RSM Construction Methods
# ============================================================================

def rsm_counts(n: int, triplets: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Standard count-based: P(i,j chosen | shown)"""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        rsm = (counts + alpha) / (shown + 2 * alpha)
    rsm[~np.isfinite(rsm)] = 0.5
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_log_odds(n: int, triplets: np.ndarray) -> np.ndarray:
    """Log-odds transformation of counts."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # Laplace smoothing
    prob = (counts + 1) / (shown + 3)
    log_odds = np.log(prob) - np.log(1 - prob)

    # Normalize to [0, 1]
    np.fill_diagonal(log_odds, np.nan)
    log_odds = (log_odds - np.nanmin(log_odds)) / (np.nanmax(log_odds) - np.nanmin(log_odds))
    np.fill_diagonal(log_odds, 1.0)
    return log_odds


def rsm_wins_minus_losses(n: int, triplets: np.ndarray) -> np.ndarray:
    """Net wins: (times chosen) - (times not chosen when shown)"""
    wins = np.zeros((n, n))
    losses = np.zeros((n, n))
    for i, j, k in triplets:
        # (i,j) won against (i,k) and (j,k)
        wins[i, j] += 1
        wins[j, i] += 1
        losses[i, k] += 1
        losses[k, i] += 1
        losses[j, k] += 1
        losses[k, j] += 1

    net = wins - losses
    # Normalize
    np.fill_diagonal(net, np.nan)
    net = (net - np.nanmin(net)) / (np.nanmax(net) - np.nanmin(net))
    np.fill_diagonal(net, 1.0)
    return net


def rsm_psd_projection(rsm: np.ndarray) -> np.ndarray:
    """Project RSM to nearest PSD matrix."""
    eigenvalues, eigenvectors = eigh(rsm)
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative eigenvalues
    rsm_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Renormalize
    np.fill_diagonal(rsm_psd, 1.0)
    return rsm_psd


def rsm_kernel_smoothed(n: int, triplets: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Kernel smoothing: use embedding similarity to smooth the counts.
    First get counts, then smooth based on item similarity.
    """
    # Start with counts
    rsm = rsm_counts(n, triplets)

    # Use eigendecomposition to get initial embedding
    eigenvalues, eigenvectors = eigh(rsm)
    # Take top positive eigenvalues
    pos_mask = eigenvalues > 0
    k = min(100, pos_mask.sum())
    idx = np.argsort(eigenvalues)[::-1][:k]
    w = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

    # Compute item-item kernel
    sims = w @ w.T
    sims = sims / (np.linalg.norm(w, axis=1, keepdims=True) @ np.linalg.norm(w, axis=1, keepdims=True).T + 1e-10)
    kernel = np.exp(sims / sigma)
    kernel /= kernel.sum(axis=1, keepdims=True)

    # Smooth RSM
    rsm_smooth = kernel @ rsm @ kernel.T
    np.fill_diagonal(rsm_smooth, 1.0)
    return rsm_smooth


def rsm_sqrt_counts(n: int, triplets: np.ndarray) -> np.ndarray:
    """Square root of counts (variance stabilization)."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    # sqrt transform before ratio
    rsm = np.sqrt(counts + 0.5) / np.sqrt(shown + 1)
    rsm[shown == 0] = 0.5
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_rank_based(n: int, triplets: np.ndarray) -> np.ndarray:
    """
    Rank-based: for each item i, rank all other items by their similarity to i.
    Then convert ranks to similarities.
    """
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        prob = counts / shown
    prob[~np.isfinite(prob)] = 0.5

    # For each row, convert to ranks
    rsm = np.zeros((n, n))
    for i in range(n):
        ranks = np.argsort(np.argsort(-prob[i]))  # Higher prob = lower rank
        rsm[i] = 1 - ranks / (n - 1)

    rsm = (rsm + rsm.T) / 2
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_doubly_stochastic(n: int, triplets: np.ndarray, n_iters: int = 20) -> np.ndarray:
    """
    Sinkhorn normalization to make RSM doubly stochastic.
    This often helps with factorization.
    """
    rsm = rsm_counts(n, triplets)
    rsm = np.maximum(rsm, 1e-10)  # Ensure positive

    for _ in range(n_iters):
        rsm = rsm / rsm.sum(axis=1, keepdims=True)
        rsm = rsm / rsm.sum(axis=0, keepdims=True)

    rsm = (rsm + rsm.T) / 2
    # Rescale to [0, 1]
    rsm = (rsm - rsm.min()) / (rsm.max() - rsm.min())
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=66)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things/triplets_47")
    train_triplets = np.loadtxt(data_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "validationset.txt").astype(int)
    n = 1854

    if args.subsample < 1.0:
        n_samples = int(len(train_triplets) * args.subsample)
        idx = np.random.choice(len(train_triplets), size=n_samples, replace=False)
        train_triplets = train_triplets[idx]

    log.info(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

    # Baselines
    spose = np.maximum(np.loadtxt(Path("/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt")), 0)
    vice = np.loadtxt(Path("/LOCAL/fmahner/similarity-factorization/data/things/vice_embedding_66d.txt"))

    log.info(f"\n=== Baselines ===")
    log.info(f"SPoSE: train={embedding_accuracy(spose, train_triplets):.4f}, val={embedding_accuracy(spose, val_triplets):.4f}")
    log.info(f"VICE:  train={embedding_accuracy(vice, train_triplets):.4f}, val={embedding_accuracy(vice, val_triplets):.4f}")

    # RSM construction methods
    methods = {
        "counts": lambda: rsm_counts(n, train_triplets),
        "counts_laplace": lambda: rsm_counts(n, train_triplets, alpha=1.0),
        "log_odds": lambda: rsm_log_odds(n, train_triplets),
        "wins_minus_losses": lambda: rsm_wins_minus_losses(n, train_triplets),
        "sqrt_counts": lambda: rsm_sqrt_counts(n, train_triplets),
        "rank_based": lambda: rsm_rank_based(n, train_triplets),
        "doubly_stochastic": lambda: rsm_doubly_stochastic(n, train_triplets),
    }

    from pysrf import SRF

    results = []
    for name, rsm_fn in methods.items():
        log.info(f"\n=== {name} ===")
        rsm = rsm_fn()

        # Check eigenvalues
        eigenvalues = eigh(rsm, eigvals_only=True)
        n_neg = np.sum(eigenvalues < -1e-10)
        log.info(f"  Negative eigenvalues: {n_neg}/{n}")

        # Also try PSD projection
        for use_psd in [False, True]:
            rsm_input = rsm_psd_projection(rsm) if use_psd else rsm
            suffix = "_psd" if use_psd else ""

            model = SRF(rank=args.rank, random_state=args.seed)
            w = model.fit_transform(rsm_input)

            train_acc = embedding_accuracy(w, train_triplets)
            val_acc = embedding_accuracy(w, val_triplets)
            log.info(f"  SRF{suffix}: train={train_acc:.4f}, val={val_acc:.4f}")
            results.append((f"{name}{suffix}", val_acc))

    # Summary
    log.info(f"\n=== Summary (sorted by val acc) ===")
    log.info(f"SPoSE:  {embedding_accuracy(spose, val_triplets):.4f}")
    log.info(f"VICE:   {embedding_accuracy(vice, val_triplets):.4f}")
    for name, acc in sorted(results, key=lambda x: -x[1]):
        log.info(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
