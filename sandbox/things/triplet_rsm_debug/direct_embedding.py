#!/usr/bin/env python3
"""
Direct triplet embedding WITHOUT going through RSM.

Instead of: triplets -> RSM -> factorize -> embedding
We do:      triplets -> embedding directly

Key insight: The RSM is not positive semi-definite, so factorization destroys
the signal. But we can learn an embedding directly from triplet constraints.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax
from numba import njit, prange
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def triplet_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


def triplet_accuracy_softmax(w: np.ndarray, triplets: np.ndarray) -> float:
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if np.argmax(softmax(sims)) == 0:
            correct += 1
    return correct / len(triplets)


# ============================================================================
# Method 1: Soft Ordinal Embedding (SOE) - margin-based
# ============================================================================

def train_soe(
    n: int,
    triplets: np.ndarray,
    rank: int = 66,
    lr: float = 0.01,
    margin: float = 0.1,
    n_epochs: int = 10,
    batch_size: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Soft Ordinal Embedding with hinge loss.
    Constraint: w_i · w_j > w_i · w_k + margin AND w_i · w_j > w_j · w_k + margin
    """
    rng = np.random.default_rng(seed)

    # Initialize with small random values
    w = rng.standard_normal((n, rank)) * 0.01

    n_batches = len(triplets) // batch_size

    for epoch in range(n_epochs):
        # Shuffle triplets
        perm = rng.permutation(len(triplets))
        triplets_shuffled = triplets[perm]

        total_loss = 0
        total_violations = 0

        for batch_idx in range(n_batches):
            batch = triplets_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            grad = np.zeros_like(w)
            batch_loss = 0
            violations = 0

            for i, j, k in batch:
                sij = w[i] @ w[j]
                sik = w[i] @ w[k]
                sjk = w[j] @ w[k]

                # Constraint 1: sij > sik + margin
                if sij < sik + margin:
                    loss1 = sik + margin - sij
                    batch_loss += loss1
                    violations += 1
                    # Gradients
                    grad[i] -= w[j]  # Increase sij
                    grad[j] -= w[i]
                    grad[i] += w[k]  # Decrease sik
                    grad[k] += w[i]

                # Constraint 2: sij > sjk + margin
                if sij < sjk + margin:
                    loss2 = sjk + margin - sij
                    batch_loss += loss2
                    violations += 1
                    grad[i] -= w[j]
                    grad[j] -= w[i]
                    grad[j] += w[k]
                    grad[k] += w[j]

            # Update
            w -= lr * grad / batch_size
            total_loss += batch_loss
            total_violations += violations

        log.info(f"Epoch {epoch}: loss={total_loss/len(triplets):.4f}, violations={total_violations}")

    return w


# ============================================================================
# Method 2: Triplet loss with log-likelihood (like SPoSE but from scratch)
# ============================================================================

def train_triplet_softmax(
    n: int,
    triplets: np.ndarray,
    rank: int = 66,
    lr: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 10000,
    l2_reg: float = 0.001,
    seed: int = 42,
) -> np.ndarray:
    """
    Train embedding with softmax triplet loss (same as SPoSE).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, rank)) * 0.1
    w = np.maximum(w, 0)  # Non-negative like SPoSE

    n_batches = len(triplets) // batch_size

    for epoch in range(n_epochs):
        perm = rng.permutation(len(triplets))
        triplets_shuffled = triplets[perm]

        total_loss = 0

        for batch_idx in range(n_batches):
            batch = triplets_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            grad = np.zeros_like(w)
            batch_loss = 0

            for i, j, k in batch:
                sij = w[i] @ w[j]
                sik = w[i] @ w[k]
                sjk = w[j] @ w[k]

                # Softmax probabilities
                sims = np.array([sij, sik, sjk])
                probs = softmax(sims)

                # Negative log likelihood
                batch_loss -= np.log(probs[0] + 1e-10)

                # Gradients
                # d_loss/d_sij = -(1 - p_ij)
                # d_loss/d_sik = p_ik
                # d_loss/d_sjk = p_jk

                grad[i] -= (1 - probs[0]) * w[j]
                grad[j] -= (1 - probs[0]) * w[i]

                grad[i] += probs[1] * w[k]
                grad[k] += probs[1] * w[i]

                grad[j] += probs[2] * w[k]
                grad[k] += probs[2] * w[j]

            # L2 regularization
            grad += l2_reg * w

            # Update with projection to non-negative
            w -= lr * grad / batch_size
            w = np.maximum(w, 0)

            total_loss += batch_loss

        log.info(f"Epoch {epoch}: loss={total_loss/len(triplets):.4f}")

    return w


# ============================================================================
# Method 3: t-STE (t-distributed Stochastic Triplet Embedding)
# ============================================================================

def train_tste(
    n: int,
    triplets: np.ndarray,
    rank: int = 66,
    lr: float = 1.0,
    n_epochs: int = 10,
    batch_size: int = 10000,
    alpha: float = 1.0,  # Degrees of freedom for t-distribution
    seed: int = 42,
) -> np.ndarray:
    """
    t-STE uses t-distribution instead of Gaussian/softmax.
    P(i,j preferred) = (1 + d_ij^2 / alpha)^(-(alpha+1)/2) / Z

    This is more robust to outliers.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, rank)) * 0.1

    n_batches = len(triplets) // batch_size

    for epoch in range(n_epochs):
        perm = rng.permutation(len(triplets))
        triplets_shuffled = triplets[perm]

        total_loss = 0

        for batch_idx in range(n_batches):
            batch = triplets_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            grad = np.zeros_like(w)
            batch_loss = 0

            for i, j, k in batch:
                # Squared distances
                dij2 = np.sum((w[i] - w[j]) ** 2)
                dik2 = np.sum((w[i] - w[k]) ** 2)
                djk2 = np.sum((w[j] - w[k]) ** 2)

                # t-distribution kernel
                kij = (1 + dij2 / alpha) ** (-(alpha + 1) / 2)
                kik = (1 + dik2 / alpha) ** (-(alpha + 1) / 2)
                kjk = (1 + djk2 / alpha) ** (-(alpha + 1) / 2)

                # We want d_ij < d_ik and d_ij < d_jk
                # Equivalently: k_ij > k_ik and k_ij > k_jk
                Z = kij + kik + kjk
                p_ij = kij / Z

                batch_loss -= np.log(p_ij + 1e-10)

                # Gradient computation
                c = (alpha + 1) / alpha

                # d k_ij / d w_i = -c * k_ij * (w_i - w_j) / (1 + d_ij^2/alpha)
                dkij_dwi = -c * kij * (w[i] - w[j]) / (1 + dij2 / alpha)
                dkij_dwj = -c * kij * (w[j] - w[i]) / (1 + dij2 / alpha)

                dkik_dwi = -c * kik * (w[i] - w[k]) / (1 + dik2 / alpha)
                dkik_dwk = -c * kik * (w[k] - w[i]) / (1 + dik2 / alpha)

                dkjk_dwj = -c * kjk * (w[j] - w[k]) / (1 + djk2 / alpha)
                dkjk_dwk = -c * kjk * (w[k] - w[j]) / (1 + djk2 / alpha)

                # d_loss / d_w = -d_log(p_ij) / d_w = -(1/p_ij) * d_p_ij / d_w
                # p_ij = k_ij / Z
                # d p_ij / d k_ij = (Z - k_ij) / Z^2 = (1 - p_ij) / Z
                # d p_ij / d k_ik = -k_ij / Z^2 = -p_ij / Z

                grad[i] -= (1 - p_ij) / kij * dkij_dwi - p_ij / kik * dkik_dwi
                grad[j] -= (1 - p_ij) / kij * dkij_dwj - p_ij / kjk * dkjk_dwj
                grad[k] -= -p_ij / kik * dkik_dwk - p_ij / kjk * dkjk_dwk

            w -= lr * grad / batch_size
            total_loss += batch_loss

        log.info(f"Epoch {epoch}: loss={total_loss/len(triplets):.4f}")

    return w


# ============================================================================
# Method 4: GNMDS-style with non-negative constraints
# ============================================================================

def train_gnmds_nonneg(
    n: int,
    triplets: np.ndarray,
    rank: int = 66,
    lr: float = 0.1,
    n_epochs: int = 10,
    batch_size: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generalized Non-metric MDS with non-negative embedding.
    Uses similarity (not distance) with hinge loss.
    """
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank))) * 0.1

    n_batches = len(triplets) // batch_size

    for epoch in range(n_epochs):
        perm = rng.permutation(len(triplets))
        triplets_shuffled = triplets[perm]

        total_loss = 0
        total_violations = 0

        for batch_idx in range(n_batches):
            batch = triplets_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            grad = np.zeros_like(w)
            batch_loss = 0

            for i, j, k in batch:
                sij = w[i] @ w[j]
                sik = w[i] @ w[k]
                sjk = w[j] @ w[k]

                # Soft margin loss
                # We want sij > max(sik, sjk)
                margin = 0.0
                max_other = max(sik, sjk)

                if sij < max_other + margin:
                    loss = max_other + margin - sij
                    batch_loss += loss
                    total_violations += 1

                    # Gradient: increase sij, decrease max_other
                    grad[i] -= w[j]
                    grad[j] -= w[i]

                    if sik >= sjk:
                        grad[i] += w[k]
                        grad[k] += w[i]
                    else:
                        grad[j] += w[k]
                        grad[k] += w[j]

            w -= lr * grad / batch_size
            w = np.maximum(w, 0)  # Project to non-negative
            total_loss += batch_loss

        acc = triplet_accuracy(w, triplets_shuffled[:10000])
        log.info(f"Epoch {epoch}: loss={total_loss/len(triplets):.4f}, violations={total_violations}, acc={acc:.4f}")

    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=66)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--method", type=str, default="all", choices=["all", "soe", "softmax", "tste", "gnmds"])
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
    log.info(f"Using {len(train_triplets)} train triplets")

    # Baseline
    spose = np.maximum(np.loadtxt(Path("/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt")), 0)
    log.info(f"\n=== Baseline: SPoSE ===")
    log.info(f"Train: {triplet_accuracy(spose, train_triplets):.4f}")
    log.info(f"Val:   {triplet_accuracy(spose, val_triplets):.4f}")

    methods = {
        "soe": ("SOE (margin)", lambda: train_soe(n, train_triplets, rank=args.rank, n_epochs=args.n_epochs, seed=args.seed)),
        "softmax": ("Softmax (SPoSE-style)", lambda: train_triplet_softmax(n, train_triplets, rank=args.rank, n_epochs=args.n_epochs, seed=args.seed)),
        "tste": ("t-STE", lambda: train_tste(n, train_triplets, rank=args.rank, n_epochs=args.n_epochs, seed=args.seed)),
        "gnmds": ("GNMDS non-neg", lambda: train_gnmds_nonneg(n, train_triplets, rank=args.rank, n_epochs=args.n_epochs, seed=args.seed)),
    }

    if args.method == "all":
        to_run = methods.keys()
    else:
        to_run = [args.method]

    results = {}
    for method_name in to_run:
        name, train_fn = methods[method_name]
        log.info(f"\n=== {name} ===")
        w = train_fn()
        train_acc = triplet_accuracy(w, train_triplets)
        val_acc = triplet_accuracy(w, val_triplets)
        log.info(f"Final - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        results[name] = val_acc

    log.info(f"\n=== Summary ===")
    log.info(f"SPoSE:    {triplet_accuracy(spose, val_triplets):.4f}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        log.info(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
