#!/usr/bin/env python3
"""
Debug script to compare different triplet-to-similarity conversion methods.

The core issue: The count-based RSM doesn't recover the underlying similarity
because triplet choices follow a softmax model:
    P(choose i,j | i,j,k) = exp(s_ij) / (exp(s_ij) + exp(s_ik) + exp(s_jk))

The count-based estimate gives E[P(choose i,j)] which is NOT equal to s_ij.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax
from joblib import Parallel, delayed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_count_based_rsm(n: int, triplets: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Original count-based approach."""
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


def compute_log_odds_rsm(n: int, triplets: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert counts to log-odds, which is closer to the underlying similarity.

    If P = exp(s) / Z, then log(P / (1-P)) = s - log(Z - exp(s))
    For 3-way softmax: P = exp(s_ij) / (exp(s_ij) + exp(s_ik) + exp(s_jk))

    This is an approximation that may work better than raw counts.
    """
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

    # Compute probabilities with Laplace smoothing
    prob = (counts + 1) / (shown + 3)

    # Convert to log-odds
    log_odds = np.log(prob + eps) - np.log(1 - prob + eps)

    # Symmetrize and normalize
    np.fill_diagonal(log_odds, np.max(log_odds))  # Set diagonal to max

    # Shift to [0, 1] range
    log_odds = (log_odds - log_odds.min()) / (log_odds.max() - log_odds.min())
    np.fill_diagonal(log_odds, 1.0)

    return log_odds


def triplet_loss(s: np.ndarray, triplets: np.ndarray) -> float:
    """Negative log likelihood of triplet choices."""
    loss = 0.0
    for i, j, k in triplets:
        sims = np.array([s[i, j], s[i, k], s[j, k]])
        log_probs = sims - np.log(np.sum(np.exp(sims)))
        loss -= log_probs[0]  # Negative log prob of correct choice
    return loss / len(triplets)


def triplet_gradient(s: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Gradient of negative log likelihood w.r.t. similarity matrix."""
    n = s.shape[0]
    grad = np.zeros_like(s)

    for i, j, k in triplets:
        sims = np.array([s[i, j], s[i, k], s[j, k]])
        probs = softmax(sims)

        # Gradient for s_ij (correct pair)
        grad[i, j] -= (1 - probs[0])
        grad[j, i] -= (1 - probs[0])

        # Gradient for s_ik (incorrect pair)
        grad[i, k] += probs[1]
        grad[k, i] += probs[1]

        # Gradient for s_jk (incorrect pair)
        grad[j, k] += probs[2]
        grad[k, j] += probs[2]

    return grad / len(triplets)


def compute_mle_rsm(
    n: int,
    triplets: np.ndarray,
    lr: float = 0.1,
    n_iters: int = 100,
    init: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Maximum Likelihood Estimation of similarity matrix from triplets.

    Optimizes: log L(S) = sum_triplets log P(correct choice | S)
    where P(i,j | i,j,k; S) = softmax([S_ij, S_ik, S_jk])[0]
    """
    # Initialize with count-based estimate (in log-odds space)
    if init is not None:
        s = init.copy()
    else:
        s = compute_log_odds_rsm(n, triplets)

    # Use Adam optimizer
    m = np.zeros_like(s)
    v = np.zeros_like(s)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    iterator = range(n_iters)
    if verbose:
        iterator = tqdm(iterator, desc="MLE optimization")

    for t in iterator:
        grad = triplet_gradient(s, triplets)

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))

        s -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Ensure symmetry
        s = (s + s.T) / 2

        if verbose and t % 20 == 0:
            loss = triplet_loss(s, triplets)
            acc = compute_triplet_accuracy(s, triplets)
            log.info(f"Iter {t}: loss={loss:.4f}, acc={acc:.4f}")

    # Normalize to [0, 1]
    np.fill_diagonal(s, np.max(s))
    s = (s - s.min()) / (s.max() - s.min())
    np.fill_diagonal(s, 1.0)

    return s


def compute_triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    """Compute triplet prediction accuracy using a similarity matrix."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([s[i, j], s[i, k], s[j, k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


def compute_embedding_triplet_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    """Compute triplet accuracy from embedding using softmax prediction."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        probs = softmax(sims)
        if np.argmax(probs) == 0:
            correct += 1
    return correct / len(triplets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=50, help="MLE iterations")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=0.1, help="Subsample fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=66, help="SRF rank")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things/triplets_47")
    train_triplets = np.loadtxt(data_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "validationset.txt").astype(int)
    n = 1854

    log.info(f"Train triplets: {len(train_triplets)}, Val triplets: {len(val_triplets)}")

    # Subsample for faster iteration
    if args.subsample < 1.0:
        n_samples = int(len(train_triplets) * args.subsample)
        idx = np.random.choice(len(train_triplets), size=n_samples, replace=False)
        train_triplets = train_triplets[idx]
        log.info(f"Subsampled to {len(train_triplets)} triplets")

    # Method 1: Count-based RSM
    log.info("\n=== Count-based RSM ===")
    rsm_counts = compute_count_based_rsm(n, train_triplets)
    acc_counts_train = compute_triplet_accuracy(rsm_counts, train_triplets)
    acc_counts_val = compute_triplet_accuracy(rsm_counts, val_triplets)
    log.info(f"Count RSM: train acc={acc_counts_train:.4f}, val acc={acc_counts_val:.4f}")

    # Method 2: Log-odds RSM
    log.info("\n=== Log-odds RSM ===")
    rsm_logodds = compute_log_odds_rsm(n, train_triplets)
    acc_logodds_train = compute_triplet_accuracy(rsm_logodds, train_triplets)
    acc_logodds_val = compute_triplet_accuracy(rsm_logodds, val_triplets)
    log.info(f"Log-odds RSM: train acc={acc_logodds_train:.4f}, val acc={acc_logodds_val:.4f}")

    # Method 3: MLE RSM
    log.info("\n=== MLE RSM ===")
    rsm_mle = compute_mle_rsm(n, train_triplets, lr=args.lr, n_iters=args.n_iters)
    acc_mle_train = compute_triplet_accuracy(rsm_mle, train_triplets)
    acc_mle_val = compute_triplet_accuracy(rsm_mle, val_triplets)
    log.info(f"MLE RSM: train acc={acc_mle_train:.4f}, val acc={acc_mle_val:.4f}")

    # Now fit SRF to each RSM
    log.info("\n=== SRF Embeddings ===")
    from pysrf import SRF

    for name, rsm in [("counts", rsm_counts), ("logodds", rsm_logodds), ("mle", rsm_mle)]:
        log.info(f"\nFitting SRF on {name} RSM...")

        # Handle any NaNs
        rsm_clean = rsm.copy()
        nan_mask = np.isnan(rsm_clean)
        if nan_mask.any():
            rsm_clean[nan_mask] = 0.5  # Fill with neutral value
            log.info(f"  Filled {nan_mask.sum()} NaN values")

        model = SRF(rank=args.rank, random_state=args.seed)
        w = model.fit_transform(rsm_clean)

        acc_emb_train = compute_embedding_triplet_accuracy(w, train_triplets)
        acc_emb_val = compute_embedding_triplet_accuracy(w, val_triplets)
        log.info(f"  SRF({name}): train acc={acc_emb_train:.4f}, val acc={acc_emb_val:.4f}")

    # Compare with SPoSE
    log.info("\n=== Baseline: SPoSE ===")
    spose_path = Path("/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt")
    if spose_path.exists():
        spose = np.maximum(np.loadtxt(spose_path), 0)
        acc_spose_train = compute_embedding_triplet_accuracy(spose, train_triplets)
        acc_spose_val = compute_embedding_triplet_accuracy(spose, val_triplets)
        log.info(f"SPoSE: train acc={acc_spose_train:.4f}, val acc={acc_spose_val:.4f}")


if __name__ == "__main__":
    main()
