#!/usr/bin/env python3
"""
Alternative triplet-to-similarity models that DON'T assume softmax.

The goal: Find a similarity representation that captures human behavior
better than the softmax assumption used by SPoSE/VICE.

Key alternatives:
1. Thurstone Model (Gaussian noise instead of Gumbel)
2. Linear probability model
3. Ordinal embedding (margin-based)
4. Adaptive kernel estimation
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.special import softmax
from numba import njit, prange
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Evaluation functions
# ============================================================================

def compute_rsm_triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    """Triplet accuracy using RSM similarities directly (argmax of 3 sims)."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([s[i, j], s[i, k], s[j, k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


def compute_softmax_triplet_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    """Triplet accuracy using softmax on dot products (SPoSE style)."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        probs = softmax(sims)
        if np.argmax(probs) == 0:
            correct += 1
    return correct / len(triplets)


def compute_argmax_triplet_accuracy(w: np.ndarray, triplets: np.ndarray) -> float:
    """Triplet accuracy using argmax on dot products (no softmax)."""
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


# ============================================================================
# Method 1: Count-based (baseline)
# ============================================================================

def compute_count_rsm(n: int, triplets: np.ndarray) -> np.ndarray:
    """Standard count-based RSM."""
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


# ============================================================================
# Method 2: Thurstone Model (probit instead of logit)
# ============================================================================

def thurstone_probability(s_ij: float, s_ik: float, s_jk: float, sigma: float = 1.0) -> float:
    """
    Thurstone's model: Each similarity has Gaussian noise.
    P(choose i,j) = P(s_ij + e_ij > s_ik + e_ik AND s_ij + e_ij > s_jk + e_jk)

    For simplicity, approximate as product of two probit terms.
    """
    p1 = norm.cdf((s_ij - s_ik) / (sigma * np.sqrt(2)))
    p2 = norm.cdf((s_ij - s_jk) / (sigma * np.sqrt(2)))
    return p1 * p2


def compute_thurstone_rsm(
    n: int,
    triplets: np.ndarray,
    lr: float = 0.1,
    n_iters: int = 50,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Estimate RSM using Thurstone model via gradient descent.
    """
    # Initialize
    s = compute_count_rsm(n, triplets)

    for it in tqdm(range(n_iters), desc="Thurstone"):
        grad = np.zeros_like(s)
        loss = 0.0

        for i, j, k in triplets:
            # Current similarities
            sij, sik, sjk = s[i, j], s[i, k], s[j, k]

            # Probit terms
            d1 = (sij - sik) / (sigma * np.sqrt(2))
            d2 = (sij - sjk) / (sigma * np.sqrt(2))

            p1 = norm.cdf(d1)
            p2 = norm.cdf(d2)
            prob = p1 * p2 + 1e-10

            loss -= np.log(prob)

            # Gradients via chain rule
            pdf1 = norm.pdf(d1) / (sigma * np.sqrt(2))
            pdf2 = norm.pdf(d2) / (sigma * np.sqrt(2))

            # d_loss/d_sij
            g_sij = -(pdf1 * p2 + p1 * pdf2) / prob
            grad[i, j] += g_sij
            grad[j, i] += g_sij

            # d_loss/d_sik
            g_sik = (pdf1 * p2) / prob
            grad[i, k] += g_sik
            grad[k, i] += g_sik

            # d_loss/d_sjk
            g_sjk = (p1 * pdf2) / prob
            grad[j, k] += g_sjk
            grad[k, j] += g_sjk

        s -= lr * grad / len(triplets)
        s = (s + s.T) / 2  # Symmetrize

        if it % 10 == 0:
            acc = compute_rsm_triplet_accuracy(s, triplets)
            log.info(f"  Iter {it}: loss={loss/len(triplets):.4f}, acc={acc:.4f}")

    # Normalize
    np.fill_diagonal(s, np.max(s))
    s = (s - s.min()) / (s.max() - s.min())
    np.fill_diagonal(s, 1.0)
    return s


# ============================================================================
# Method 3: Margin-based ordinal embedding
# ============================================================================

def compute_margin_rsm(
    n: int,
    triplets: np.ndarray,
    margin: float = 0.1,
    lr: float = 0.01,
    n_iters: int = 50,
) -> np.ndarray:
    """
    Ordinal embedding: enforce s_ij > s_ik + margin and s_ij > s_jk + margin.
    Uses hinge loss instead of log-likelihood.
    """
    s = compute_count_rsm(n, triplets)

    for it in tqdm(range(n_iters), desc="Margin"):
        grad = np.zeros_like(s)
        loss = 0.0
        violations = 0

        for i, j, k in triplets:
            sij, sik, sjk = s[i, j], s[i, k], s[j, k]

            # Hinge loss for s_ij > s_ik + margin
            if sij < sik + margin:
                loss += (sik + margin - sij)
                grad[i, j] -= 1
                grad[j, i] -= 1
                grad[i, k] += 1
                grad[k, i] += 1
                violations += 1

            # Hinge loss for s_ij > s_jk + margin
            if sij < sjk + margin:
                loss += (sjk + margin - sij)
                grad[i, j] -= 1
                grad[j, i] -= 1
                grad[j, k] += 1
                grad[k, j] += 1
                violations += 1

        s -= lr * grad / len(triplets)
        s = (s + s.T) / 2
        s = np.clip(s, 0, None)  # Keep non-negative

        if it % 10 == 0:
            acc = compute_rsm_triplet_accuracy(s, triplets)
            log.info(f"  Iter {it}: loss={loss/len(triplets):.4f}, violations={violations}, acc={acc:.4f}")

    np.fill_diagonal(s, np.max(s))
    s = (s - s.min()) / (s.max() - s.min())
    np.fill_diagonal(s, 1.0)
    return s


# ============================================================================
# Method 4: Weighted count with context correction
# ============================================================================

def compute_context_corrected_rsm(n: int, triplets: np.ndarray) -> np.ndarray:
    """
    The problem with raw counts: easy triplets (where k is very different)
    contribute the same as hard triplets.

    Solution: Weight each vote by how informative it is.
    - If all 3 items are similar, the vote is very informative
    - If k is very different, the vote is less informative

    We estimate this iteratively.
    """
    # First pass: raw counts
    s = compute_count_rsm(n, triplets)

    for iteration in range(5):
        log.info(f"  Context correction iteration {iteration}")
        weighted_counts = np.zeros((n, n))
        weights = np.zeros((n, n))

        for i, j, k in triplets:
            # How "hard" is this triplet based on current estimate?
            sij, sik, sjk = s[i, j], s[i, k], s[j, k]

            # Difficulty = how close are the three similarities
            max_other = max(sik, sjk)
            difficulty = 1.0 / (1.0 + np.exp(5 * (sij - max_other)))  # Sigmoid

            # Weight = difficulty (hard triplets are more informative)
            w = 0.5 + 0.5 * difficulty  # Range [0.5, 1.0]

            # Accumulate weighted votes
            for a, b in [(i, j), (i, k), (j, k)]:
                weights[a, b] += w
                weights[b, a] += w

            weighted_counts[i, j] += w
            weighted_counts[j, i] += w

        # Update similarity
        with np.errstate(divide='ignore', invalid='ignore'):
            s = weighted_counts / weights
        s[~np.isfinite(s)] = 0.5
        np.fill_diagonal(s, 1.0)

    return s


# ============================================================================
# Method 5: Inverse softmax (recover s from P)
# ============================================================================

def compute_inverse_softmax_rsm(n: int, triplets: np.ndarray, reg: float = 0.1) -> np.ndarray:
    """
    If P(i,j|i,j,k) = softmax([s_ij, s_ik, s_jk])[0], can we invert?

    For each triplet, if we observe choice (i,j), we know:
        s_ij > s_ik  (probably)
        s_ij > s_jk  (probably)

    Collect all such pairwise constraints and solve via least squares.
    """
    # Build constraint matrix: for each triplet, s_ij - s_ik > 0 and s_ij - s_jk > 0
    # We'll set up a linear system with slack

    # Simpler approach: for each pair, estimate relative strength
    pair_wins = {}  # (a,b) -> number of triplets where s_ab > s_other

    for i, j, k in triplets:
        # (i,j) beat (i,k)
        pair_wins[(i, j, i, k)] = pair_wins.get((i, j, i, k), 0) + 1
        pair_wins[(i, k, i, j)] = pair_wins.get((i, k, i, j), 0)

        # (i,j) beat (j,k)
        pair_wins[(i, j, j, k)] = pair_wins.get((i, j, j, k), 0) + 1
        pair_wins[(j, k, i, j)] = pair_wins.get((j, k, i, j), 0)

    # For each pair (a,b), compute Bradley-Terry style strength
    s = np.zeros((n, n))
    count = np.zeros((n, n))

    for (a, b, c, d), wins in pair_wins.items():
        # This means in comparisons between (a,b) and (c,d), (a,b) won 'wins' times
        total = pair_wins.get((a, b, c, d), 0) + pair_wins.get((c, d, a, b), 0)
        if total > 0:
            # (a,b) is stronger than (c,d) by this margin
            margin = (wins / total) - 0.5  # Range [-0.5, 0.5]
            s[a, b] += margin
            s[b, a] += margin
            s[c, d] -= margin
            s[d, c] -= margin
            count[a, b] += 1
            count[b, a] += 1
            count[c, d] += 1
            count[d, c] += 1

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        s = s / (count + 1)
    s[~np.isfinite(s)] = 0

    # Shift and scale
    s = s - s.min()
    s = s / (s.max() + 1e-10)
    np.fill_diagonal(s, 1.0)
    return s


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=66)
    parser.add_argument("--n-iters", type=int, default=30)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things/triplets_47")
    train_triplets = np.loadtxt(data_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "validationset.txt").astype(int)
    n = 1854

    # Subsample
    if args.subsample < 1.0:
        n_samples = int(len(train_triplets) * args.subsample)
        idx = np.random.choice(len(train_triplets), size=n_samples, replace=False)
        train_triplets = train_triplets[idx]
        log.info(f"Subsampled to {len(train_triplets)} triplets")

    # Load SPoSE baseline
    spose_path = Path("/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt")
    spose = np.maximum(np.loadtxt(spose_path), 0)

    log.info("\n=== Baseline: SPoSE ===")
    acc_spose_train = compute_softmax_triplet_accuracy(spose, train_triplets)
    acc_spose_val = compute_softmax_triplet_accuracy(spose, val_triplets)
    log.info(f"SPoSE (softmax): train={acc_spose_train:.4f}, val={acc_spose_val:.4f}")

    acc_spose_argmax_train = compute_argmax_triplet_accuracy(spose, train_triplets)
    acc_spose_argmax_val = compute_argmax_triplet_accuracy(spose, val_triplets)
    log.info(f"SPoSE (argmax):  train={acc_spose_argmax_train:.4f}, val={acc_spose_argmax_val:.4f}")

    results = {"SPoSE": acc_spose_val}

    # Method 1: Count-based
    log.info("\n=== Method 1: Count-based RSM ===")
    rsm_count = compute_count_rsm(n, train_triplets)
    acc = compute_rsm_triplet_accuracy(rsm_count, val_triplets)
    log.info(f"Count RSM: val acc={acc:.4f}")
    results["Count"] = acc

    # Method 2: Context-corrected
    log.info("\n=== Method 2: Context-corrected RSM ===")
    rsm_context = compute_context_corrected_rsm(n, train_triplets)
    acc = compute_rsm_triplet_accuracy(rsm_context, val_triplets)
    log.info(f"Context-corrected RSM: val acc={acc:.4f}")
    results["Context"] = acc

    # Method 3: Inverse softmax / Bradley-Terry
    log.info("\n=== Method 3: Bradley-Terry style ===")
    rsm_bt = compute_inverse_softmax_rsm(n, train_triplets)
    acc = compute_rsm_triplet_accuracy(rsm_bt, val_triplets)
    log.info(f"Bradley-Terry RSM: val acc={acc:.4f}")
    results["BT"] = acc

    # Method 4: Margin-based
    log.info("\n=== Method 4: Margin-based RSM ===")
    rsm_margin = compute_margin_rsm(n, train_triplets, n_iters=args.n_iters, lr=0.05)
    acc = compute_rsm_triplet_accuracy(rsm_margin, val_triplets)
    log.info(f"Margin RSM: val acc={acc:.4f}")
    results["Margin"] = acc

    # Now fit SRF and compare
    log.info("\n=== SRF Embeddings ===")
    from pysrf import SRF

    for name, rsm in [("count", rsm_count), ("context", rsm_context), ("margin", rsm_margin)]:
        rsm_clean = rsm.copy()
        rsm_clean[~np.isfinite(rsm_clean)] = 0.5

        model = SRF(rank=args.rank, random_state=args.seed)
        w = model.fit_transform(rsm_clean)

        # Test both softmax and argmax evaluation
        acc_softmax = compute_softmax_triplet_accuracy(w, val_triplets)
        acc_argmax = compute_argmax_triplet_accuracy(w, val_triplets)
        log.info(f"SRF({name}): softmax={acc_softmax:.4f}, argmax={acc_argmax:.4f}")

    # Summary
    log.info("\n=== Summary ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        log.info(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
