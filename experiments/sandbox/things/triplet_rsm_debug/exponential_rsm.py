#!/usr/bin/env python3
"""
Explore exponential-based RSM constructions inspired by SPoSE's softmax model.

Key insight: In softmax, P = exp(s) / Z, so s = log(P) + log(Z).
If we work in log-space, we might better capture the structure SPoSE learns.
"""
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax

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


def build_counts(n: int, triplets: np.ndarray):
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    return counts, shown


# ============================================================================
# Method 1: Standard (baseline)
# ============================================================================
def rsm_standard(counts, shown):
    rsm = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


# ============================================================================
# Method 2: Log-probability space
# ============================================================================
def rsm_log_prob(counts, shown, eps=0.5):
    """
    Work in log-probability space: s_ij ∝ log(P_ij)
    """
    # Smoothed probability
    p = (counts + eps) / (shown + 2 * eps)
    # Log transform
    s = np.log(p)
    # Shift to make minimum = 0
    s = s - s.min()
    # Scale so max off-diagonal = 1
    np.fill_diagonal(s, 0)
    s = s / (s.max() + 1e-10)
    np.fill_diagonal(s, 1.0)
    return s


# ============================================================================
# Method 3: Softmax-inspired normalization per triplet context
# ============================================================================
def rsm_softmax_normalized(n: int, triplets: np.ndarray, temp: float = 1.0):
    """
    For each triplet, instead of just counting wins, accumulate
    a softmax-weighted contribution based on the margin.

    If (i,j) won in triplet (i,j,k), we know:
    - s_ij > s_ik (probably)
    - s_ij > s_jk (probably)

    Weight the contribution by how confident we are.
    """
    # First pass: get raw counts for initial estimate
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1

    p_init = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)

    # Second pass: accumulate softmax-weighted scores
    scores = np.zeros((n, n))
    weights = np.zeros((n, n))

    for i, j, k in triplets:
        # Current probability estimates
        p_ij = p_init[i, j]
        p_ik = p_init[i, k]
        p_jk = p_init[j, k]

        # Softmax to get "confidence" weights
        logits = np.array([p_ij, p_ik, p_jk]) * temp
        probs = softmax(logits)

        # The winner (i,j) gets positive score proportional to confidence
        # The losers get negative scores
        score_ij = probs[0]  # How confident we are that (i,j) is most similar
        score_ik = probs[1]
        score_jk = probs[2]

        # Accumulate
        scores[i, j] += score_ij
        scores[j, i] += score_ij
        scores[i, k] += score_ik
        scores[k, i] += score_ik
        scores[j, k] += score_jk
        scores[k, j] += score_jk

        weights[i, j] += 1
        weights[j, i] += 1
        weights[i, k] += 1
        weights[k, i] += 1
        weights[j, k] += 1
        weights[k, j] += 1

    rsm = np.divide(scores, weights, out=0.5 * np.ones_like(scores), where=weights != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


# ============================================================================
# Method 4: Exponential of win rate
# ============================================================================
def rsm_exp_win_rate(counts, shown, scale: float = 2.0):
    """
    Instead of P, use exp(scale * P) to amplify differences.
    """
    p = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    # Exponential transformation
    s = np.exp(scale * (p - 0.5))  # Center at 0.5
    # Normalize to [0, 1]
    np.fill_diagonal(s, 0)
    s = (s - s.min()) / (s.max() - s.min())
    np.fill_diagonal(s, 1.0)
    return s


# ============================================================================
# Method 5: Bradley-Terry strength estimation
# ============================================================================
def rsm_bradley_terry(n: int, triplets: np.ndarray, n_iters: int = 20):
    """
    Bradley-Terry model: each pair has a "strength" parameter.
    P(ij beats ik) = strength_ij / (strength_ij + strength_ik)

    Iteratively estimate strengths from win/loss counts.
    """
    # Initialize strengths
    strength = np.ones((n, n)) * 0.5

    # Count wins and losses between pairs
    wins = {}  # (a,b) beat (c,d)
    for i, j, k in triplets:
        # (i,j) beat both (i,k) and (j,k)
        key1 = ((min(i,j), max(i,j)), (min(i,k), max(i,k)))
        key2 = ((min(i,j), max(i,j)), (min(j,k), max(j,k)))
        wins[key1] = wins.get(key1, 0) + 1
        wins[key2] = wins.get(key2, 0) + 1

    # Iterative update
    for iteration in range(n_iters):
        new_strength = np.ones((n, n)) * 0.5

        for ((a, b), (c, d)), win_count in wins.items():
            # Get reverse count
            reverse_key = ((c, d), (a, b))
            loss_count = wins.get(reverse_key, 0)
            total = win_count + loss_count

            if total > 0:
                # MLE for Bradley-Terry
                # P(ab wins) = s_ab / (s_ab + s_cd)
                # s_ab = wins / (wins/s_ab + losses/s_cd) ... but this is complex

                # Simple approximation: strength proportional to win rate
                win_rate = win_count / total
                new_strength[a, b] = max(new_strength[a, b], win_rate)
                new_strength[b, a] = new_strength[a, b]

        strength = new_strength

    np.fill_diagonal(strength, 1.0)
    return strength


# ============================================================================
# Method 6: Inverse softmax per triplet
# ============================================================================
def rsm_inverse_softmax(n: int, triplets: np.ndarray):
    """
    For each triplet where (i,j) won, we know (approximately):
    exp(s_ij) > exp(s_ik) and exp(s_ij) > exp(s_jk)

    So: s_ij > s_ik and s_ij > s_jk

    Accumulate these pairwise constraints.
    """
    # For each pair, accumulate "how much greater" it is than others
    greater_than = np.zeros((n, n))
    comparison_count = np.zeros((n, n))

    for i, j, k in triplets:
        # (i,j) > (i,k) and (i,j) > (j,k)
        # We'll give (i,j) a bonus and (i,k), (j,k) a penalty

        greater_than[i, j] += 2  # Won twice
        greater_than[j, i] += 2
        greater_than[i, k] -= 1  # Lost once
        greater_than[k, i] -= 1
        greater_than[j, k] -= 1
        greater_than[k, j] -= 1

        comparison_count[i, j] += 2
        comparison_count[j, i] += 2
        comparison_count[i, k] += 1
        comparison_count[k, i] += 1
        comparison_count[j, k] += 1
        comparison_count[k, j] += 1

    # Normalize by comparison count
    rsm = np.divide(greater_than, comparison_count,
                    out=np.zeros_like(greater_than), where=comparison_count != 0)

    # Shift and scale to [0, 1]
    np.fill_diagonal(rsm, 0)
    rsm = (rsm - rsm.min()) / (rsm.max() - rsm.min() + 1e-10)
    np.fill_diagonal(rsm, 1.0)
    return rsm


# ============================================================================
# Method 7: Exponential moving average during triplet processing
# ============================================================================
def rsm_exponential_update(n: int, triplets: np.ndarray, lr: float = 0.01):
    """
    Process triplets sequentially, updating RSM with exponential smoothing.
    This mimics online learning like SPoSE does.
    """
    rsm = np.ones((n, n)) * 0.5
    np.fill_diagonal(rsm, 1.0)

    for i, j, k in triplets:
        # Current estimates
        s_ij = rsm[i, j]
        s_ik = rsm[i, k]
        s_jk = rsm[j, k]

        # Softmax probabilities based on current estimates
        logits = np.array([s_ij, s_ik, s_jk])
        probs = softmax(logits)

        # Update: move toward observed outcome
        # (i,j) won, so increase s_ij, decrease s_ik, s_jk
        rsm[i, j] += lr * (1 - probs[0])
        rsm[j, i] = rsm[i, j]
        rsm[i, k] -= lr * probs[1]
        rsm[k, i] = rsm[i, k]
        rsm[j, k] -= lr * probs[2]
        rsm[k, j] = rsm[j, k]

    # Clip and normalize
    rsm = np.clip(rsm, 0, 1)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main():
    # Load data
    log.info("Loading 4.7M dataset...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    counts, shown = build_counts(n, train_triplets)

    # Baseline
    spose_acc = compute_triplet_accuracy(spose, val_triplets)
    log.info(f"\nBaseline SPoSE: {spose_acc:.4f}")
    log.info(f"Noise ceiling: 0.6667")

    from pysrf import SRF

    methods = {
        "standard": rsm_standard(counts, shown),
        "log_prob": rsm_log_prob(counts, shown),
        "exp_win_rate (s=1)": rsm_exp_win_rate(counts, shown, scale=1.0),
        "exp_win_rate (s=2)": rsm_exp_win_rate(counts, shown, scale=2.0),
        "exp_win_rate (s=3)": rsm_exp_win_rate(counts, shown, scale=3.0),
        "inverse_softmax": rsm_inverse_softmax(n, train_triplets),
        "softmax_norm (t=1)": rsm_softmax_normalized(n, train_triplets, temp=1.0),
        "softmax_norm (t=2)": rsm_softmax_normalized(n, train_triplets, temp=2.0),
    }

    log.info(f"\n{'='*50}")
    log.info("Exponential-based RSM Methods")
    log.info(f"{'='*50}")

    results = []
    for name, rsm in methods.items():
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        emb = model.fit_transform(rsm)
        acc = compute_triplet_accuracy(emb, val_triplets)
        log.info(f"{name}: {acc:.4f}")
        results.append((name, acc))

    # Try exponential update with different learning rates
    log.info(f"\n{'='*50}")
    log.info("Exponential Online Update")
    log.info(f"{'='*50}")

    for lr in [0.001, 0.005, 0.01, 0.02]:
        rsm = rsm_exponential_update(n, train_triplets, lr=lr)
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        emb = model.fit_transform(rsm)
        acc = compute_triplet_accuracy(emb, val_triplets)
        log.info(f"exp_update (lr={lr}): {acc:.4f}")
        results.append((f"exp_update (lr={lr})", acc))

    # Summary
    log.info(f"\n{'='*50}")
    log.info("Summary (sorted by accuracy)")
    log.info(f"{'='*50}")
    log.info(f"SPoSE: {spose_acc:.4f}")
    for name, acc in sorted(results, key=lambda x: -x[1]):
        delta = (acc - spose_acc) * 100
        log.info(f"{name}: {acc:.4f} ({delta:+.2f}% vs SPoSE)")


if __name__ == "__main__":
    main()
