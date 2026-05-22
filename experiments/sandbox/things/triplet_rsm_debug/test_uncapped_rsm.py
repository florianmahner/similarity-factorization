#!/usr/bin/env python3
"""
Test RSM construction that doesn't cap at 1.0.
Goal: Capture that some pairs are MUCH more similar than others.
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


def rsm_triplet_accuracy(rsm: np.ndarray, triplets: np.ndarray) -> float:
    correct = 0
    for i, j, k in triplets:
        sims = np.array([rsm[i, j], rsm[i, k], rsm[j, k]])
        if np.argmax(sims) == 0:
            correct += 1
    return correct / len(triplets)


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


def rsm_standard(counts, shown):
    rsm = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def rsm_log_odds_uncapped(counts, shown, scale: float = 1.0):
    """
    Log-odds don't cap at 1. If p -> 1, log-odds -> infinity.
    This naturally gives higher values to more certain similar pairs.
    """
    # Laplace smoothing
    p = (counts + 0.5) / (shown + 1)
    # Log-odds
    log_odds = np.log(p / (1 - p + 1e-10))
    # Scale and shift to be non-negative
    log_odds = scale * log_odds
    # Shift minimum to 0
    log_odds = log_odds - log_odds.min()
    np.fill_diagonal(log_odds, log_odds.max())
    return log_odds


def rsm_inverse_sigmoid(counts, shown, temp: float = 1.0):
    """
    Inverse of softmax: s = temp * log(p).
    If p is high, s is less negative. If p -> 1, s -> 0.
    """
    p = (counts + 0.5) / (shown + 1)
    s = temp * np.log(p + 1e-10)
    # Shift to positive
    s = s - s.min()
    np.fill_diagonal(s, s.max())
    return s


def rsm_bradley_terry(n: int, triplets: np.ndarray, n_iters: int = 10):
    """
    Bradley-Terry model for paired comparisons.
    Estimate "strength" parameter for each item, then s_ij = strength_i * strength_j.
    """
    # Initialize strengths
    strength = np.ones(n)

    for _ in range(n_iters):
        new_strength = np.zeros(n)
        for i in range(n):
            # Sum of comparisons involving item i
            wins = 0
            total_prob = 0
            for ii, jj, kk in triplets:
                if ii == i or jj == i:
                    # i was in the winning pair
                    wins += 1
                    # What was the expected probability?
                    if ii == i:
                        other_win = jj
                    else:
                        other_win = ii
                    other_lose = kk
                    p = (strength[i] * strength[other_win]) / (
                        strength[i] * strength[other_win] +
                        strength[i] * strength[other_lose] +
                        strength[other_win] * strength[other_lose] + 1e-10
                    )
                    total_prob += p
                elif kk == i:
                    # i was the odd one out
                    p = (strength[ii] * strength[jj]) / (
                        strength[ii] * strength[jj] +
                        strength[ii] * strength[i] +
                        strength[jj] * strength[i] + 1e-10
                    )
                    total_prob += (1 - p)  # Expected times i would "lose"

            if total_prob > 0:
                new_strength[i] = wins / total_prob
            else:
                new_strength[i] = strength[i]

        # Normalize
        strength = new_strength / new_strength.mean()

    # Build RSM from strengths
    rsm = np.outer(strength, strength)
    np.fill_diagonal(rsm, rsm.max())
    return rsm


def rsm_elo_style(n: int, triplets: np.ndarray, k: float = 32):
    """
    Elo-style rating update for each pair.
    """
    # Initialize ratings for each pair
    pair_rating = 0.5 * np.ones((n, n))

    for i, j, kk in triplets:
        # (i,j) won against (i,k) and (j,k)

        # Update (i,j) vs (i,k)
        expected_ij = pair_rating[i, j] / (pair_rating[i, j] + pair_rating[i, kk] + 1e-10)
        pair_rating[i, j] += k * (1 - expected_ij) / (len(triplets) / n)
        pair_rating[j, i] = pair_rating[i, j]
        pair_rating[i, kk] += k * (0 - (1 - expected_ij)) / (len(triplets) / n)
        pair_rating[kk, i] = pair_rating[i, kk]

        # Update (i,j) vs (j,k)
        expected_ij2 = pair_rating[i, j] / (pair_rating[i, j] + pair_rating[j, kk] + 1e-10)
        pair_rating[i, j] += k * (1 - expected_ij2) / (len(triplets) / n)
        pair_rating[j, i] = pair_rating[i, j]
        pair_rating[j, kk] += k * (0 - (1 - expected_ij2)) / (len(triplets) / n)
        pair_rating[kk, j] = pair_rating[j, kk]

    np.fill_diagonal(pair_rating, pair_rating.max())
    pair_rating = np.maximum(pair_rating, 0)
    return pair_rating


def rsm_weighted_by_margin(counts, shown, triplets, n):
    """
    Weight each win by how decisive it was (based on other pairs in that triplet).
    """
    weighted_counts = np.zeros((n, n))
    total_weight = np.zeros((n, n))

    # First compute standard RSM for estimating margins
    p = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)

    for i, j, k in triplets:
        # How decisive was this win?
        # If p[i,j] >> p[i,k] and p[i,j] >> p[j,k], it was easy
        # If p[i,j] ≈ p[i,k] ≈ p[j,k], it was hard
        margin = p[i, j] - max(p[i, k], p[j, k])
        weight = 1.0 / (1.0 + np.exp(-5 * margin))  # Higher weight for clearer wins

        for a, b in [(i, j), (i, k), (j, k)]:
            total_weight[a, b] += weight
            total_weight[b, a] += weight

        weighted_counts[i, j] += weight
        weighted_counts[j, i] += weight

    rsm = np.divide(weighted_counts, total_weight, out=0.5 * np.ones_like(counts), where=total_weight != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main():
    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    # Baseline
    log.info(f"\nBaseline SPoSE: {compute_triplet_accuracy(spose, val_triplets):.4f}")
    rsm_spose = spose @ spose.T
    log.info(f"SPoSE RSM direct: {rsm_triplet_accuracy(rsm_spose, val_triplets):.4f}")

    counts, shown = build_counts(n, train_triplets)

    from pysrf import SRF

    methods = {
        "standard": rsm_standard(counts, shown),
        "log_odds (scale=0.5)": rsm_log_odds_uncapped(counts, shown, scale=0.5),
        "log_odds (scale=1)": rsm_log_odds_uncapped(counts, shown, scale=1.0),
        "log_odds (scale=2)": rsm_log_odds_uncapped(counts, shown, scale=2.0),
        "inverse_sigmoid (t=0.5)": rsm_inverse_sigmoid(counts, shown, temp=0.5),
        "inverse_sigmoid (t=1)": rsm_inverse_sigmoid(counts, shown, temp=1.0),
        "inverse_sigmoid (t=2)": rsm_inverse_sigmoid(counts, shown, temp=2.0),
        "margin_weighted": rsm_weighted_by_margin(counts, shown, train_triplets, n),
    }

    log.info(f"\n=== RSM Methods ===")
    for name, rsm in methods.items():
        # Direct RSM accuracy
        direct_acc = rsm_triplet_accuracy(rsm, val_triplets)

        # SRF embedding accuracy
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm)
        emb_acc = compute_triplet_accuracy(w, val_triplets)

        log.info(f"{name}:")
        log.info(f"  RSM direct: {direct_acc:.4f}, SRF emb: {emb_acc:.4f}")

        # Show RSM stats
        off_diag = rsm[np.triu_indices(n, k=1)]
        log.info(f"  RSM range: [{off_diag.min():.2f}, {off_diag.max():.2f}], mean: {off_diag.mean():.2f}")


if __name__ == "__main__":
    main()
