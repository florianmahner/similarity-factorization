"""Fast test: rank-K approximation in different spaces.

The key question: does doing rank-K approx in logit space
preserve ordinal structure better than in proportion space?

Uses eigendecomposition (instant) instead of SRF (slow).
"""

import logging
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_spose_embedding, load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N = 1854


def triplet_acc(rsm, triplets):
    idx = triplets.astype(int)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    sij = rsm[i, j]
    sik = rsm[i, k]
    sjk = rsm[j, k]
    valid = ~(np.isnan(sij) | np.isnan(sik) | np.isnan(sjk))
    return float(np.mean((sij[valid] > sik[valid]) & (sij[valid] > sjk[valid])))


def rank_k_psd(matrix, k):
    """Best rank-K PSD approximation."""
    vals, vecs = np.linalg.eigh(matrix)
    idx = np.argsort(vals)[::-1][:k]
    idx = idx[vals[idx] > 0]
    return vecs[:, idx] @ np.diag(vals[idx]) @ vecs[:, idx].T


def rank_k_symmetric(matrix, k):
    """Best rank-K symmetric approximation (allows negative eigenvalues)."""
    vals, vecs = np.linalg.eigh(matrix)
    idx = np.argsort(np.abs(vals))[::-1][:k]
    return vecs[:, idx] @ np.diag(vals[idx]) @ vecs[:, idx].T


def main():
    train, val = load_triplets(DATA_DIR)
    spose = load_spose_embedding(DATA_DIR, num_dims=66)

    spose_rsm = spose @ spose.T
    log.info("SPoSE:               %.4f", triplet_acc(spose_rsm, val))
    log.info("")

    # Build RSMs
    s0 = compute_similarity_matrix_from_triplets(N, train, alpha=0.0)
    s1 = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)
    s0_filled = np.where(np.isnan(s0), 0.5, s0)
    s1_filled = np.where(np.isnan(s1), 0.5, s1)

    # === Approach 1: Rank-K PSD in proportion space ===
    log.info("=== Rank-K PSD in PROPORTION space (alpha=0) ===")
    for k in [20, 30, 35, 40, 50, 66, 100]:
        s_k = rank_k_psd(s0_filled, k)
        log.info("  k=%3d: %.4f", k, triplet_acc(s_k, val))

    # === Approach 2: Rank-K in LOGIT space, then sigmoid back ===
    log.info("\n=== Rank-K in LOGIT space -> sigmoid (alpha=1) ===")
    eps = 0.02
    s1_clip = np.clip(s1_filled, eps, 1 - eps)
    t = logit(s1_clip)
    for k in [20, 30, 35, 40, 50, 66, 100]:
        t_k = rank_k_symmetric(t, k)
        s_back = expit(t_k)
        log.info("  k=%3d: %.4f", k, triplet_acc(s_back, val))

    # === Approach 3: Rank-K in LOGIT space, evaluate in logit space ===
    log.info("\n=== Rank-K in LOGIT space, evaluate as-is ===")
    for k in [20, 30, 35, 40, 50, 66, 100]:
        t_k = rank_k_symmetric(t, k)
        log.info("  k=%3d: %.4f", k, triplet_acc(t_k, val))

    # === Approach 4: Rank-K PSD in centered proportion space ===
    log.info("\n=== Rank-K PSD in CENTERED proportion space (alpha=0) ===")
    s0_centered = s0_filled - np.mean(s0_filled)
    for k in [20, 30, 35, 40, 50, 66, 100]:
        s_k = rank_k_symmetric(s0_centered, k)
        log.info("  k=%3d: %.4f", k, triplet_acc(s_k, val))

    # === Approach 5: Double-centered ===
    log.info("\n=== Rank-K in DOUBLE-CENTERED proportion space ===")
    row_mean = np.mean(s0_filled, axis=1, keepdims=True)
    col_mean = np.mean(s0_filled, axis=0, keepdims=True)
    grand_mean = np.mean(s0_filled)
    s0_dc = s0_filled - row_mean - col_mean + grand_mean
    for k in [20, 30, 35, 40, 50, 66, 100]:
        s_k = rank_k_symmetric(s0_dc, k)
        log.info("  k=%3d: %.4f", k, triplet_acc(s_k, val))

    # === Approach 6: Rank-K in odds space ===
    log.info("\n=== Rank-K PSD in ODDS space (alpha=1) ===")
    odds = s1_clip / (1 - s1_clip)
    for k in [20, 30, 35, 40, 50, 66, 100]:
        o_k = rank_k_psd(odds, k)
        log.info("  k=%3d: %.4f", k, triplet_acc(o_k, val))


if __name__ == "__main__":
    main()
