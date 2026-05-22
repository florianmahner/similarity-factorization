"""Fast diagnostic: evaluate RSM transforms directly on triplets (no SRF).

If transform T of the RSM predicts triplets better than raw RSM,
then SRF on T should also beat SRF on raw RSM.

Takes ~10 seconds total.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.special import logit

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_spose_embedding, load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")
N = 1854


def triplet_acc_from_rsm(rsm, triplets):
    """Predict triplets directly from RSM entries (no factorization)."""
    idx = triplets.astype(int)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    sij = rsm[i, j]
    sik = rsm[i, k]
    sjk = rsm[j, k]
    valid = ~(np.isnan(sij) | np.isnan(sik) | np.isnan(sjk))
    correct = (sij > sik) & (sij > sjk)
    return float(np.mean(correct[valid])), float(np.mean(valid))


def main():
    train, val = load_triplets(DATA_DIR)
    spose = load_spose_embedding(DATA_DIR, num_dims=66)

    spose_rsm = spose @ spose.T
    acc_spose, _ = triplet_acc_from_rsm(spose_rsm, val)
    log.info("SPoSE RSM direct:  %.4f", acc_spose)

    s0 = compute_similarity_matrix_from_triplets(N, train, alpha=0.0)
    s1 = compute_similarity_matrix_from_triplets(N, train, alpha=1.0)

    acc0, cov0 = triplet_acc_from_rsm(s0, val)
    acc1, cov1 = triplet_acc_from_rsm(s1, val)
    log.info("Raw alpha=0:       %.4f (coverage=%.2f%%)", acc0, 100 * cov0)
    log.info("Raw alpha=1:       %.4f (coverage=%.2f%%)", acc1, 100 * cov1)

    # Logit transform (alpha=1 to avoid 0/1)
    eps = 0.01
    s1_clip = np.clip(s1, eps, 1 - eps)
    t_logit = logit(s1_clip)
    t_logit[np.isnan(s1)] = np.nan
    acc_logit, cov_logit = triplet_acc_from_rsm(t_logit, val)
    log.info("Logit(alpha=1):    %.4f (coverage=%.2f%%)", acc_logit, cov_logit)

    # Power transforms on alpha=0
    for gamma in [0.5, 2.0, 3.0]:
        s_pow = np.power(np.clip(s0, 0, None), gamma)
        s_pow[np.isnan(s0)] = np.nan
        acc_pow, _ = triplet_acc_from_rsm(s_pow, val)
        log.info("Power(%.1f) alpha=0: %.4f", gamma, acc_pow)

    # Odds ratio: S/(1-S)
    s1_clip2 = np.clip(s1, eps, 1 - eps)
    odds = s1_clip2 / (1 - s1_clip2)
    odds[np.isnan(s1)] = np.nan
    acc_odds, _ = triplet_acc_from_rsm(odds, val)
    log.info("Odds ratio:        %.4f", acc_odds)

    # Key insight check: how many triplets have ALL 3 pairs observed?
    log.info("\n=== Coverage ===")
    log.info("alpha=0: %.2f%% of val triplets fully observed", 100 * cov0)
    log.info("alpha=1: %.2f%% of val triplets fully observed", 100 * cov1)

    # Eigenvalue spectrum of the RSM
    s0_filled = np.where(np.isnan(s0), 0.5, s0)
    evals = np.linalg.eigvalsh(s0_filled)
    n_neg = np.sum(evals < 0)
    log.info("\n=== Spectrum ===")
    log.info("Negative eigenvalues: %d / %d", n_neg, N)
    log.info("Top-5 eigenvalues: %s", np.sort(evals)[::-1][:5].round(1))
    log.info("Energy in top-35: %.1f%%",
             100 * np.sum(np.sort(evals)[::-1][:35]) / np.sum(np.abs(evals)))


if __name__ == "__main__":
    main()
