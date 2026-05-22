#!/usr/bin/env python3
"""
Analyze what makes SPoSE's RSM structure better for triplet prediction.
"""
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


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


def main():
    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    counts, shown = build_counts(n, train_triplets)
    rsm_counts = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm_counts, 1.0)

    rsm_spose = spose @ spose.T

    # Analyze per-triplet predictions
    log.info(f"\n=== Per-Triplet Analysis ===")

    # For each validation triplet, get:
    # 1. The three RSM values (counts-based)
    # 2. The three RSM values (SPoSE-based)
    # 3. Whether each method predicts correctly

    counts_margins = []
    spose_margins = []
    counts_correct = []
    spose_correct = []

    for i, j, k in val_triplets:
        # Counts-based
        c_ij, c_ik, c_jk = rsm_counts[i, j], rsm_counts[i, k], rsm_counts[j, k]
        c_margin = c_ij - max(c_ik, c_jk)
        counts_margins.append(c_margin)
        counts_correct.append(c_margin > 0)

        # SPoSE-based
        s_ij, s_ik, s_jk = rsm_spose[i, j], rsm_spose[i, k], rsm_spose[j, k]
        s_margin = s_ij - max(s_ik, s_jk)
        spose_margins.append(s_margin)
        spose_correct.append(s_margin > 0)

    counts_margins = np.array(counts_margins)
    spose_margins = np.array(spose_margins)

    # Correlation between margins
    log.info(f"Margin correlation (Counts vs SPoSE): {pearsonr(counts_margins, spose_margins)[0]:.4f}")

    # Where do they disagree?
    counts_only = np.array(counts_correct) & ~np.array(spose_correct)
    spose_only = ~np.array(counts_correct) & np.array(spose_correct)

    log.info(f"\nCounts correct, SPoSE wrong: {counts_only.sum()} ({100*counts_only.mean():.1f}%)")
    log.info(f"SPoSE correct, Counts wrong: {spose_only.sum()} ({100*spose_only.mean():.1f}%)")

    # What's different about the triplets where SPoSE is right but Counts is wrong?
    log.info(f"\n=== Triplets where SPoSE wins ===")
    spose_wins_idx = np.where(spose_only)[0]

    if len(spose_wins_idx) > 0:
        # Look at the counts margin for these
        counts_margins_spose_wins = counts_margins[spose_wins_idx]
        spose_margins_spose_wins = spose_margins[spose_wins_idx]

        log.info(f"Counts margin: mean={counts_margins_spose_wins.mean():.4f}, std={counts_margins_spose_wins.std():.4f}")
        log.info(f"SPoSE margin: mean={spose_margins_spose_wins.mean():.4f}, std={spose_margins_spose_wins.std():.4f}")

        # What makes SPoSE flip the prediction?
        # Look at the actual RSM values
        sample_triplets = val_triplets[spose_wins_idx[:100]]
        log.info(f"\nSample triplets where SPoSE wins:")

        for idx, (i, j, k) in enumerate(sample_triplets[:10]):
            c_ij, c_ik, c_jk = rsm_counts[i, j], rsm_counts[i, k], rsm_counts[j, k]
            s_ij, s_ik, s_jk = rsm_spose[i, j], rsm_spose[i, k], rsm_spose[j, k]
            log.info(f"  Triplet {i},{j},{k}:")
            log.info(f"    Counts: ij={c_ij:.3f}, ik={c_ik:.3f}, jk={c_jk:.3f} -> max other: {max(c_ik, c_jk):.3f}")
            log.info(f"    SPoSE:  ij={s_ij:.3f}, ik={s_ik:.3f}, jk={s_jk:.3f} -> max other: {max(s_ik, s_jk):.3f}")

    # Key question: is there a systematic relationship between counts and SPoSE values?
    log.info(f"\n=== Systematic Relationship ===")

    # For each pair, compute counts RSM value and SPoSE RSM value
    triu_idx = np.triu_indices(n, k=1)
    counts_flat = rsm_counts[triu_idx]
    spose_flat = rsm_spose[triu_idx]
    shown_flat = shown[triu_idx]

    # Bin by counts value and see average SPoSE value
    log.info(f"\nSPoSE value by Counts bin:")
    for lo, hi in [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]:
        mask = (counts_flat >= lo) & (counts_flat < hi)
        if mask.sum() > 0:
            avg_spose = spose_flat[mask].mean()
            std_spose = spose_flat[mask].std()
            avg_shown = shown_flat[mask].mean()
            log.info(f"  Counts [{lo:.2f}, {hi:.2f}): n={mask.sum():6d}, SPoSE mean={avg_spose:.3f} ± {std_spose:.3f}, obs={avg_shown:.1f}")

    # What transformation would map counts to SPoSE?
    log.info(f"\n=== Finding Mapping from Counts to SPoSE ===")

    # Simple polynomial fit
    from numpy.polynomial import polynomial as P
    # Fit polynomial: spose = a + b*counts + c*counts^2 + ...
    coeffs = np.polyfit(counts_flat, spose_flat, deg=3)
    spose_predicted = np.polyval(coeffs, counts_flat)
    residual = np.sqrt(np.mean((spose_flat - spose_predicted) ** 2))
    log.info(f"Polynomial fit (deg=3): RMSE={residual:.3f}")
    log.info(f"Coefficients: {coeffs}")

    # Apply this transformation to the full RSM
    rsm_transformed = np.polyval(coeffs, rsm_counts)
    np.fill_diagonal(rsm_transformed, rsm_transformed.max())

    # Test it
    def rsm_triplet_accuracy(rsm, triplets):
        correct = 0
        for i, j, k in triplets:
            if np.argmax([rsm[i, j], rsm[i, k], rsm[j, k]]) == 0:
                correct += 1
        return correct / len(triplets)

    log.info(f"\nTransformed RSM accuracy: {rsm_triplet_accuracy(rsm_transformed, val_triplets):.4f}")
    log.info(f"Original counts RSM accuracy: {rsm_triplet_accuracy(rsm_counts, val_triplets):.4f}")
    log.info(f"SPoSE RSM accuracy: {rsm_triplet_accuracy(rsm_spose, val_triplets):.4f}")

    # Now factorize the transformed RSM
    from pysrf import SRF

    def compute_triplet_accuracy(emb, triplets):
        correct = 0
        for i, j, k in triplets:
            sims = np.array([emb[i] @ emb[j], emb[i] @ emb[k], emb[j] @ emb[k]])
            exp_sims = np.exp(sims - sims.max())
            if np.argmax(exp_sims) == 0:
                correct += 1
        return correct / len(triplets)

    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm_transformed)
    log.info(f"\nSRF on transformed RSM: {compute_triplet_accuracy(w, val_triplets):.4f}")


if __name__ == "__main__":
    main()
