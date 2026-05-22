#!/usr/bin/env python3
"""
Compare the structure of count-based RSM vs SPoSE-implied RSM.
Goal: Understand what makes SPoSE better for triplet prediction.
"""
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build_rsm(n: int, triplets: np.ndarray) -> np.ndarray:
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    rsm = np.divide(counts, shown, out=0.5 * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm


def main():
    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    vice = np.loadtxt(data_dir / "vice_embedding_66d.txt")
    n = 1854

    # Build count-based RSM
    rsm_counts = build_rsm(n, train_triplets)

    # SPoSE-implied RSM (dot product)
    rsm_spose = spose @ spose.T
    rsm_vice = vice @ vice.T

    # SRF embedding
    from pysrf import SRF
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    srf_emb = model.fit_transform(rsm_counts)
    rsm_srf = srf_emb @ srf_emb.T

    # Compare RSMs
    log.info(f"\n=== RSM Statistics ===")
    for name, rsm in [("Counts", rsm_counts), ("SPoSE", rsm_spose), ("VICE", rsm_vice), ("SRF", rsm_srf)]:
        off_diag = rsm[np.triu_indices(n, k=1)]
        log.info(f"{name}:")
        log.info(f"  Range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
        log.info(f"  Mean: {off_diag.mean():.3f}, Std: {off_diag.std():.3f}")

    # Correlations between RSMs
    log.info(f"\n=== RSM Correlations (upper triangle) ===")
    triu_idx = np.triu_indices(n, k=1)

    counts_flat = rsm_counts[triu_idx]
    spose_flat = rsm_spose[triu_idx]
    vice_flat = rsm_vice[triu_idx]
    srf_flat = rsm_srf[triu_idx]

    log.info(f"Counts vs SPoSE: r={pearsonr(counts_flat, spose_flat)[0]:.4f}")
    log.info(f"Counts vs VICE:  r={pearsonr(counts_flat, vice_flat)[0]:.4f}")
    log.info(f"Counts vs SRF:   r={pearsonr(counts_flat, srf_flat)[0]:.4f}")
    log.info(f"SPoSE vs VICE:   r={pearsonr(spose_flat, vice_flat)[0]:.4f}")
    log.info(f"SPoSE vs SRF:    r={pearsonr(spose_flat, srf_flat)[0]:.4f}")

    # Where do they disagree most?
    log.info(f"\n=== Largest Disagreements ===")
    diff_spose_counts = np.abs(spose_flat - counts_flat)
    diff_srf_spose = np.abs(srf_flat - spose_flat)

    log.info(f"SPoSE vs Counts: max diff = {diff_spose_counts.max():.3f}")
    log.info(f"SRF vs SPoSE: max diff = {diff_srf_spose.max():.3f}")

    # Analyze pairs where SPoSE and Counts disagree significantly
    log.info(f"\n=== Pairs with Large SPoSE vs Counts Difference ===")
    large_diff_mask = diff_spose_counts > np.percentile(diff_spose_counts, 99)

    log.info(f"Number of pairs with >99th percentile difference: {large_diff_mask.sum()}")
    log.info(f"In these pairs:")
    log.info(f"  Counts mean: {counts_flat[large_diff_mask].mean():.3f}")
    log.info(f"  SPoSE mean: {spose_flat[large_diff_mask].mean():.3f}")

    # Which direction is the difference?
    spose_higher = (spose_flat > counts_flat) & large_diff_mask
    counts_higher = (counts_flat > spose_flat) & large_diff_mask
    log.info(f"  SPoSE higher: {spose_higher.sum()}")
    log.info(f"  Counts higher: {counts_higher.sum()}")

    # Now the key question: for triplet prediction, what matters?
    log.info(f"\n=== Triplet Prediction Analysis ===")

    def triplet_accuracy_from_rsm(rsm, triplets):
        correct = 0
        for i, j, k in triplets:
            sims = np.array([rsm[i, j], rsm[i, k], rsm[j, k]])
            if np.argmax(sims) == 0:
                correct += 1
        return correct / len(triplets)

    log.info(f"Val accuracy using RSM directly:")
    log.info(f"  Counts: {triplet_accuracy_from_rsm(rsm_counts, val_triplets):.4f}")
    log.info(f"  SPoSE:  {triplet_accuracy_from_rsm(rsm_spose, val_triplets):.4f}")
    log.info(f"  VICE:   {triplet_accuracy_from_rsm(rsm_vice, val_triplets):.4f}")
    log.info(f"  SRF:    {triplet_accuracy_from_rsm(rsm_srf, val_triplets):.4f}")

    # Rank correlation might be more relevant than Pearson
    log.info(f"\n=== Rank Correlations ===")
    log.info(f"Counts vs SPoSE (Spearman): {spearmanr(counts_flat, spose_flat)[0]:.4f}")
    log.info(f"Counts vs SRF (Spearman):   {spearmanr(counts_flat, srf_flat)[0]:.4f}")

    # What if we use SPoSE's RSM structure to create a "target" RSM, then factorize that?
    log.info(f"\n=== Factorizing SPoSE RSM (sanity check) ===")
    # Normalize SPoSE RSM to [0, 1]
    rsm_spose_norm = (rsm_spose - rsm_spose.min()) / (rsm_spose.max() - rsm_spose.min())
    np.fill_diagonal(rsm_spose_norm, 1.0)

    model2 = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    srf_from_spose = model2.fit_transform(rsm_spose_norm)

    def compute_triplet_accuracy(emb, triplets):
        correct = 0
        for i, j, k in triplets:
            sims = np.array([emb[i] @ emb[j], emb[i] @ emb[k], emb[j] @ emb[k]])
            exp_sims = np.exp(sims - sims.max())
            if np.argmax(exp_sims) == 0:
                correct += 1
        return correct / len(triplets)

    log.info(f"SRF fitted to SPoSE RSM: {compute_triplet_accuracy(srf_from_spose, val_triplets):.4f}")
    log.info(f"Original SPoSE: {compute_triplet_accuracy(spose, val_triplets):.4f}")


if __name__ == "__main__":
    main()
