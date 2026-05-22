#!/usr/bin/env python3
"""
Analyze which triplets are misclassified and why.
Find patterns that could inform better RSM construction.
"""
import logging
from pathlib import Path

import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def predict_triplet(embedding: np.ndarray, i: int, j: int, k: int) -> int:
    """Returns predicted choice: 0=(i,j), 1=(i,k), 2=(j,k)"""
    sims = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
    exp_sims = np.exp(sims - sims.max())  # Numerical stability
    return np.argmax(exp_sims)


def rsm_predict_triplet(rsm: np.ndarray, i: int, j: int, k: int) -> int:
    """Predict using RSM directly (no factorization)"""
    sims = np.array([rsm[i, j], rsm[i, k], rsm[j, k]])
    return np.argmax(sims)


def build_rsm(n: int, triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns RSM and observation counts"""
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
    return rsm, shown


def main():
    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    vice = np.loadtxt(data_dir / "vice_embedding_66d.txt")
    n = 1854

    # Build RSM and fit SRF
    log.info("Building RSM and fitting SRF...")
    rsm, shown = build_rsm(n, train_triplets)

    from pysrf import SRF
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    srf_emb = model.fit_transform(rsm)

    # Classify all validation triplets
    log.info("Classifying validation triplets...")
    results = []
    for idx, (i, j, k) in enumerate(val_triplets):
        srf_pred = predict_triplet(srf_emb, i, j, k)
        spose_pred = predict_triplet(spose, i, j, k)
        vice_pred = predict_triplet(vice, i, j, k)
        rsm_pred = rsm_predict_triplet(rsm, i, j, k)

        # Get RSM values and observation counts
        rsm_ij, rsm_ik, rsm_jk = rsm[i, j], rsm[i, k], rsm[j, k]
        n_ij, n_ik, n_jk = shown[i, j], shown[i, k], shown[j, k]

        results.append({
            'idx': idx, 'i': i, 'j': j, 'k': k,
            'srf_pred': srf_pred, 'spose_pred': spose_pred, 'vice_pred': vice_pred, 'rsm_pred': rsm_pred,
            'srf_correct': srf_pred == 0, 'spose_correct': spose_pred == 0,
            'vice_correct': vice_pred == 0, 'rsm_correct': rsm_pred == 0,
            'rsm_ij': rsm_ij, 'rsm_ik': rsm_ik, 'rsm_jk': rsm_jk,
            'n_ij': n_ij, 'n_ik': n_ik, 'n_jk': n_jk,
        })

    # Summary statistics
    srf_acc = sum(r['srf_correct'] for r in results) / len(results)
    spose_acc = sum(r['spose_correct'] for r in results) / len(results)
    vice_acc = sum(r['vice_correct'] for r in results) / len(results)
    rsm_acc = sum(r['rsm_correct'] for r in results) / len(results)

    log.info(f"\n=== Accuracy ===")
    log.info(f"SRF:   {srf_acc:.4f}")
    log.info(f"SPoSE: {spose_acc:.4f}")
    log.info(f"VICE:  {vice_acc:.4f}")
    log.info(f"RSM (direct): {rsm_acc:.4f}")

    # Analyze error patterns
    log.info(f"\n=== Error Analysis ===")

    # Where does SRF fail but SPoSE succeeds?
    srf_fail_spose_win = [r for r in results if not r['srf_correct'] and r['spose_correct']]
    srf_win_spose_fail = [r for r in results if r['srf_correct'] and not r['spose_correct']]
    both_fail = [r for r in results if not r['srf_correct'] and not r['spose_correct']]
    both_win = [r for r in results if r['srf_correct'] and r['spose_correct']]

    log.info(f"Both correct:     {len(both_win)} ({100*len(both_win)/len(results):.1f}%)")
    log.info(f"Both wrong:       {len(both_fail)} ({100*len(both_fail)/len(results):.1f}%)")
    log.info(f"SRF wrong, SPoSE right: {len(srf_fail_spose_win)} ({100*len(srf_fail_spose_win)/len(results):.1f}%)")
    log.info(f"SRF right, SPoSE wrong: {len(srf_win_spose_fail)} ({100*len(srf_win_spose_fail)/len(results):.1f}%)")

    # Analyze RSM values for different error categories
    log.info(f"\n=== RSM Properties by Error Category ===")

    def analyze_category(name, subset):
        if not subset:
            return
        # RSM margin: how much higher is rsm_ij vs max(rsm_ik, rsm_jk)?
        margins = [r['rsm_ij'] - max(r['rsm_ik'], r['rsm_jk']) for r in subset]
        # Observation counts
        min_obs = [min(r['n_ij'], r['n_ik'], r['n_jk']) for r in subset]
        # RSM prediction correctness
        rsm_correct_rate = sum(r['rsm_correct'] for r in subset) / len(subset)

        log.info(f"\n{name} (n={len(subset)}):")
        log.info(f"  RSM margin: mean={np.mean(margins):.4f}, std={np.std(margins):.4f}")
        log.info(f"  Min observations: mean={np.mean(min_obs):.1f}, median={np.median(min_obs):.1f}")
        log.info(f"  RSM direct accuracy: {rsm_correct_rate:.4f}")

    analyze_category("Both correct", both_win)
    analyze_category("Both wrong", both_fail)
    analyze_category("SRF wrong, SPoSE right", srf_fail_spose_win)
    analyze_category("SRF right, SPoSE wrong", srf_win_spose_fail)

    # Key insight: What makes SRF fail where RSM succeeds?
    log.info(f"\n=== RSM vs SRF Comparison ===")
    rsm_correct_srf_wrong = [r for r in results if r['rsm_correct'] and not r['srf_correct']]
    rsm_wrong_srf_correct = [r for r in results if not r['rsm_correct'] and r['srf_correct']]

    log.info(f"RSM correct, SRF wrong: {len(rsm_correct_srf_wrong)}")
    log.info(f"RSM wrong, SRF correct: {len(rsm_wrong_srf_correct)}")

    if rsm_correct_srf_wrong:
        # These are cases where factorization destroys the signal
        margins = [r['rsm_ij'] - max(r['rsm_ik'], r['rsm_jk']) for r in rsm_correct_srf_wrong]
        log.info(f"  RSM margin when RSM right/SRF wrong: mean={np.mean(margins):.4f}")

        # What's the distribution of SRF predictions?
        srf_preds = Counter(r['srf_pred'] for r in rsm_correct_srf_wrong)
        log.info(f"  SRF predictions: {dict(srf_preds)}")

    # Analyze by RSM margin bins
    log.info(f"\n=== Accuracy by RSM Margin ===")
    for lo, hi in [(-1, -0.1), (-0.1, -0.01), (-0.01, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 1)]:
        subset = [r for r in results if lo <= (r['rsm_ij'] - max(r['rsm_ik'], r['rsm_jk'])) < hi]
        if subset:
            srf_acc = sum(r['srf_correct'] for r in subset) / len(subset)
            spose_acc = sum(r['spose_correct'] for r in subset) / len(subset)
            rsm_acc = sum(r['rsm_correct'] for r in subset) / len(subset)
            log.info(f"  Margin [{lo:.2f}, {hi:.2f}): n={len(subset):6d}, RSM={rsm_acc:.3f}, SRF={srf_acc:.3f}, SPoSE={spose_acc:.3f}")

    # Analyze by observation count
    log.info(f"\n=== Accuracy by Min Observation Count ===")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)]:
        subset = [r for r in results if lo <= min(r['n_ij'], r['n_ik'], r['n_jk']) < hi]
        if subset:
            srf_acc = sum(r['srf_correct'] for r in subset) / len(subset)
            spose_acc = sum(r['spose_correct'] for r in subset) / len(subset)
            rsm_acc = sum(r['rsm_correct'] for r in subset) / len(subset)
            log.info(f"  Obs [{lo:3d}, {hi:3d}): n={len(subset):6d}, RSM={rsm_acc:.3f}, SRF={srf_acc:.3f}, SPoSE={spose_acc:.3f}")

    # Check if SRF distorts the relative ordering
    log.info(f"\n=== Ordering Preservation Analysis ===")
    # For each triplet, check if the ordering of similarities is preserved
    ordering_preserved = 0
    ordering_flipped = 0
    for r in results:
        i, j, k = r['i'], r['j'], r['k']
        rsm_order = np.argsort([rsm[i,j], rsm[i,k], rsm[j,k]])[::-1]
        srf_sims = [srf_emb[i] @ srf_emb[j], srf_emb[i] @ srf_emb[k], srf_emb[j] @ srf_emb[k]]
        srf_order = np.argsort(srf_sims)[::-1]
        if np.array_equal(rsm_order, srf_order):
            ordering_preserved += 1
        else:
            ordering_flipped += 1

    log.info(f"Ordering preserved: {ordering_preserved} ({100*ordering_preserved/len(results):.1f}%)")
    log.info(f"Ordering flipped:   {ordering_flipped} ({100*ordering_flipped/len(results):.1f}%)")


if __name__ == "__main__":
    main()
