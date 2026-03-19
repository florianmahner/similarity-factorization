#!/usr/bin/env python3
"""
Analyze cases where RSM has the right signal but SRF destroys it.
Goal: Understand what makes these pairs vulnerable to factorization.
"""
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def predict_triplet(embedding: np.ndarray, i: int, j: int, k: int) -> int:
    sims = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
    exp_sims = np.exp(sims - sims.max())
    return np.argmax(exp_sims)


def build_rsm(n: int, triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return rsm, shown, counts


def main():
    # Load data
    log.info("Loading data...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_47/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_47/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    # Build RSM
    rsm, shown, counts = build_rsm(n, train_triplets)

    from pysrf import SRF
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    srf_emb = model.fit_transform(rsm)

    # Compute SRF reconstruction
    srf_rsm = srf_emb @ srf_emb.T

    # Find triplets where RSM is right but SRF is wrong
    rsm_right_srf_wrong = []
    for i, j, k in val_triplets:
        rsm_sims = np.array([rsm[i, j], rsm[i, k], rsm[j, k]])
        srf_sims = np.array([srf_emb[i] @ srf_emb[j], srf_emb[i] @ srf_emb[k], srf_emb[j] @ srf_emb[k]])

        rsm_pred = np.argmax(rsm_sims)
        srf_pred = np.argmax(np.exp(srf_sims - srf_sims.max()))

        if rsm_pred == 0 and srf_pred != 0:
            rsm_right_srf_wrong.append({
                'i': i, 'j': j, 'k': k,
                'rsm_ij': rsm[i, j], 'rsm_ik': rsm[i, k], 'rsm_jk': rsm[j, k],
                'srf_ij': srf_rsm[i, j], 'srf_ik': srf_rsm[i, k], 'srf_jk': srf_rsm[j, k],
                'n_ij': shown[i, j], 'n_ik': shown[i, k], 'n_jk': shown[j, k],
                'srf_pred': srf_pred,
            })

    log.info(f"\nFound {len(rsm_right_srf_wrong)} triplets where RSM correct but SRF wrong")

    # Analyze reconstruction error for these pairs
    log.info(f"\n=== RSM vs SRF Reconstruction ===")

    # For the correct pair (i,j): how much does SRF underestimate it?
    ij_rsm = [r['rsm_ij'] for r in rsm_right_srf_wrong]
    ij_srf = [r['srf_ij'] for r in rsm_right_srf_wrong]
    ij_recon_error = [r['srf_ij'] - r['rsm_ij'] for r in rsm_right_srf_wrong]

    log.info(f"Correct pair (i,j):")
    log.info(f"  RSM mean: {np.mean(ij_rsm):.4f}")
    log.info(f"  SRF mean: {np.mean(ij_srf):.4f}")
    log.info(f"  Reconstruction error: {np.mean(ij_recon_error):.4f}")

    # For the wrong pairs: how much does SRF overestimate them?
    # SRF predicted (i,k) or (j,k) - check which one was overestimated
    overestimated_pair_errors = []
    for r in rsm_right_srf_wrong:
        if r['srf_pred'] == 1:  # SRF predicted (i,k)
            overestimated_pair_errors.append(r['srf_ik'] - r['rsm_ik'])
        else:  # SRF predicted (j,k)
            overestimated_pair_errors.append(r['srf_jk'] - r['rsm_jk'])

    log.info(f"Overestimated wrong pair:")
    log.info(f"  Reconstruction error: {np.mean(overestimated_pair_errors):.4f}")

    # Check if there's a pattern with item frequency
    log.info(f"\n=== Item Frequency Analysis ===")
    item_freq = np.zeros(n)
    for i, j, k in train_triplets:
        item_freq[i] += 1
        item_freq[j] += 1
        item_freq[k] += 1

    # For problematic triplets, what's the frequency of each item?
    i_freqs = [item_freq[r['i']] for r in rsm_right_srf_wrong]
    j_freqs = [item_freq[r['j']] for r in rsm_right_srf_wrong]
    k_freqs = [item_freq[r['k']] for r in rsm_right_srf_wrong]

    log.info(f"Item i freq: mean={np.mean(i_freqs):.0f}, median={np.median(i_freqs):.0f}")
    log.info(f"Item j freq: mean={np.mean(j_freqs):.0f}, median={np.median(j_freqs):.0f}")
    log.info(f"Item k freq: mean={np.mean(k_freqs):.0f}, median={np.median(k_freqs):.0f}")

    # Compare with all triplets
    all_i_freqs = [item_freq[i] for i, j, k in val_triplets]
    log.info(f"All triplets item freq: mean={np.mean(all_i_freqs):.0f}")

    # Look at the RSM neighborhood
    log.info(f"\n=== Neighborhood Analysis ===")

    # For items in problematic triplets, compute their overall similarity to other items
    def item_avg_similarity(item_idx, rsm):
        row = rsm[item_idx].copy()
        row[item_idx] = np.nan
        return np.nanmean(row)

    # Compare correct pair vs wrong pair average similarities
    correct_pair_avg_sims = []
    wrong_pair_avg_sims = []
    for r in rsm_right_srf_wrong:
        i, j, k = r['i'], r['j'], r['k']
        # Avg similarity of the correct pair's items to everything else
        correct_pair_avg_sims.append((item_avg_similarity(i, rsm) + item_avg_similarity(j, rsm)) / 2)

        # For the pair SRF wrongly preferred
        if r['srf_pred'] == 1:
            wrong_pair_avg_sims.append((item_avg_similarity(i, rsm) + item_avg_similarity(k, rsm)) / 2)
        else:
            wrong_pair_avg_sims.append((item_avg_similarity(j, rsm) + item_avg_similarity(k, rsm)) / 2)

    log.info(f"Correct pair avg item similarity: {np.mean(correct_pair_avg_sims):.4f}")
    log.info(f"Wrong pair avg item similarity: {np.mean(wrong_pair_avg_sims):.4f}")

    # Check embedding norms
    log.info(f"\n=== Embedding Norm Analysis ===")
    norms = np.linalg.norm(srf_emb, axis=1)
    log.info(f"SRF embedding norms: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")

    # For problematic items
    problematic_items = set()
    for r in rsm_right_srf_wrong:
        problematic_items.add(r['i'])
        problematic_items.add(r['j'])
        problematic_items.add(r['k'])
    problematic_norms = [norms[i] for i in problematic_items]
    log.info(f"Problematic items norms: mean={np.mean(problematic_norms):.4f}")

    # What if we normalize embeddings?
    log.info(f"\n=== Effect of Embedding Normalization ===")
    srf_normed = srf_emb / (norms[:, None] + 1e-10)

    correct_normed = 0
    for i, j, k in val_triplets:
        sims = np.array([srf_normed[i] @ srf_normed[j], srf_normed[i] @ srf_normed[k], srf_normed[j] @ srf_normed[k]])
        if np.argmax(np.exp(sims - sims.max())) == 0:
            correct_normed += 1

    log.info(f"SRF normalized accuracy: {correct_normed/len(val_triplets):.4f}")

    # What about SPoSE normalized?
    spose_norms = np.linalg.norm(spose, axis=1)
    spose_normed = spose / (spose_norms[:, None] + 1e-10)

    correct_spose_normed = 0
    for i, j, k in val_triplets:
        sims = np.array([spose_normed[i] @ spose_normed[j], spose_normed[i] @ spose_normed[k], spose_normed[j] @ spose_normed[k]])
        if np.argmax(np.exp(sims - sims.max())) == 0:
            correct_spose_normed += 1

    log.info(f"SPoSE normalized accuracy: {correct_spose_normed/len(val_triplets):.4f}")

    # Analyze the margin in RSM space vs SRF space
    log.info(f"\n=== Margin Compression ===")
    rsm_margins = []
    srf_margins = []
    for r in rsm_right_srf_wrong:
        i, j, k = r['i'], r['j'], r['k']
        rsm_margin = rsm[i, j] - max(rsm[i, k], rsm[j, k])
        srf_margin = (srf_emb[i] @ srf_emb[j]) - max(srf_emb[i] @ srf_emb[k], srf_emb[j] @ srf_emb[k])
        rsm_margins.append(rsm_margin)
        srf_margins.append(srf_margin)

    log.info(f"RSM margin (these triplets): mean={np.mean(rsm_margins):.4f}")
    log.info(f"SRF margin (these triplets): mean={np.mean(srf_margins):.4f}")
    log.info(f"Margin flipped negative: {sum(m < 0 for m in srf_margins)} / {len(srf_margins)}")


if __name__ == "__main__":
    main()
