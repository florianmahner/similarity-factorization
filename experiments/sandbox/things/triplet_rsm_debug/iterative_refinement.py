#!/usr/bin/env python3
"""
Iterative refinement: use embedding to smooth RSM, then re-factorize.
Goal: Let the global structure inform local estimates.
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


def build_rsm(n: int, triplets: np.ndarray):
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


def smooth_rsm_with_embedding(rsm: np.ndarray, embedding: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend original RSM with embedding-reconstructed RSM.
    rsm_new = alpha * rsm_original + (1-alpha) * (W @ W.T normalized)
    """
    rsm_emb = embedding @ embedding.T
    # Normalize to same scale as original
    rsm_emb_norm = (rsm_emb - rsm_emb.min()) / (rsm_emb.max() - rsm_emb.min())

    rsm_blend = alpha * rsm + (1 - alpha) * rsm_emb_norm
    np.fill_diagonal(rsm_blend, 1.0)
    return rsm_blend


def smooth_rsm_with_neighbors(rsm: np.ndarray, k: int = 20) -> np.ndarray:
    """
    For each pair (i,j), smooth based on similarity to k-nearest neighbors.
    If items i and j are both similar to the same other items, they should be similar.
    """
    n = rsm.shape[0]

    # For each item, find k-nearest neighbors
    neighbors = np.zeros((n, k), dtype=int)
    for i in range(n):
        row = rsm[i].copy()
        row[i] = -np.inf
        neighbors[i] = np.argsort(row)[-k:]

    # Smooth: new_rsm[i,j] = avg similarity to shared neighbors
    rsm_smooth = np.zeros_like(rsm)
    for i in range(n):
        for j in range(i + 1, n):
            # Jaccard similarity of neighbor sets
            ni = set(neighbors[i])
            nj = set(neighbors[j])
            shared = ni & nj
            union = ni | nj

            # Average RSM similarity over shared neighbors
            if shared:
                avg_sim = np.mean([rsm[i, s] + rsm[j, s] for s in shared]) / 2
            else:
                avg_sim = rsm[i, j]

            # Blend with original
            rsm_smooth[i, j] = 0.5 * rsm[i, j] + 0.5 * avg_sim
            rsm_smooth[j, i] = rsm_smooth[i, j]

    np.fill_diagonal(rsm_smooth, 1.0)
    return rsm_smooth


def smooth_rsm_confidence_weighted(rsm: np.ndarray, shown: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """
    Weight blend by confidence: trust counts more when many observations, embedding when few.
    """
    rsm_emb = embedding @ embedding.T
    rsm_emb_norm = (rsm_emb - rsm_emb.min()) / (rsm_emb.max() - rsm_emb.min())

    # Confidence based on observation count
    # More observations = trust counts more
    max_obs = shown.max()
    confidence = shown / (shown + 10)  # Asymptotes to 1 as obs increases

    rsm_blend = confidence * rsm + (1 - confidence) * rsm_emb_norm
    np.fill_diagonal(rsm_blend, 1.0)
    return rsm_blend


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

    rsm, shown = build_rsm(n, train_triplets)

    from pysrf import SRF

    # Standard
    log.info(f"\n=== Standard SRF ===")
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm)
    log.info(f"Standard: {compute_triplet_accuracy(w, val_triplets):.4f}")

    # Iterative refinement
    log.info(f"\n=== Iterative Refinement ===")
    rsm_current = rsm.copy()
    for iteration in range(5):
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm_current)
        acc = compute_triplet_accuracy(w, val_triplets)
        log.info(f"Iteration {iteration}: {acc:.4f}")

        # Smooth RSM with embedding
        rsm_current = smooth_rsm_with_embedding(rsm, w, alpha=0.7)

    # Confidence-weighted blending
    log.info(f"\n=== Confidence-Weighted Blending ===")
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w_init = model.fit_transform(rsm)

    rsm_conf = smooth_rsm_confidence_weighted(rsm, shown, w_init)
    model2 = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w_conf = model2.fit_transform(rsm_conf)
    log.info(f"Confidence-weighted: {compute_triplet_accuracy(w_conf, val_triplets):.4f}")

    # Try different alpha values for simple blending
    log.info(f"\n=== Embedding Blend (varying alpha) ===")
    for alpha in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]:
        rsm_blend = smooth_rsm_with_embedding(rsm, w_init, alpha=alpha)
        model_b = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w_blend = model_b.fit_transform(rsm_blend)
        acc = compute_triplet_accuracy(w_blend, val_triplets)
        log.info(f"alpha={alpha}: {acc:.4f}")

    # Neighbor-based smoothing
    log.info(f"\n=== Neighbor-based Smoothing ===")
    for k in [10, 20, 50]:
        log.info(f"Computing k={k} neighbor smoothing...")
        rsm_neighbor = smooth_rsm_with_neighbors(rsm, k=k)
        model_n = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w_neighbor = model_n.fit_transform(rsm_neighbor)
        acc = compute_triplet_accuracy(w_neighbor, val_triplets)
        log.info(f"k={k}: {acc:.4f}")


if __name__ == "__main__":
    main()
