#!/usr/bin/env python3
"""
Diagnose why SRF embedding performs worse than raw RSM on triplets.

Hypothesis: The RSM computed from triplets is NOT a proper similarity matrix
that can be factorized as W @ W.T
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.linalg import eigh

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_count_rsm(n: int, triplets: np.ndarray) -> np.ndarray:
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


def rsm_triplet_accuracy(s: np.ndarray, triplets: np.ndarray) -> float:
    correct = 0
    for i, j, k in triplets:
        if np.argmax([s[i,j], s[i,k], s[j,k]]) == 0:
            correct += 1
    return correct / len(triplets)


def embedding_triplet_accuracy(w: np.ndarray, triplets: np.ndarray, use_softmax: bool = True) -> float:
    correct = 0
    for i, j, k in triplets:
        sims = np.array([w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]])
        if use_softmax:
            probs = softmax(sims)
            pred = np.argmax(probs)
        else:
            pred = np.argmax(sims)
        if pred == 0:
            correct += 1
    return correct / len(triplets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things/triplets_47")
    train_triplets = np.loadtxt(data_dir / "trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "validationset.txt").astype(int)
    n = 1854

    if args.subsample < 1.0:
        n_samples = int(len(train_triplets) * args.subsample)
        idx = np.random.choice(len(train_triplets), size=n_samples, replace=False)
        train_triplets = train_triplets[idx]
    log.info(f"Using {len(train_triplets)} train triplets")

    # Compute RSM
    rsm = compute_count_rsm(n, train_triplets)
    log.info(f"\nRSM stats:")
    log.info(f"  Shape: {rsm.shape}")
    log.info(f"  Range: [{rsm.min():.3f}, {rsm.max():.3f}]")
    log.info(f"  Mean: {rsm.mean():.3f}")
    log.info(f"  Symmetric: {np.allclose(rsm, rsm.T)}")

    # Check eigenvalues - is RSM positive semi-definite?
    log.info(f"\nEigenvalue analysis:")
    eigenvalues = eigh(rsm, eigvals_only=True)
    n_negative = np.sum(eigenvalues < -1e-10)
    log.info(f"  Min eigenvalue: {eigenvalues.min():.4f}")
    log.info(f"  Max eigenvalue: {eigenvalues.max():.4f}")
    log.info(f"  Negative eigenvalues: {n_negative} / {len(eigenvalues)}")
    log.info(f"  Sum of negative eigenvalues: {eigenvalues[eigenvalues < 0].sum():.4f}")

    # The RSM is NOT positive semi-definite!
    # This means it cannot be perfectly represented as W @ W.T

    log.info(f"\nRSM triplet accuracy:")
    log.info(f"  Train: {rsm_triplet_accuracy(rsm, train_triplets):.4f}")
    log.info(f"  Val: {rsm_triplet_accuracy(rsm, val_triplets):.4f}")

    # Try different factorization approaches
    log.info(f"\n=== Factorization experiments ===")

    # 1. Standard eigendecomposition (allow negative eigenvalues)
    log.info(f"\n1. Eigendecomposition (full spectrum):")
    eigenvalues, eigenvectors = eigh(rsm)
    for rank in [10, 30, 66, 100, 200]:
        # Take top-k eigenvalues (by magnitude)
        idx = np.argsort(np.abs(eigenvalues))[::-1][:rank]
        w = eigenvectors[:, idx] * np.sqrt(np.abs(eigenvalues[idx]))
        acc_train = embedding_triplet_accuracy(w, train_triplets, use_softmax=False)
        acc_val = embedding_triplet_accuracy(w, val_triplets, use_softmax=False)
        log.info(f"  Rank {rank}: train={acc_train:.4f}, val={acc_val:.4f}")

    # 2. Only positive eigenvalues
    log.info(f"\n2. Eigendecomposition (positive only):")
    pos_mask = eigenvalues > 1e-10
    for rank in [10, 30, 66, 100, 200]:
        pos_eigenvalues = eigenvalues[pos_mask]
        pos_eigenvectors = eigenvectors[:, pos_mask]
        # Take top-k positive eigenvalues
        idx = np.argsort(pos_eigenvalues)[::-1][:min(rank, len(pos_eigenvalues))]
        w = pos_eigenvectors[:, idx] * np.sqrt(pos_eigenvalues[idx])
        acc_train = embedding_triplet_accuracy(w, train_triplets, use_softmax=False)
        acc_val = embedding_triplet_accuracy(w, val_triplets, use_softmax=False)
        log.info(f"  Rank {rank}: train={acc_train:.4f}, val={acc_val:.4f}")

    # 3. SRF (for comparison)
    log.info(f"\n3. SRF factorization:")
    from pysrf import SRF
    for rank in [10, 30, 66, 100]:
        model = SRF(rank=rank, random_state=args.seed)
        w = model.fit_transform(rsm)
        acc_train = embedding_triplet_accuracy(w, train_triplets, use_softmax=False)
        acc_val = embedding_triplet_accuracy(w, val_triplets, use_softmax=False)
        log.info(f"  Rank {rank}: train={acc_train:.4f}, val={acc_val:.4f}")

    # 4. What if we shift RSM to be PSD?
    log.info(f"\n4. Shifted RSM (make PSD):")
    min_eig = eigenvalues.min()
    rsm_shifted = rsm - min_eig * np.eye(n) + 0.01 * np.eye(n)
    eigenvalues_shifted, eigenvectors_shifted = eigh(rsm_shifted)
    log.info(f"  Min eigenvalue after shift: {eigenvalues_shifted.min():.4f}")

    for rank in [10, 30, 66, 100]:
        idx = np.argsort(eigenvalues_shifted)[::-1][:rank]
        w = eigenvectors_shifted[:, idx] * np.sqrt(eigenvalues_shifted[idx])
        acc_train = embedding_triplet_accuracy(w, train_triplets, use_softmax=False)
        acc_val = embedding_triplet_accuracy(w, val_triplets, use_softmax=False)
        log.info(f"  Rank {rank}: train={acc_train:.4f}, val={acc_val:.4f}")

    # 5. What about the reconstruction error?
    log.info(f"\n5. Reconstruction analysis:")
    for rank in [66, 100, 200, 500]:
        idx = np.argsort(eigenvalues)[::-1][:rank]
        w = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))
        rsm_reconstructed = w @ w.T
        mse = np.mean((rsm - rsm_reconstructed) ** 2)
        corr = np.corrcoef(rsm.flatten(), rsm_reconstructed.flatten())[0, 1]
        log.info(f"  Rank {rank}: MSE={mse:.6f}, corr={corr:.4f}")

    # 6. Key insight: compare RSM similarity ordering vs embedding similarity ordering
    log.info(f"\n6. Similarity preservation analysis:")
    # For a sample of triplets, check if the ordering is preserved
    sample_triplets = val_triplets[:1000]

    eigenvalues, eigenvectors = eigh(rsm)
    for rank in [66, 200, 500, 1000]:
        idx = np.argsort(eigenvalues)[::-1][:rank]
        w = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

        ordering_preserved = 0
        for i, j, k in sample_triplets:
            rsm_order = np.argsort([rsm[i,j], rsm[i,k], rsm[j,k]])[::-1]
            emb_sims = [w[i] @ w[j], w[i] @ w[k], w[j] @ w[k]]
            emb_order = np.argsort(emb_sims)[::-1]
            if np.array_equal(rsm_order, emb_order):
                ordering_preserved += 1

        log.info(f"  Rank {rank}: {ordering_preserved/len(sample_triplets)*100:.1f}% orderings preserved")


if __name__ == "__main__":
    main()
