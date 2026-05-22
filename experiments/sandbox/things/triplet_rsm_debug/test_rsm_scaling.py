#!/usr/bin/env python3
"""
Test RSM transformations that amplify differences before factorization.
Goal: Counteract the compression-toward-mean effect.
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


def transform_power(rsm: np.ndarray, power: float) -> np.ndarray:
    """Apply power transformation to amplify differences."""
    # Center around 0.5, apply power, rescale
    centered = rsm - 0.5
    sign = np.sign(centered)
    transformed = sign * np.abs(centered) ** power
    # Rescale back
    result = transformed / (np.abs(transformed).max() + 1e-10) * 0.5 + 0.5
    np.fill_diagonal(result, 1.0)
    return result


def transform_sigmoid(rsm: np.ndarray, steepness: float) -> np.ndarray:
    """Apply sigmoid to amplify differences around 0.5."""
    # Sigmoid centered at 0.5
    x = (rsm - 0.5) * steepness
    result = 1 / (1 + np.exp(-x))
    np.fill_diagonal(result, 1.0)
    return result


def transform_quantile(rsm: np.ndarray) -> np.ndarray:
    """Transform to uniform distribution (rank-based)."""
    n = rsm.shape[0]
    result = np.zeros_like(rsm)
    # For each row, transform to ranks
    for i in range(n):
        row = rsm[i].copy()
        row[i] = np.nan  # Exclude diagonal
        ranks = np.argsort(np.argsort(row))
        result[i] = ranks / (n - 2)  # Normalize to [0, 1]
    result = (result + result.T) / 2
    np.fill_diagonal(result, 1.0)
    return result


def transform_log_ratio(rsm: np.ndarray) -> np.ndarray:
    """Log-odds transformation."""
    eps = 0.01
    rsm_clipped = np.clip(rsm, eps, 1 - eps)
    log_odds = np.log(rsm_clipped / (1 - rsm_clipped))
    # Normalize to [0, 1]
    result = (log_odds - log_odds.min()) / (log_odds.max() - log_odds.min())
    np.fill_diagonal(result, 1.0)
    return result


def transform_contrast_stretch(rsm: np.ndarray, low_pct: float = 5, high_pct: float = 95) -> np.ndarray:
    """Contrast stretching - clip percentiles and stretch."""
    off_diag = rsm[np.triu_indices_from(rsm, k=1)]
    low_val = np.percentile(off_diag, low_pct)
    high_val = np.percentile(off_diag, high_pct)

    result = np.clip(rsm, low_val, high_val)
    result = (result - low_val) / (high_val - low_val)
    np.fill_diagonal(result, 1.0)
    return result


def transform_exponential(rsm: np.ndarray, scale: float) -> np.ndarray:
    """Exponential transformation to spread values."""
    # exp(scale * (rsm - 0.5)) centered transformation
    result = np.exp(scale * (rsm - 0.5))
    # Normalize
    result = (result - result.min()) / (result.max() - result.min())
    np.fill_diagonal(result, 1.0)
    return result


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

    # Build RSM
    rsm = build_rsm(n, train_triplets)

    from pysrf import SRF

    # Standard RSM
    log.info(f"\n=== Standard RSM ===")
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm)
    log.info(f"Standard: {compute_triplet_accuracy(w, val_triplets):.4f}")

    # Power transformations
    log.info(f"\n=== Power Transformations ===")
    for power in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
        rsm_t = transform_power(rsm, power)
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm_t)
        acc = compute_triplet_accuracy(w, val_triplets)
        log.info(f"power={power}: {acc:.4f}")

    # Sigmoid transformations
    log.info(f"\n=== Sigmoid Transformations ===")
    for steepness in [2, 4, 6, 8, 10]:
        rsm_t = transform_sigmoid(rsm, steepness)
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm_t)
        acc = compute_triplet_accuracy(w, val_triplets)
        log.info(f"sigmoid steepness={steepness}: {acc:.4f}")

    # Log-odds
    log.info(f"\n=== Log-odds Transformation ===")
    rsm_t = transform_log_ratio(rsm)
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm_t)
    log.info(f"log-odds: {compute_triplet_accuracy(w, val_triplets):.4f}")

    # Quantile (rank-based)
    log.info(f"\n=== Quantile Transformation ===")
    rsm_t = transform_quantile(rsm)
    model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
    w = model.fit_transform(rsm_t)
    log.info(f"quantile: {compute_triplet_accuracy(w, val_triplets):.4f}")

    # Contrast stretch
    log.info(f"\n=== Contrast Stretch ===")
    for low, high in [(1, 99), (5, 95), (10, 90)]:
        rsm_t = transform_contrast_stretch(rsm, low, high)
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm_t)
        acc = compute_triplet_accuracy(w, val_triplets)
        log.info(f"contrast [{low}, {high}]: {acc:.4f}")

    # Exponential
    log.info(f"\n=== Exponential ===")
    for scale in [1, 2, 3, 5]:
        rsm_t = transform_exponential(rsm, scale)
        model = SRF(rank=66, random_state=0, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        w = model.fit_transform(rsm_t)
        acc = compute_triplet_accuracy(w, val_triplets)
        log.info(f"exp scale={scale}: {acc:.4f}")


if __name__ == "__main__":
    main()
