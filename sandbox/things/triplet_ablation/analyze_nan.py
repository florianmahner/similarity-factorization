#!/usr/bin/env python3
"""
Deep analysis of NaN handling in RSM construction.
Test different strategies for dealing with unobserved pairs.
"""
import logging
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def compute_triplet_accuracy(embedding: np.ndarray, triplets: np.ndarray) -> float:
    acc = 0
    for i, j, k in triplets:
        sims = np.array([embedding[i] @ embedding[j], embedding[i] @ embedding[k], embedding[j] @ embedding[k]])
        probas = np.exp(sims - sims.max())
        probas /= probas.sum()
        acc += np.argmax(probas) == 0
    return acc / len(triplets)


def build_rsm_raw(n: int, triplets: np.ndarray):
    """Build RSM with NaN for unobserved pairs."""
    counts = np.zeros((n, n))
    shown = np.zeros((n, n))
    for i, j, k in triplets:
        for a, b in [(i, j), (i, k), (j, k)]:
            shown[a, b] += 1
            shown[b, a] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    rsm = np.divide(counts, shown, out=np.nan * np.ones_like(counts), where=shown != 0)
    np.fill_diagonal(rsm, 1.0)
    return rsm, counts, shown


def fill_nan_constant(rsm: np.ndarray, value: float) -> np.ndarray:
    """Fill NaN with constant value."""
    rsm_filled = rsm.copy()
    rsm_filled[np.isnan(rsm_filled)] = value
    return rsm_filled


def fill_nan_row_mean(rsm: np.ndarray) -> np.ndarray:
    """Fill NaN with average of both row means (to maintain symmetry)."""
    rsm_filled = rsm.copy()
    n = rsm.shape[0]
    row_means = np.nanmean(rsm, axis=1)

    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(rsm[i, j]):
                # Use average of both items' mean similarities
                fill_val = (row_means[i] + row_means[j]) / 2
                rsm_filled[i, j] = fill_val
                rsm_filled[j, i] = fill_val
    return rsm_filled


def fill_nan_global_mean(rsm: np.ndarray) -> np.ndarray:
    """Fill NaN with global mean."""
    rsm_filled = rsm.copy()
    global_mean = np.nanmean(rsm)
    rsm_filled[np.isnan(rsm_filled)] = global_mean
    return rsm_filled


def fill_nan_neighbor_avg(rsm: np.ndarray, k: int = 10) -> np.ndarray:
    """Fill NaN by averaging k-nearest neighbors' values (symmetric)."""
    rsm_filled = rsm.copy()
    n = rsm.shape[0]

    # Precompute neighbors for each item
    neighbors = np.zeros((n, k), dtype=int)
    for i in range(n):
        row_i = rsm[i].copy()
        row_i[np.isnan(row_i)] = -np.inf
        row_i[i] = -np.inf
        neighbors[i] = np.argsort(row_i)[-k:]

    global_mean = np.nanmean(rsm)

    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(rsm[i, j]):
                # Estimate i-j similarity from neighbors
                estimates = []
                for ni in neighbors[i]:
                    if not np.isnan(rsm[ni, j]):
                        estimates.append(rsm[ni, j])
                for nj in neighbors[j]:
                    if not np.isnan(rsm[i, nj]):
                        estimates.append(rsm[i, nj])

                if estimates:
                    fill_val = np.mean(estimates)
                else:
                    fill_val = global_mean

                rsm_filled[i, j] = fill_val
                rsm_filled[j, i] = fill_val

    return rsm_filled


def fill_nan_embedding_predict(rsm: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """Fill NaN using predictions from fitted embedding (symmetric by construction)."""
    rsm_filled = rsm.copy()
    rsm_pred = embedding @ embedding.T  # Already symmetric

    # Normalize to same scale as original RSM
    valid_mask = ~np.isnan(rsm)
    if valid_mask.any():
        rsm_min, rsm_max = rsm[valid_mask].min(), rsm[valid_mask].max()
        rsm_pred_norm = (rsm_pred - rsm_pred.min()) / (rsm_pred.max() - rsm_pred.min())
        rsm_pred_norm = rsm_pred_norm * (rsm_max - rsm_min) + rsm_min
    else:
        rsm_pred_norm = rsm_pred

    nan_mask = np.isnan(rsm)
    rsm_filled[nan_mask] = rsm_pred_norm[nan_mask]
    return rsm_filled


def main():
    # Load 1.47M data
    log.info("Loading 1.47M dataset...")
    data_dir = Path("/LOCAL/fmahner/similarity-factorization/data/things")
    train_triplets = np.loadtxt(data_dir / "triplets_147/trainset.txt").astype(int)
    val_triplets = np.loadtxt(data_dir / "triplets_147/validationset.txt").astype(int)
    spose = np.maximum(np.loadtxt(data_dir / "spose_embedding_66d.txt"), 0)
    n = 1854

    # Build raw RSM
    rsm, counts, shown = build_rsm_raw(n, train_triplets)
    n_nan = np.isnan(rsm).sum()
    log.info(f"NaN entries: {n_nan:,} ({100*n_nan/(n*n):.2f}%)")

    from pysrf import SRF

    # Test different NaN fill strategies
    log.info("\n=== NaN Fill Strategies ===")

    strategies = {
        "constant_0.0": lambda r: fill_nan_constant(r, 0.0),
        "constant_0.25": lambda r: fill_nan_constant(r, 0.25),
        "constant_0.33": lambda r: fill_nan_constant(r, 0.33),
        "constant_0.5": lambda r: fill_nan_constant(r, 0.5),
        "constant_0.75": lambda r: fill_nan_constant(r, 0.75),
        "row_mean": fill_nan_row_mean,
        "global_mean": fill_nan_global_mean,
    }

    results = []

    for name, fill_fn in strategies.items():
        log.info(f"\n{name}:")
        rsm_filled = fill_fn(rsm)

        # Check if there are still NaNs
        remaining_nan = np.isnan(rsm_filled).sum()
        if remaining_nan > 0:
            log.info(f"  Warning: {remaining_nan} NaN remaining, filling with 0.5")
            rsm_filled[np.isnan(rsm_filled)] = 0.5

        model = SRF(rank=66, random_state=42, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        emb = model.fit_transform(rsm_filled)
        acc = compute_triplet_accuracy(emb, val_triplets)
        log.info(f"  Accuracy: {acc:.4f}")

        results.append({"strategy": name, "accuracy": acc})

    # Iterative refinement: use embedding to fill NaN, then refit
    log.info("\n=== Iterative Embedding-Based Fill ===")
    rsm_current = fill_nan_constant(rsm, 0.5)

    for iteration in range(3):
        model = SRF(rank=66, random_state=42, max_outer=2000, max_inner=50, tol=1e-4, verbose=0)
        emb = model.fit_transform(rsm_current)
        acc = compute_triplet_accuracy(emb, val_triplets)
        log.info(f"Iteration {iteration}: {acc:.4f}")

        # Update NaN values with embedding predictions
        rsm_current = fill_nan_embedding_predict(rsm, emb)
        results.append({"strategy": f"iterative_{iteration}", "accuracy": acc})

    # Compare with SPoSE
    spose_acc = compute_triplet_accuracy(spose, val_triplets)
    log.info(f"\nSPoSE baseline: {spose_acc:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "nan_strategies.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(results_df["strategy"], results_df["accuracy"], alpha=0.7, edgecolor='black')
    ax.axhline(y=spose_acc, color='red', linestyle='--', label=f"SPoSE: {spose_acc:.4f}")
    ax.axhline(y=0.6667, color='gray', linestyle='--', label="Noise ceiling")
    ax.set_xlabel("NaN Fill Strategy")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Effect of NaN Fill Strategy on SRF Performance (1.47M dataset)")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nan_strategies.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"\nSaved nan_strategies.pdf")

    # Analysis: Where are the NaN pairs?
    log.info("\n=== NaN Location Analysis ===")

    # For validation triplets, how many contain NaN pairs?
    nan_mask = np.isnan(rsm)
    n_val_with_nan = 0
    n_val_all_nan = 0

    for i, j, k in val_triplets:
        has_nan = nan_mask[i, j] or nan_mask[i, k] or nan_mask[j, k]
        all_nan = nan_mask[i, j] and nan_mask[i, k] and nan_mask[j, k]
        if has_nan:
            n_val_with_nan += 1
        if all_nan:
            n_val_all_nan += 1

    log.info(f"Validation triplets with at least one NaN pair: {n_val_with_nan:,} ({100*n_val_with_nan/len(val_triplets):.2f}%)")
    log.info(f"Validation triplets with all NaN pairs: {n_val_all_nan:,}")

    # Item-level NaN analysis
    nan_per_item = nan_mask.sum(axis=1) - 1  # Exclude diagonal
    log.info(f"\nNaN pairs per item:")
    log.info(f"  Mean: {nan_per_item.mean():.1f}")
    log.info(f"  Max: {nan_per_item.max()}")
    log.info(f"  Items with >100 NaN: {(nan_per_item > 100).sum()}")
    log.info(f"  Items with 0 NaN: {(nan_per_item == 0).sum()}")

    # Plot NaN per item distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nan_per_item, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("NaN pairs per item")
    ax.set_ylabel("Number of items")
    ax.set_title("Distribution of NaN pairs per item")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nan_per_item.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved nan_per_item.pdf")


if __name__ == "__main__":
    main()
