#!/usr/bin/env python3
"""Average trials, filter channels, run SRF, and visualize embeddings."""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from pysrf import SRF
from src.utils import get_output_dir
from src.colors import TEAL, CYAN, GRAY
from src.utils.figure_theme import create_figure, despine

OUTPUT_DIR = get_output_dir()
RELIAB_THRESHOLD = 0.3
K = 10


def load_dataset(path: Path, roi: str = "it"):
    with h5py.File(path / "THINGS_normMUA_raw.mat", "r") as f:
        data = np.array(f[f"data_{roi}"])
    zi_list = pd.read_csv(path / "zi_list.csv", header=None)[0].values
    valid = ~np.isnan(data).any(axis=1)
    if valid.sum() < len(valid):
        print(f"    Removing {(~valid).sum()} NaN trials")
    return data[valid], zi_list[valid]


def zscore_channels(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std


def average_across_trials(data: np.ndarray, zi_list: np.ndarray, classes: np.ndarray):
    """Average trials per stimulus to get (n_objects, n_channels)."""
    n_objects = len(classes)
    n_channels = data.shape[1]
    averaged = np.zeros((n_objects, n_channels))

    for i, c in enumerate(classes):
        trials = data[zi_list == c]
        averaged[i] = trials.mean(axis=0)

    return averaged


def compute_rsm(X: np.ndarray) -> np.ndarray:
    """Compute Gaussian kernel RSM with median heuristic."""
    from src.tools.metrics import gaussian_kernel_similarity
    return gaussian_kernel_similarity(X, X)


def plot_data_quality(X_n, X_nnew, X_combined, reliab, out_dir):
    """Plot data quality diagnostics."""

    # 1. Channel response distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (X, title) in zip(axes, [(X_n, "MonkeyN"), (X_nnew, "MonkeyNnew"), (X_combined, "Combined")]):
        mean_resp = X.mean(axis=0)
        ax.hist(mean_resp, bins=40, color=TEAL, alpha=0.7, edgecolor='white')
        ax.set_xlabel("Mean response")
        ax.set_ylabel("Channels")
        ax.set_title(title)
        despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "channel_distributions.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. RSM comparison (before filtering)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (X, title) in zip(axes, [(X_n, "MonkeyN"), (X_nnew, "MonkeyNnew"), (X_combined, "Combined")]):
        rsm = np.corrcoef(X)
        im = ax.imshow(rsm, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title(f"{title} RSM")
        ax.set_xlabel("Objects")
        ax.set_ylabel("Objects")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "rsm_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. RSM correlation between recordings
    rsm_n = np.corrcoef(X_n)
    rsm_nnew = np.corrcoef(X_nnew)
    rsm_combined = np.corrcoef(X_combined)

    triu = np.triu_indices(rsm_n.shape[0], k=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    r = pearsonr(rsm_n[triu], rsm_nnew[triu])[0]
    ax.scatter(rsm_n[triu], rsm_nnew[triu], alpha=0.1, s=1, c=GRAY)
    ax.plot([-1, 1], [-1, 1], 'r--', lw=1)
    ax.set_xlabel("MonkeyN RSM")
    ax.set_ylabel("MonkeyNnew RSM")
    ax.set_title(f"RSM correlation: r={r:.3f}")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    despine(ax)

    ax = axes[1]
    r = pearsonr(rsm_n[triu], rsm_combined[triu])[0]
    ax.scatter(rsm_n[triu], rsm_combined[triu], alpha=0.1, s=1, c=GRAY)
    ax.plot([-1, 1], [-1, 1], 'r--', lw=1)
    ax.set_xlabel("MonkeyN RSM")
    ax.set_ylabel("Combined RSM")
    ax.set_title(f"r={r:.3f}")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    despine(ax)

    ax = axes[2]
    r = pearsonr(rsm_nnew[triu], rsm_combined[triu])[0]
    ax.scatter(rsm_nnew[triu], rsm_combined[triu], alpha=0.1, s=1, c=GRAY)
    ax.plot([-1, 1], [-1, 1], 'r--', lw=1)
    ax.set_xlabel("MonkeyNnew RSM")
    ax.set_ylabel("Combined RSM")
    ax.set_title(f"r={r:.3f}")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    despine(ax)

    plt.tight_layout()
    fig.savefig(out_dir / "rsm_correlations.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 4. Reliability vs mean response
    fig, ax = plt.subplots(figsize=(5, 4))
    mean_resp = X_combined.mean(axis=0)
    ax.scatter(mean_resp, reliab, alpha=0.6, c=TEAL, s=20)
    ax.axhline(RELIAB_THRESHOLD, color='r', ls='--', label=f'threshold={RELIAB_THRESHOLD}')
    ax.set_xlabel("Mean channel response")
    ax.set_ylabel("Split-half reliability")
    ax.legend(frameon=False)
    despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "reliability_vs_response.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved diagnostic plots to {out_dir}")


def plot_embedding(W: np.ndarray, out_dir: Path):
    """Plot SRF embedding."""
    n_objects, k = W.shape

    # 1. Embedding heatmap (top objects per dimension)
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each dimension, get top 20 objects
    n_top = 20
    top_indices = np.zeros((k, n_top), dtype=int)
    for d in range(k):
        top_indices[d] = np.argsort(W[:, d])[-n_top:][::-1]

    # Create matrix for visualization
    W_top = np.zeros((n_top, k))
    for d in range(k):
        W_top[:, d] = W[top_indices[d], d]

    im = ax.imshow(W_top.T, aspect='auto', cmap='viridis')
    ax.set_xlabel(f"Top {n_top} objects per dimension")
    ax.set_ylabel("Dimension")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"D{i+1}" for i in range(k)])
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_title(f"SRF Embedding (k={k})")

    plt.tight_layout()
    fig.savefig(out_dir / "embedding_heatmap.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Dimension weight distributions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for d in range(k):
        ax = axes[d]
        weights = W[:, d]
        ax.hist(weights, bins=30, color=TEAL, alpha=0.7, edgecolor='white')
        ax.set_title(f"D{d+1}")
        ax.set_xlabel("Weight")
        if d % 5 == 0:
            ax.set_ylabel("Objects")
        despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "dimension_distributions.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. Reconstruction quality
    S_recon = W @ W.T

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.imshow(S_recon, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_title("Reconstructed RSM")
    ax.set_xlabel("Objects")
    ax.set_ylabel("Objects")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    ax.hist(S_recon.flatten(), bins=50, color=CYAN, alpha=0.7, edgecolor='white')
    ax.set_xlabel("Reconstructed similarity")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction distribution")
    despine(ax)

    plt.tight_layout()
    fig.savefig(out_dir / "reconstruction.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved embedding plots to {out_dir}")


def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "things-monkey" / "THINGS-2k"
    reliab_path = Path(__file__).parent / "outputs/260116_112528/reliability.npz"

    print("Loading data...")
    data_n, zi_n = load_dataset(data_dir / "monkeyN")
    data_nnew, zi_nnew = load_dataset(data_dir / "monkeyNnew")

    # Find common classes and filter FIRST
    classes_n = np.unique(zi_n)
    classes_nnew = np.unique(zi_nnew)
    common = np.array(sorted(set(classes_n) & set(classes_nnew)))
    print(f"  Common stimuli: {len(common)}")

    # Filter to common stimuli before z-scoring
    mask_n = np.isin(zi_n, common)
    mask_nnew = np.isin(zi_nnew, common)
    data_n, zi_n = data_n[mask_n], zi_n[mask_n]
    data_nnew, zi_nnew = data_nnew[mask_nnew], zi_nnew[mask_nnew]
    print(f"  Filtered: MonkeyN {data_n.shape[0]} trials, MonkeyNnew {data_nnew.shape[0]} trials")

    print("Z-scoring and averaging across trials...")
    # Z-score each recording separately AFTER filtering
    data_n_z = zscore_channels(data_n)
    data_nnew_z = zscore_channels(data_nnew)

    X_n = average_across_trials(data_n_z, zi_n, common)
    X_nnew = average_across_trials(data_nnew_z, zi_nnew, common)

    # Combine z-scored data then average
    data_combined = np.vstack([data_n_z, data_nnew_z])
    zi_combined = np.concatenate([zi_n, zi_nnew])
    X_combined = average_across_trials(data_combined, zi_combined, common)

    print(f"  X_n: {X_n.shape}, X_nnew: {X_nnew.shape}, X_combined: {X_combined.shape}")

    # Load reliability and filter channels
    print("Loading reliability...")
    reliab_data = np.load(reliab_path)
    reliab = reliab_data["reliab_combined"]

    good_channels = reliab > RELIAB_THRESHOLD
    n_good = good_channels.sum()
    print(f"  Channels with reliability > {RELIAB_THRESHOLD}: {n_good}/{len(reliab)}")

    # Plot data quality diagnostics
    print("Plotting data quality...")
    plot_data_quality(X_n, X_nnew, X_combined, reliab, OUTPUT_DIR)

    # Filter channels
    X_filtered = X_combined[:, good_channels]
    print(f"  Filtered data: {X_filtered.shape}")

    # Compute RSM (Gaussian kernel with median heuristic)
    print("Computing RSM (Gaussian kernel)...")
    rsm = compute_rsm(X_filtered)
    print(f"  RSM shape: {rsm.shape}")

    # Run SRF
    print(f"Running SRF (k={K})...")
    model = SRF(rank=K, max_outer=200, verbose=1, random_state=42)
    model.fit(rsm)
    W = model.components_
    print(f"  Embedding shape: {W.shape}")
    print(f"  Iterations: {model.n_iter_}")

    # Plot embedding
    print("Plotting embedding...")
    plot_embedding(W, OUTPUT_DIR)

    # Save results
    np.savez(OUTPUT_DIR / "srf_results.npz",
             embedding=W, rsm=rsm, classes=common,
             good_channels=good_channels, reliab=reliab)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
