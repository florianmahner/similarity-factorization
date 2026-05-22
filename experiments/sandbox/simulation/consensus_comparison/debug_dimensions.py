"""
Debug mur92 dimensions - understand what each dimension captures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.pipeline import Pipeline

from pysrf import SRF, EnsembleEmbedding, AlignedConsensus
from src.datasets.loaders import load_mur92
from src.colors import CYCLE, GRAY, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "debug_dimensions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    image_paths = dataset.metadata.get("images", [])
    n_samples = rsm.shape[0]
    rank = 3
    n_runs = 50

    print(f"Fitting consensus embedding (rank={rank}, n_runs={n_runs})...")
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=n_runs, n_jobs=-1, random_state=42)),
        ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
    ])
    embedding = pipeline.fit_transform(rsm)

    # Load images
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((80, 80))
            images.append(np.array(img))
        except:
            images.append(np.zeros((80, 80, 3), dtype=np.uint8))

    # =========================================================================
    # Analysis 1: Print embedding statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"\nPer-dimension statistics:")
    for j in range(rank):
        vals = embedding[:, j]
        print(f"  D{j+1}: min={vals.min():.3f}, max={vals.max():.3f}, "
              f"mean={vals.mean():.3f}, std={vals.std():.3f}")

    print(f"\nCorrelation between dimensions:")
    corr = np.corrcoef(embedding.T)
    for i in range(rank):
        for j in range(i+1, rank):
            print(f"  D{i+1}-D{j+1}: {corr[i,j]:.3f}")

    # =========================================================================
    # Analysis 2: Dominant dimension distribution
    # =========================================================================
    dominant = np.argmax(embedding, axis=1)
    print(f"\nDominant dimension distribution:")
    for j in range(rank):
        count = (dominant == j).sum()
        print(f"  D{j+1}: {count} items ({100*count/n_samples:.1f}%)")

    # =========================================================================
    # Analysis 3: Items with mixed loadings (no clear dominant dimension)
    # =========================================================================
    # Compute "purity" - how much the max loading dominates
    max_loading = embedding.max(axis=1)
    sum_loading = embedding.sum(axis=1)
    purity = max_loading / sum_loading  # 1.0 = pure, 0.33 = uniform

    print(f"\nPurity distribution (max/sum):")
    print(f"  min={purity.min():.3f}, max={purity.max():.3f}, mean={purity.mean():.3f}")

    # Most "impure" items (mixed membership)
    mixed_idx = np.argsort(purity)[:10]
    print(f"\nMost mixed items (low purity):")
    for idx in mixed_idx:
        print(f"  {Path(image_paths[idx]).stem}: {embedding[idx]} (purity={purity[idx]:.3f})")

    # =========================================================================
    # Plot 1: All items sorted by each dimension (full view)
    # =========================================================================
    fig, axes = plt.subplots(rank, 1, figsize=(16, 4*rank))

    for j, ax in enumerate(axes):
        sort_idx = np.argsort(embedding[:, j])[::-1]  # High to low

        for pos, idx in enumerate(sort_idx):
            img = images[idx]
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, (pos, 0.5), frameon=True, pad=0.05,
                               bboxprops=dict(edgecolor=CYCLE[dominant[idx] % len(CYCLE)], linewidth=1.5))
            ax.add_artist(ab)

        ax.set_xlim(-0.5, n_samples - 0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(f"Dimension {j+1} (sorted high→low)", fontsize=12, color=CYCLE[j % len(CYCLE)], fontweight="bold")
        ax.set_xlabel("Rank position")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "plot1_all_items_sorted.pdf", bbox_inches="tight", dpi=100)
    plt.close(fig)
    print("\nSaved plot1_all_items_sorted.pdf")

    # =========================================================================
    # Plot 2: Loading profiles for each item
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by dominant dimension, then by max loading
    sort_key = dominant * 1000 + (1 - purity)  # Group by dominant, then by purity
    sort_idx = np.argsort(sort_key)

    emb_sorted = embedding[sort_idx]

    # Stacked bar chart
    bottom = np.zeros(n_samples)
    for j in range(rank):
        ax.bar(range(n_samples), emb_sorted[:, j], bottom=bottom,
               color=CYCLE[j % len(CYCLE)], label=f"D{j+1}", width=1.0)
        bottom += emb_sorted[:, j]

    ax.set_xlabel("Items (sorted by dominant dimension)")
    ax.set_ylabel("Loading")
    ax.set_title("Loading profiles for all items")
    ax.legend(loc="upper right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "plot2_loading_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot2_loading_profiles.pdf")

    # =========================================================================
    # Plot 3: Dimension loadings as bar chart per item
    # =========================================================================
    fig, axes = plt.subplots(rank, 1, figsize=(14, 3*rank), sharex=True)

    for j, ax in enumerate(axes):
        sort_idx = np.argsort(embedding[:, j])[::-1]
        vals = embedding[sort_idx, j]
        colors = [CYCLE[dominant[i] % len(CYCLE)] for i in sort_idx]

        ax.bar(range(n_samples), vals, color=colors, width=0.8)
        ax.axhline(embedding[:, j].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(f"D{j+1} loading", color=CYCLE[j % len(CYCLE)])
        ax.set_title(f"Dimension {j+1} loadings (colored by dominant dim)", fontsize=10)
        despine(ax)

    axes[-1].set_xlabel("Items (sorted by this dimension)")
    plt.tight_layout()
    fig.savefig(output_dir / "plot3_dimension_loadings.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot3_dimension_loadings.pdf")

    # =========================================================================
    # Plot 4: Top and bottom 15 for each dimension with images
    # =========================================================================
    k = 15
    fig, axes = plt.subplots(rank, 2, figsize=(16, 3.5*rank))

    for j in range(rank):
        sort_idx = np.argsort(embedding[:, j])

        # Bottom k (low loading)
        ax = axes[j, 0]
        bottom_idx = sort_idx[:k]
        for pos, idx in enumerate(bottom_idx):
            img = images[idx]
            imagebox = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(imagebox, (pos, 0.5), frameon=True, pad=0.05,
                               bboxprops=dict(edgecolor=GRAY, linewidth=1))
            ax.add_artist(ab)
            ax.text(pos, -0.15, f"{embedding[idx, j]:.2f}", ha="center", fontsize=7)
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(-0.35, 1.1)
        ax.set_title(f"D{j+1} LOWEST {k}", fontsize=10, color=GRAY_DARK)
        ax.axis("off")

        # Top k (high loading)
        ax = axes[j, 1]
        top_idx = sort_idx[-k:][::-1]
        for pos, idx in enumerate(top_idx):
            img = images[idx]
            imagebox = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(imagebox, (pos, 0.5), frameon=True, pad=0.05,
                               bboxprops=dict(edgecolor=CYCLE[j % len(CYCLE)], linewidth=2))
            ax.add_artist(ab)
            ax.text(pos, -0.15, f"{embedding[idx, j]:.2f}", ha="center", fontsize=7)
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(-0.35, 1.1)
        ax.set_title(f"D{j+1} HIGHEST {k}", fontsize=10, color=CYCLE[j % len(CYCLE)], fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "plot4_top_bottom_items.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved plot4_top_bottom_items.pdf")

    # =========================================================================
    # Plot 5: RSM with items grouped by dominant dimension
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Original RSM
    ax = axes[0]
    im = ax.imshow(rsm, cmap="viridis")
    ax.set_title("Original RSM (unsorted)")
    ax.set_xlabel("Items")
    ax.set_ylabel("Items")

    # RSM sorted by dominant dimension
    ax = axes[1]
    sort_idx = np.lexsort((embedding[:, 0], dominant))  # Sort by dominant, then D1
    rsm_sorted = rsm[np.ix_(sort_idx, sort_idx)]
    im = ax.imshow(rsm_sorted, cmap="viridis")
    ax.set_title("RSM sorted by dominant dimension")
    ax.set_xlabel("Items")
    ax.set_ylabel("Items")

    # Add lines to show groups
    cumsum = np.cumsum([(dominant == j).sum() for j in range(rank)])
    for c in cumsum[:-1]:
        ax.axhline(c - 0.5, color="white", linewidth=2)
        ax.axvline(c - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "plot5_rsm_sorted.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot5_rsm_sorted.pdf")

    # =========================================================================
    # Plot 6: Check reconstruction - what does the model actually see?
    # =========================================================================
    recon = embedding @ embedding.T
    residual = rsm - recon

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    vmax = max(abs(rsm).max(), abs(recon).max())

    im = axes[0].imshow(rsm, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Original RSM")

    im = axes[1].imshow(recon, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("Reconstruction (WW^T)")

    im = axes[2].imshow(residual, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[2].set_title(f"Residual (RMSE={np.sqrt((residual**2).mean()):.3f})")

    for ax in axes:
        ax.set_xlabel("Items")
        ax.set_ylabel("Items")

    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "plot6_reconstruction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot6_reconstruction.pdf")

    # =========================================================================
    # Plot 7: Scatter matrix - pairwise dimension relationships
    # =========================================================================
    fig, axes = plt.subplots(rank, rank, figsize=(10, 10))

    for i in range(rank):
        for j in range(rank):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                ax.hist(embedding[:, i], bins=20, color=CYCLE[i % len(CYCLE)], alpha=0.7)
                ax.set_ylabel(f"D{i+1}")
            else:
                # Scatter off-diagonal
                ax.scatter(embedding[:, j], embedding[:, i], c=[CYCLE[d % len(CYCLE)] for d in dominant],
                          s=20, alpha=0.7)

            if i == rank - 1:
                ax.set_xlabel(f"D{j+1}")
            if j == 0 and i != j:
                ax.set_ylabel(f"D{i+1}")

            despine(ax)

    fig.suptitle("Dimension Scatter Matrix", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "plot7_scatter_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot7_scatter_matrix.pdf")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("INTERPRETATION HINTS")
    print("=" * 60)
    print("""
Key observations to check in the plots:

1. plot1_all_items_sorted.pdf - See ALL items sorted by each dimension
   → Look for semantic patterns (animals together? tools together?)

2. plot4_top_bottom_items.pdf - Top and bottom 15 for each dimension
   → What is HIGH on D1? What is LOW on D1? Same for D2, D3

3. plot5_rsm_sorted.pdf - RSM grouped by dominant dimension
   → Are the blocks clean? Or is there structure within blocks?

4. plot7_scatter_matrix.pdf - Relationships between dimensions
   → Are dimensions independent or correlated?

If dimensions are not "pure" categories, possible reasons:
- The similarity structure has more than 3 clean clusters
- Some items genuinely belong to multiple categories
- The dimensions capture continuous gradients, not discrete categories
""")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
