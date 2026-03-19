"""
Compare different ranks to see which gives cleanest category separation.
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
from src.colors import TEAL, CYAN, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure


def compute_purity(embedding):
    """Compute how "pure" each item's membership is (max/sum)."""
    max_loading = embedding.max(axis=1)
    sum_loading = embedding.sum(axis=1)
    return max_loading / sum_loading


def main():
    output_dir = Path(__file__).parent / "outputs" / "rank_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    image_paths = dataset.metadata.get("images", [])
    n_samples = rsm.shape[0]

    # Load images
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((60, 60))
            images.append(np.array(img))
        except:
            images.append(np.zeros((60, 60, 3), dtype=np.uint8))

    # Test different ranks
    ranks = [2, 3, 4, 5, 6]
    results = []

    for rank in ranks:
        print(f"\nRank {rank}...")
        pipeline = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=50, n_jobs=-1, random_state=42)),
            ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
        ])
        embedding = pipeline.fit_transform(rsm)

        # Compute metrics
        purity = compute_purity(embedding)
        recon = embedding @ embedding.T
        error = np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")

        # Dominant dimension distribution
        dominant = np.argmax(embedding, axis=1)
        dom_counts = [(dominant == j).sum() for j in range(rank)]

        results.append({
            "rank": rank,
            "error": error,
            "mean_purity": purity.mean(),
            "min_purity": purity.min(),
            "dom_counts": dom_counts,
            "embedding": embedding,
            "dominant": dominant,
        })

        print(f"  Error: {error:.4f}")
        print(f"  Mean purity: {purity.mean():.3f}")
        print(f"  Dominant counts: {dom_counts}")

    # =========================================================================
    # Plot 1: Purity and error vs rank
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    ranks_arr = [r["rank"] for r in results]

    ax = axes[0]
    ax.plot(ranks_arr, [r["error"] for r in results], "o-", color=TEAL, linewidth=2, markersize=8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Error vs Rank")
    despine(ax)

    ax = axes[1]
    ax.plot(ranks_arr, [r["mean_purity"] for r in results], "o-", color=CYAN, linewidth=2, markersize=8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean purity (max/sum)")
    ax.set_title("Purity vs Rank")
    despine(ax)

    save_figure(fig, output_dir / "plot1_rank_metrics.pdf")
    plt.close(fig)
    print("\nSaved plot1_rank_metrics.pdf")

    # =========================================================================
    # Plot 2: Top items per dimension for each rank
    # =========================================================================
    for res in results:
        rank = res["rank"]
        embedding = res["embedding"]

        fig, axes = plt.subplots(rank, 1, figsize=(14, 2.5*rank))
        if rank == 1:
            axes = [axes]

        k = 12
        for j, ax in enumerate(axes):
            top_idx = np.argsort(embedding[:, j])[-k:][::-1]

            for pos, idx in enumerate(top_idx):
                img = images[idx]
                imagebox = OffsetImage(img, zoom=0.7)
                ab = AnnotationBbox(imagebox, (pos, 0.5), frameon=True, pad=0.05,
                                   bboxprops=dict(edgecolor=CYCLE[j % len(CYCLE)], linewidth=2))
                ax.add_artist(ab)
                ax.text(pos, -0.1, f"{embedding[idx, j]:.2f}", ha="center", fontsize=7)

            ax.set_xlim(-0.5, k - 0.5)
            ax.set_ylim(-0.25, 1.1)
            ax.set_title(f"Dimension {j+1}", fontsize=10, color=CYCLE[j % len(CYCLE)], fontweight="bold")
            ax.axis("off")

        fig.suptitle(f"Rank {rank} (purity={res['mean_purity']:.3f}, error={res['error']:.4f})", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / f"plot2_rank{rank}_top_items.pdf", bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Saved plot2_rank{rank}_top_items.pdf")

    # =========================================================================
    # Plot 3: RSM sorted by dominant dimension for rank=3
    # =========================================================================
    res = results[1]  # rank=3
    embedding = res["embedding"]
    dominant = res["dominant"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by dominant dim, then by loading within that dim
    sort_idx = np.lexsort((-embedding.max(axis=1), dominant))

    # RSM sorted
    ax = axes[0]
    rsm_sorted = rsm[np.ix_(sort_idx, sort_idx)]
    im = ax.imshow(rsm_sorted, cmap="viridis")
    ax.set_title("RSM sorted by dominant dimension (rank=3)")

    # Add group boundaries
    cumsum = np.cumsum([(dominant == j).sum() for j in range(3)])
    for c in cumsum[:-1]:
        ax.axhline(c - 0.5, color="white", linewidth=2)
        ax.axvline(c - 0.5, color="white", linewidth=2)

    # Show images along the diagonal
    ax = axes[1]
    for pos, idx in enumerate(sort_idx):
        img = images[idx]
        imagebox = OffsetImage(img, zoom=0.45)
        row = pos // 10
        col = pos % 10
        ab = AnnotationBbox(imagebox, (col, 9-row), frameon=True, pad=0.02,
                           bboxprops=dict(edgecolor=CYCLE[dominant[idx] % len(CYCLE)], linewidth=1.5))
        ax.add_artist(ab)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_title("Items sorted by dominant dimension")
    ax.axis("off")

    # Add legend
    for j in range(3):
        ax.scatter([], [], c=CYCLE[j % len(CYCLE)], s=100, label=f"D{j+1}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "plot3_sorted_items.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved plot3_sorted_items.pdf")

    # =========================================================================
    # Analysis: What categories exist in mur92?
    # =========================================================================
    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)

    # Look at rank=3 top items for each dimension
    res = results[1]
    embedding = res["embedding"]

    print("\nTop 15 items per dimension (rank=3):")
    for j in range(3):
        top_idx = np.argsort(embedding[:, j])[-15:][::-1]
        print(f"\n  D{j+1}: {[Path(image_paths[i]).stem for i in top_idx]}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The mur92 dataset may not have exactly 3 clean categories.
Looking at the images will reveal what each dimension captures.

Possible interpretations:
1. Dimensions might capture GRADIENTS rather than discrete categories
   e.g., animacy, size, complexity rather than animal/tool/fruit

2. Some items might genuinely belong to multiple categories
   e.g., a coconut could be both "fruit" and "round object"

3. The similarity judgments might reflect multiple overlapping features
   not just category membership
""")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
