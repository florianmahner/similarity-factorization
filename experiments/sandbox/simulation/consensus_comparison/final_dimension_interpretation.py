"""
Final interpretation of mur92 dimensions.
Shows what each dimension captures with clear visualizations.
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
from src.colors import CYCLE, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "dimension_interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    image_paths = dataset.metadata.get("images", [])
    n_samples = rsm.shape[0]

    # Load images at different sizes
    images_small = []
    images_large = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            small = img.copy()
            small.thumbnail((50, 50))
            images_small.append(np.array(small))

            large = img.copy()
            large.thumbnail((100, 100))
            images_large.append(np.array(large))
        except:
            images_small.append(np.zeros((50, 50, 3), dtype=np.uint8))
            images_large.append(np.zeros((100, 100, 3), dtype=np.uint8))

    # Fit rank=3 embedding
    print("Fitting rank=3 embedding...")
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(SRF(rank=3), n_runs=50, n_jobs=-1, random_state=42)),
        ("consensus", AlignedConsensus(rank=3, aggregation="refine")),
    ])
    embedding = pipeline.fit_transform(rsm)
    rank = 3

    # =========================================================================
    # Figure 1: What each dimension captures (large, clear visualization)
    # =========================================================================
    fig = plt.figure(figsize=(16, 12))

    for j in range(rank):
        # Top 20 items for this dimension
        sort_idx = np.argsort(embedding[:, j])[::-1]
        top_20 = sort_idx[:20]

        # Create subplot with 2 rows of 10
        for row in range(2):
            for col in range(10):
                idx = row * 10 + col
                if idx >= 20:
                    continue
                item_idx = top_20[idx]

                ax = fig.add_subplot(rank * 2, 10, j * 20 + idx + 1)
                ax.imshow(images_large[item_idx])
                ax.set_title(f"{embedding[item_idx, j]:.2f}", fontsize=8)
                ax.axis("off")

                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(CYCLE[j % len(CYCLE)])
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        # Add dimension label
        fig.text(0.02, 1 - (j + 0.5) / rank, f"D{j+1}", fontsize=16, fontweight="bold",
                color=CYCLE[j % len(CYCLE)], va="center", ha="left")

    plt.suptitle("mur92: Top 20 Items per Dimension (rank=3)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_top20_per_dimension.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved fig1_top20_per_dimension.pdf")

    # =========================================================================
    # Figure 2: Bottom 20 items (what each dimension does NOT capture)
    # =========================================================================
    fig = plt.figure(figsize=(16, 12))

    for j in range(rank):
        sort_idx = np.argsort(embedding[:, j])  # Low to high
        bottom_20 = sort_idx[:20]

        for row in range(2):
            for col in range(10):
                idx = row * 10 + col
                if idx >= 20:
                    continue
                item_idx = bottom_20[idx]

                ax = fig.add_subplot(rank * 2, 10, j * 20 + idx + 1)
                ax.imshow(images_large[item_idx])
                ax.set_title(f"{embedding[item_idx, j]:.2f}", fontsize=8)
                ax.axis("off")

                for spine in ax.spines.values():
                    spine.set_edgecolor(GRAY)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

        fig.text(0.02, 1 - (j + 0.5) / rank, f"D{j+1}\nLOW", fontsize=12,
                color=GRAY_DARK, va="center", ha="left")

    plt.suptitle("mur92: Bottom 20 Items per Dimension (what's NOT in each dim)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_bottom20_per_dimension.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved fig2_bottom20_per_dimension.pdf")

    # =========================================================================
    # Figure 3: All 92 items sorted by each dimension (comprehensive view)
    # =========================================================================
    for j in range(rank):
        fig = plt.figure(figsize=(20, 10))
        sort_idx = np.argsort(embedding[:, j])[::-1]

        cols = 10
        rows = (n_samples + cols - 1) // cols

        for pos, idx in enumerate(sort_idx):
            ax = fig.add_subplot(rows, cols, pos + 1)
            ax.imshow(images_small[idx])

            # Color border by loading strength
            loading = embedding[idx, j]
            max_loading = embedding[:, j].max()
            intensity = loading / max_loading

            for spine in ax.spines.values():
                spine.set_edgecolor(CYCLE[j % len(CYCLE)])
                spine.set_linewidth(2 * intensity + 0.5)
                spine.set_visible(True)

            ax.set_title(f"{loading:.2f}", fontsize=6)
            ax.axis("off")

        plt.suptitle(f"All 92 items sorted by Dimension {j+1} (high→low)", fontsize=14,
                    color=CYCLE[j % len(CYCLE)], fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / f"fig3_all_sorted_by_D{j+1}.pdf", bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved fig3_all_sorted_by_D{j+1}.pdf")

    # =========================================================================
    # Figure 4: Mixed items (items that load on multiple dimensions)
    # =========================================================================
    # Find items with most uniform loadings
    loading_std = np.std(embedding, axis=1)
    mixed_idx = np.argsort(loading_std)[:15]  # Most uniform = most mixed

    fig, ax = plt.subplots(figsize=(16, 4))

    for pos, idx in enumerate(mixed_idx):
        # Show image
        imagebox = OffsetImage(images_large[idx], zoom=0.8)
        ab = AnnotationBbox(imagebox, (pos, 0.6), frameon=True, pad=0.05)
        ax.add_artist(ab)

        # Show bar chart of loadings below
        bar_width = 0.25
        for j in range(rank):
            ax.bar(pos + (j - 1) * bar_width, embedding[idx, j] * 0.4,
                  width=bar_width, color=CYCLE[j % len(CYCLE)], bottom=-0.5)

        # Loading values
        ax.text(pos, -0.1, f"{embedding[idx, 0]:.2f}\n{embedding[idx, 1]:.2f}\n{embedding[idx, 2]:.2f}",
               ha="center", fontsize=6, va="top")

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.6, 1.2)
    ax.axhline(0, color=GRAY_LIGHT, linestyle="-", linewidth=0.5)
    ax.set_title("Most 'mixed' items (load similarly on all dimensions)", fontsize=12)
    ax.axis("off")

    # Legend
    for j in range(rank):
        ax.bar([], [], color=CYCLE[j % len(CYCLE)], label=f"D{j+1}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_mixed_items.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved fig4_mixed_items.pdf")

    # =========================================================================
    # Figure 5: Pure items (items that load strongly on ONE dimension only)
    # =========================================================================
    # Find items with highest loading ratio (max/sum)
    purity = embedding.max(axis=1) / embedding.sum(axis=1)
    dominant = np.argmax(embedding, axis=1)

    fig, axes = plt.subplots(1, rank, figsize=(16, 4))

    for j, ax in enumerate(axes):
        # Get purest items for this dimension
        mask = dominant == j
        purity_j = purity[mask]
        items_j = np.where(mask)[0]
        pure_order = np.argsort(purity_j)[::-1][:10]
        pure_items = items_j[pure_order]

        for pos, idx in enumerate(pure_items):
            imagebox = OffsetImage(images_large[idx], zoom=0.7)
            row = pos // 5
            col = pos % 5
            ab = AnnotationBbox(imagebox, (col, 1 - row * 0.6), frameon=True, pad=0.05,
                               bboxprops=dict(edgecolor=CYCLE[j % len(CYCLE)], linewidth=2))
            ax.add_artist(ab)
            ax.text(col, 0.65 - row * 0.6, f"{purity[idx]:.2f}", ha="center", fontsize=7)

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 1.4)
        ax.set_title(f"D{j+1} purest items", fontsize=11, color=CYCLE[j % len(CYCLE)], fontweight="bold")
        ax.axis("off")

    plt.suptitle("Purest items per dimension (load strongly on ONE dimension only)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "fig5_pure_items.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved fig5_pure_items.pdf")

    # =========================================================================
    # Print interpretation
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIMENSION INTERPRETATION")
    print("=" * 60)

    print("""
Look at the generated figures to interpret each dimension:

1. fig1_top20_per_dimension.pdf
   → What's HIGH on each dimension?
   → D1 might be: faces/animate?
   → D2 might be: objects/inanimate?
   → D3 might be: ???

2. fig2_bottom20_per_dimension.pdf
   → What's LOW on each dimension?
   → This helps confirm what each dimension captures

3. fig3_all_sorted_by_D*.pdf
   → Full ranking of all 92 items by each dimension
   → See the continuous gradient from high to low

4. fig4_mixed_items.pdf
   → Items that don't fit cleanly into any category
   → These are inherently ambiguous stimuli

5. fig5_pure_items.pdf
   → Prototypical examples of each dimension
   → These most clearly represent what each dim captures
""")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
