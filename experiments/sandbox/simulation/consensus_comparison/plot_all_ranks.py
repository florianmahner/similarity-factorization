"""
Visualize embeddings for ranks 2-6 for mur92.
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
from src.colors import ROSE, TEAL, CYAN, SAND, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "all_ranks"
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
            img.thumbnail((80, 80))
            images.append(np.array(img))
        except:
            images.append(np.zeros((80, 80, 3), dtype=np.uint8))

    # Extended color palette for more dimensions
    colors = [ROSE, TEAL, CYAN, SAND, "#FF9500", "#00CED1", "#9370DB", "#20B2AA"]

    ranks = [2, 3, 4, 5, 6]

    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"RANK {rank}")
        print(f"{'='*60}")

        # Fit embedding
        pipeline = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=50, n_jobs=-1, random_state=42)),
            ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
        ])
        embedding = pipeline.fit_transform(rsm)

        # Compute metrics
        purity = embedding.max(axis=1) / embedding.sum(axis=1)
        dominant = np.argmax(embedding, axis=1)
        recon = embedding @ embedding.T
        error = np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")

        print(f"Error: {error:.4f}, Mean purity: {purity.mean():.3f}")
        print(f"Items per dimension: {[(dominant == j).sum() for j in range(rank)]}")

        # =====================================================================
        # Figure 1: Top 15 items per dimension
        # =====================================================================
        fig = plt.figure(figsize=(16, 2.5 * rank))

        for j in range(rank):
            top_idx = np.argsort(embedding[:, j])[-15:][::-1]

            for pos, idx in enumerate(top_idx):
                ax = fig.add_subplot(rank, 15, j * 15 + pos + 1)
                ax.imshow(images[idx])
                ax.set_title(f"{embedding[idx, j]:.2f}", fontsize=7)
                ax.axis("off")

                for spine in ax.spines.values():
                    spine.set_edgecolor(colors[j % len(colors)])
                    spine.set_linewidth(2)
                    spine.set_visible(True)

            # Dimension label on left
            fig.text(0.01, 1 - (j + 0.5) / rank, f"D{j+1}", fontsize=12, fontweight="bold",
                    color=colors[j % len(colors)], va="center", ha="left")

        plt.suptitle(f"Rank {rank}: Top 15 items per dimension (error={error:.4f}, purity={purity.mean():.3f})",
                    fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / f"rank{rank}_top15.pdf", bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved rank{rank}_top15.pdf")

        # =====================================================================
        # Figure 2: Embedding heatmap
        # =====================================================================
        fig, ax = plt.subplots(figsize=(max(4, rank), 8))

        # Sort by dominant dimension, then by loading
        sort_key = dominant * 1000 - embedding.max(axis=1)
        sort_idx = np.argsort(sort_key)

        im = ax.imshow(embedding[sort_idx], aspect="auto", cmap="viridis")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Items (sorted by dominant dim)")
        ax.set_xticks(range(rank))
        ax.set_xticklabels([f"D{j+1}" for j in range(rank)])

        # Add horizontal lines between groups
        cumsum = np.cumsum([(dominant == j).sum() for j in range(rank)])
        for c in cumsum[:-1]:
            ax.axhline(c - 0.5, color="white", linewidth=1)

        fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
        ax.set_title(f"Rank {rank} Embedding Heatmap")

        plt.tight_layout()
        fig.savefig(output_dir / f"rank{rank}_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved rank{rank}_heatmap.pdf")

        # =====================================================================
        # Figure 3: All items grouped by dominant dimension
        # =====================================================================
        fig = plt.figure(figsize=(14, 3 * ((n_samples // rank + 15) // 10)))

        # Sort items by dominant dimension
        sort_idx = np.argsort(dominant * 1000 - purity)

        cols = 12
        rows = (n_samples + cols - 1) // cols

        for pos, idx in enumerate(sort_idx):
            ax = fig.add_subplot(rows, cols, pos + 1)
            ax.imshow(images[idx])
            ax.axis("off")

            dom = dominant[idx]
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[dom % len(colors)])
                spine.set_linewidth(2)
                spine.set_visible(True)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[j % len(colors)],
                                         label=f"D{j+1} ({(dominant==j).sum()})") for j in range(rank)]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.suptitle(f"Rank {rank}: Items grouped by dominant dimension", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / f"rank{rank}_grouped.pdf", bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved rank{rank}_grouped.pdf")

        # =====================================================================
        # Figure 4: Pairwise 2D projections (if rank > 2)
        # =====================================================================
        if rank <= 4:
            n_pairs = rank * (rank - 1) // 2
            fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
            if n_pairs == 1:
                axes = [axes]

            pair_idx = 0
            for i in range(rank):
                for j in range(i + 1, rank):
                    ax = axes[pair_idx]
                    scatter_colors = [colors[d % len(colors)] for d in dominant]
                    ax.scatter(embedding[:, i], embedding[:, j], c=scatter_colors, s=30, alpha=0.7)
                    ax.set_xlabel(f"D{i+1}")
                    ax.set_ylabel(f"D{j+1}")
                    ax.set_title(f"D{i+1} vs D{j+1}")
                    despine(ax)
                    pair_idx += 1

            plt.suptitle(f"Rank {rank}: 2D Projections", fontsize=12)
            plt.tight_layout()
            fig.savefig(output_dir / f"rank{rank}_2d_projections.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved rank{rank}_2d_projections.pdf")

    # =========================================================================
    # Summary figure: Compare all ranks
    # =========================================================================
    print("\nGenerating summary...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, rank in enumerate(ranks):
        ax = axes[idx]

        # Refit quickly
        pipeline = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=50, n_jobs=-1, random_state=42)),
            ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
        ])
        embedding = pipeline.fit_transform(rsm)

        purity = embedding.max(axis=1) / embedding.sum(axis=1)
        dominant = np.argmax(embedding, axis=1)
        error = np.linalg.norm(rsm - embedding @ embedding.T, "fro") / np.linalg.norm(rsm, "fro")

        # Stacked bar showing loading distribution
        sort_idx = np.argsort(dominant * 1000 - purity)
        emb_sorted = embedding[sort_idx]

        bottom = np.zeros(n_samples)
        for j in range(rank):
            ax.bar(range(n_samples), emb_sorted[:, j], bottom=bottom,
                   color=colors[j % len(colors)], width=1.0, label=f"D{j+1}")
            bottom += emb_sorted[:, j]

        ax.set_xlabel("Items")
        ax.set_ylabel("Loading")
        ax.set_title(f"Rank {rank} (error={error:.3f}, purity={purity.mean():.2f})")
        ax.legend(loc="upper right", fontsize=7, ncol=min(rank, 3))
        despine(ax)

    # Hide last subplot if odd number
    if len(ranks) < 6:
        axes[-1].axis("off")

    plt.suptitle("Loading Profiles Across Ranks", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_all_ranks.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved summary_all_ranks.pdf")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
