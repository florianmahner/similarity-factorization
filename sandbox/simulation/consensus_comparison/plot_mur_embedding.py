"""
Visualize the mur92 embedding from consensus SRF.
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
from src.colors import CYCLE, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure


def main():
    output_dir = Path(__file__).parent / "outputs" / "mur_embedding"
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

    print(f"Embedding shape: {embedding.shape}")
    print(f"Number of images: {len(image_paths)}")

    # Load images
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((64, 64))
            images.append(np.array(img))
        except Exception as e:
            print(f"Could not load {p}: {e}")
            images.append(np.zeros((64, 64, 3), dtype=np.uint8))

    # =========================================================================
    # Plot 1: 3D scatter with color by dominant dimension
    # =========================================================================
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Color by dominant dimension
    dominant_dim = np.argmax(embedding, axis=1)
    colors = [CYCLE[d % len(CYCLE)] for d in dominant_dim]

    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=colors, s=60, alpha=0.8, edgecolors="white", linewidths=0.5
    )

    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_zlabel("Dimension 3", fontsize=10)
    ax.set_title("mur92 Embedding (colored by dominant dimension)", fontsize=11)

    # Legend
    for i in range(rank):
        ax.scatter([], [], [], c=CYCLE[i % len(CYCLE)], s=60, label=f"Dim {i+1} dominant")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "plot1_3d_embedding.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved plot1_3d_embedding.pdf")

    # =========================================================================
    # Plot 2: 2D projections (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [("D1", "D2"), ("D1", "D3"), ("D2", "D3")]

    for ax, (i, j), (xlabel, ylabel) in zip(axes, pairs, labels):
        ax.scatter(embedding[:, i], embedding[:, j], c=colors, s=40, alpha=0.7,
                  edgecolors="white", linewidths=0.3)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        despine(ax)

    fig.suptitle("mur92 Embedding - 2D Projections", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "plot2_2d_projections.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot2_2d_projections.pdf")

    # =========================================================================
    # Plot 3: 2D projection with images
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use D1 vs D2
    x, y = embedding[:, 0], embedding[:, 1]
    ax.scatter(x, y, c=colors, s=20, alpha=0.3)

    # Add images
    for i, (xi, yi, img) in enumerate(zip(x, y, images)):
        imagebox = OffsetImage(img, zoom=0.35)
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False, pad=0)
        ax.add_artist(ab)

    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.set_title("mur92 Embedding with Images (D1 vs D2)", fontsize=12)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "plot3_embedding_with_images_d1d2.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved plot3_embedding_with_images_d1d2.pdf")

    # =========================================================================
    # Plot 4: D1 vs D3 with images
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    x, y = embedding[:, 0], embedding[:, 2]
    ax.scatter(x, y, c=colors, s=20, alpha=0.3)

    for i, (xi, yi, img) in enumerate(zip(x, y, images)):
        imagebox = OffsetImage(img, zoom=0.35)
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False, pad=0)
        ax.add_artist(ab)

    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 3", fontsize=11)
    ax.set_title("mur92 Embedding with Images (D1 vs D3)", fontsize=12)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "plot4_embedding_with_images_d1d3.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved plot4_embedding_with_images_d1d3.pdf")

    # =========================================================================
    # Plot 5: Top items per dimension
    # =========================================================================
    fig, axes = plt.subplots(rank, 1, figsize=(14, 3 * rank))

    k = 10  # top-k items

    for j, ax in enumerate(axes):
        top_idx = np.argsort(embedding[:, j])[-k:][::-1]

        # Create image strip
        for pos, idx in enumerate(top_idx):
            img = images[idx]
            imagebox = OffsetImage(img, zoom=0.8)
            ab = AnnotationBbox(imagebox, (pos, 0.5), frameon=True, pad=0.1,
                               bboxprops=dict(edgecolor=CYCLE[j % len(CYCLE)], linewidth=2))
            ax.add_artist(ab)

            # Add loading value
            ax.text(pos, -0.1, f"{embedding[idx, j]:.2f}", ha="center", fontsize=8)

        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(-0.3, 1.1)
        ax.set_title(f"Dimension {j+1} - Top {k} items", fontsize=11, color=CYCLE[j % len(CYCLE)])
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "plot5_top_items_per_dimension.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved plot5_top_items_per_dimension.pdf")

    # =========================================================================
    # Plot 6: Embedding heatmap sorted by each dimension
    # =========================================================================
    fig, axes = plt.subplots(1, rank, figsize=(12, 8))

    for j, ax in enumerate(axes):
        sort_idx = np.argsort(embedding[:, j])[::-1]
        im = ax.imshow(embedding[sort_idx], aspect="auto", cmap="viridis")
        ax.set_title(f"Sorted by D{j+1}", fontsize=10)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Items" if j == 0 else "")
        ax.set_xticks(range(rank))
        ax.set_xticklabels([f"D{i+1}" for i in range(rank)])

    fig.colorbar(im, ax=axes, shrink=0.6, label="Loading")
    fig.suptitle("mur92 Embedding Heatmaps", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "plot6_embedding_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot6_embedding_heatmaps.pdf")

    # =========================================================================
    # Plot 7: Ternary-style plot (for rank=3)
    # =========================================================================
    if rank == 3:
        fig, ax = plt.subplots(figsize=(10, 9))

        # Normalize to sum to 1 for ternary-like visualization
        emb_norm = embedding / embedding.sum(axis=1, keepdims=True)

        # Convert to 2D coordinates (simplex projection)
        # Using barycentric coordinates
        x_tern = 0.5 * (2 * emb_norm[:, 1] + emb_norm[:, 2])
        y_tern = (np.sqrt(3) / 2) * emb_norm[:, 2]

        ax.scatter(x_tern, y_tern, c=colors, s=30, alpha=0.3)

        # Add images
        for i, (xi, yi, img) in enumerate(zip(x_tern, y_tern, images)):
            imagebox = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(imagebox, (xi, yi), frameon=False, pad=0)
            ax.add_artist(ab)

        # Draw triangle
        triangle = plt.Polygon([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)],
                              fill=False, edgecolor=GRAY, linewidth=2)
        ax.add_patch(triangle)

        # Labels at corners
        ax.text(-0.05, -0.05, "D1", fontsize=12, ha="center", color=CYCLE[0], fontweight="bold")
        ax.text(1.05, -0.05, "D2", fontsize=12, ha="center", color=CYCLE[1], fontweight="bold")
        ax.text(0.5, np.sqrt(3)/2 + 0.05, "D3", fontsize=12, ha="center", color=CYCLE[2], fontweight="bold")

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("mur92 Embedding - Ternary Plot", fontsize=12)

        plt.tight_layout()
        fig.savefig(output_dir / "plot7_ternary.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("Saved plot7_ternary.pdf")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
