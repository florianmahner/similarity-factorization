"""
Visualization for THINGS Macaque Embedding.

Plots:
1. RSM heatmap (subset for visibility)
2. Dimension contributions
3. Top concepts per dimension with images
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.colors import TEAL, GRAY_DARK, GRAY_PALE, setup_style
from src.utils.figure_theme import create_figure, save_figure, despine


THINGS_IMAGE_DIR = Path("/SSD/datasets/things/core")


def load_results(output_dir: Path) -> dict:
    """Load results from output directory."""
    return {
        "embedding": np.load(output_dir / "embedding.npy"),
        "rsm": np.load(output_dir / "rsm.npy"),
        "filenames": np.loadtxt(output_dir / "filenames.txt", dtype=str),
        "top_concepts": pd.read_csv(output_dir / "top_concepts.csv"),
        "metadata": json.load(open(output_dir / "results.json")),
    }


def load_things_image(filename: str, image_dir: Path = THINGS_IMAGE_DIR) -> np.ndarray | None:
    """Load THINGS image for a filename like 'oven_12s.jpg'."""
    import re
    # Extract concept name: "oven_12s.jpg" -> "oven"
    concept_name = re.sub(r"_\d+[a-z]?\.jpg$", "", filename)

    # Try exact file first
    img_path = image_dir / concept_name / filename
    if img_path.exists():
        return np.array(Image.open(img_path))

    # Fallback: any image in concept folder
    concept_dir = image_dir / concept_name
    if concept_dir.exists():
        images = list(concept_dir.glob("*.jpg"))
        if images:
            return np.array(Image.open(images[0]))

    return None


def plot_rsm_heatmap(rsm: np.ndarray, output_path: Path, n_show: int = 200) -> None:
    """Plot RSM heatmap (subset for visibility)."""
    setup_style()

    fig, ax = create_figure("square")

    rsm_subset = rsm[:n_show, :n_show]
    im = ax.imshow(rsm_subset, cmap="viridis", vmin=0, vmax=1)

    ax.set_xlabel("Concept")
    ax.set_ylabel("Concept")
    ax.set_title(f"RBF Kernel RSM (first {n_show} concepts)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity")

    save_figure(fig, output_path)
    plt.close(fig)


def plot_dimension_contributions(metadata: dict, output_path: Path) -> None:
    """Bar plot of dimension contributions."""
    setup_style()

    contributions = metadata["dim_contributions"]
    rank = len(contributions)

    fig, ax = create_figure("single")

    ax.bar(range(1, rank + 1), contributions, color=TEAL, alpha=0.85)
    ax.axhline(1 / rank, color=GRAY_DARK, linestyle="--", linewidth=1, label="Uniform")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Variance contribution")
    ax.set_xticks(range(1, rank + 1))

    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_top_concepts_text(
    embedding: np.ndarray,
    filenames: np.ndarray,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Plot top concepts per dimension as text list."""
    setup_style()

    rank = embedding.shape[1]

    fig, axes = plt.subplots(1, rank, figsize=(rank * 2, 4))

    for dim in range(rank):
        ax = axes[dim]
        loadings = embedding[:, dim]
        top_idx = np.argsort(loadings)[::-1][:top_k]

        for i, idx in enumerate(top_idx):
            ax.text(
                0.1, 0.9 - i * 0.08,
                f"{filenames[idx]} ({loadings[idx]:.2f})",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )

        ax.set_title(f"Dim {dim + 1}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Top concepts per dimension", fontsize=11, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_top_images_per_dimension(
    embedding: np.ndarray,
    filenames: np.ndarray,
    output_path: Path,
    top_k: int = 8,
    image_dir: Path = THINGS_IMAGE_DIR,
) -> None:
    """Plot top-k images for each dimension."""
    rank = embedding.shape[1]

    fig, axes = plt.subplots(rank, top_k, figsize=(top_k * 1.2, rank * 1.3))

    for dim in range(rank):
        loadings = embedding[:, dim]
        top_idx = np.argsort(loadings)[::-1][:top_k]

        for k, idx in enumerate(top_idx):
            ax = axes[dim, k] if rank > 1 else axes[k]
            concept = filenames[idx]
            img = load_things_image(concept, image_dir)

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, concept[:10], ha="center", va="center", fontsize=6)
                ax.set_facecolor(GRAY_PALE)

            ax.axis("off")

            if k == 0:
                ax.set_ylabel(f"Dim {dim+1}", fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(f"Top {top_k} concepts per dimension", fontsize=10, y=1.01)
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_embedding_scatter(embedding: np.ndarray, output_path: Path) -> None:
    """Scatter plot of first two dimensions."""
    setup_style()

    fig, ax = create_figure("square")

    ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5, c=TEAL)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Embedding (Dim 1 vs Dim 2)")

    despine(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def main(output_name: str = "gaussian"):
    output_dir = Path(__file__).parent / "outputs" / output_name

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        print("Run run.py first.")
        return

    print("Loading results...")
    results = load_results(output_dir)

    print("Generating plots...")

    plot_rsm_heatmap(results["rsm"], output_dir / "rsm_heatmap.pdf")
    print("  - rsm_heatmap.pdf")

    plot_dimension_contributions(results["metadata"], output_dir / "dimension_contributions.pdf")
    print("  - dimension_contributions.pdf")

    plot_top_concepts_text(
        results["embedding"],
        results["filenames"],
        output_dir / "top_concepts_text.pdf",
    )
    print("  - top_concepts_text.pdf")

    plot_embedding_scatter(results["embedding"], output_dir / "embedding_scatter.pdf")
    print("  - embedding_scatter.pdf")

    if THINGS_IMAGE_DIR.exists():
        print("  Loading THINGS images...")
        plot_top_images_per_dimension(
            results["embedding"],
            results["filenames"],
            output_dir / "top_images.pdf",
        )
        print("  - top_images.pdf")
    else:
        print(f"  THINGS images not found at {THINGS_IMAGE_DIR}, skipping image plot")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="2k_gaussian",
                        help="Output directory name (e.g., 2k_gaussian, 22k_gaussian)")
    args = parser.parse_args()
    main(output_name=args.output)
