"""SRF on ResNet50 features for THINGS+ images."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pysrf import SRF

from src.datasets import load_dataset
from src.tools.rsa import compute_similarity
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
RANK = 10
TOP_K = 8

MODEL_ROOT = "/SSD/projects/deepsim/raw/features/architecture/IMAGENET1K_V1/resnet50"


def load_rsm():
    """Load DNN features and compute RSM."""
    print(f"Loading ResNet50 features (THINGS+ subset)...")
    ds = load_dataset("dnn", root=MODEL_ROOT, filter_plus=True)
    print(f"Features shape: {ds.data.shape}")

    print("Computing RSM with linear kernel...")
    rsm = compute_similarity(ds.data, ds.data, "linear")
    print(f"RSM shape: {rsm.shape}, range: [{rsm.min():.2f}, {rsm.max():.2f}]")

    return rsm, ds.metadata


def fit_srf(rsm, rank):
    """Fit SRF model."""
    print(f"\nFitting SRF with rank={rank}...")
    model = SRF(rank=rank, max_outer=500, verbose=1)
    model.fit(rsm)
    print(f"Embedding shape: {model.w_.shape}")
    return model.w_


def plot_all_dimensions(
    embedding: np.ndarray,
    paths: list[str],
    categories: list[str],
    top_k: int = 8,
) -> plt.Figure:
    """Plot top-k images for all dimensions in a grid."""
    rank = embedding.shape[1]
    fig, axes = plt.subplots(rank, top_k, figsize=(top_k * 1.5, rank * 1.8))

    for d in range(rank):
        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:top_k]

        for i, idx in enumerate(top_idx):
            img_path = paths[idx]
            img = Image.open(img_path)
            axes[d, i].imshow(img)
            if i == 0:
                axes[d, i].set_ylabel(f"Dim {d}", fontsize=10, fontweight="bold")
            axes[d, i].set_title(f"{categories[idx][:10]}", fontsize=6)
            axes[d, i].axis("off")

    plt.tight_layout()
    return fig


def main():
    rsm, metadata = load_rsm()
    paths = metadata["paths"]
    categories = metadata["categories"]

    embedding = fit_srf(rsm, RANK)

    np.save(OUTPUT_DIR / "embedding.npy", embedding)
    np.save(OUTPUT_DIR / "rsm.npy", rsm)

    print("\nPlotting dimensions...")
    fig = plot_all_dimensions(embedding, paths, categories, top_k=TOP_K)
    fig.savefig(OUTPUT_DIR / "all_dimensions.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Top categories per dimension:")
    print("=" * 60)
    for d in range(RANK):
        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:10]
        top_cats = [categories[i] for i in top_idx]
        print(f"Dim {d}: {', '.join(top_cats)}")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
