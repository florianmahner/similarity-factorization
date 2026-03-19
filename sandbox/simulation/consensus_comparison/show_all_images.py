"""
Show all mur92 images in a grid to understand the categories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.datasets.loaders import load_mur92


def main():
    output_dir = Path(__file__).parent / "outputs" / "image_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    image_paths = dataset.metadata.get("images", [])

    # Load all images
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((100, 100))
            images.append(np.array(img))
        except:
            images.append(np.zeros((100, 100, 3), dtype=np.uint8))

    # Create grid
    n = len(images)
    cols = 10
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 2*rows))
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.set_title(f"im{i+1:02d}", fontsize=8)
        ax.axis("off")

    # Hide empty axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("mur92 - All 92 Images", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "all_images.pdf", bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Saved to {output_dir / 'all_images.pdf'}")

    # Also print what we observe
    print("\nImage numbering patterns observed:")
    print("  im01-im12: First group")
    print("  im13-im24: Second group")
    print("  im25-im36: Third group")
    print("  ...")


if __name__ == "__main__":
    main()
