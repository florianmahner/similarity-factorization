#!/usr/bin/env python3
"""Plot top images per SRF dimension."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
THINGS_DIR = Path("/SSD/datasets/things/behav1854")
N_TOP = 10
N_DIMS = 10


def load_image(obj_name: str) -> np.ndarray:
    """Load THINGS image for an object."""
    img_path = THINGS_DIR / obj_name / f"{obj_name}_01b.jpg"
    if img_path.exists():
        return imread(img_path)
    return np.zeros((100, 100, 3), dtype=np.uint8)


def main():
    # Load SRF results
    results_path = Path(__file__).parent / "outputs/260116_123702/srf_results.npz"
    data = np.load(results_path, allow_pickle=True)
    W = data["embedding"]
    classes = data["classes"]

    print(f"Embedding: {W.shape}")
    print(f"Classes: {len(classes)}")

    # Plot top images per dimension
    fig, axes = plt.subplots(N_DIMS, N_TOP, figsize=(N_TOP * 1.2, N_DIMS * 1.2))

    for d in range(N_DIMS):
        top_idx = np.argsort(W[:, d])[-N_TOP:][::-1]
        top_objects = classes[top_idx]

        for i, obj in enumerate(top_objects):
            ax = axes[d, i]
            img = load_image(obj)
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "embedding_images.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'embedding_images.png'}")


if __name__ == "__main__":
    main()
