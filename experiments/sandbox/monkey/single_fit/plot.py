"""Plot top-12 images per dimension for monkey IT single SRF fit."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

THINGS_DIR = Path("/SSD/datasets/things/core")
OUTPUT_DIR = Path(__file__).parent / "outputs"
K = 12


def _build_image_paths(exemplars: np.ndarray, filenames: np.ndarray) -> list[Path | None]:
    """Map each of the 22248 items to its THINGS image using exemplar IDs."""
    paths = []
    for exemplar, category in zip(exemplars, filenames):
        img_path = THINGS_DIR / category / f"{exemplar}.jpg"
        paths.append(img_path if img_path.exists() else None)
    return paths


def main():
    embedding = np.load(OUTPUT_DIR / "embedding.npy")
    n_items, n_dims = embedding.shape
    log.info(f"Embedding: {embedding.shape}")

    import sys; sys.path.insert(0, "src")
    from src.datasets.monkey import DATA_ROOT
    d = np.load(DATA_ROOT / "22k" / "processed" / "F.npy", allow_pickle=True).item()
    filenames = np.array(d["stimuli"])
    exemplars = np.array(d["exemplars"])

    image_paths = _build_image_paths(exemplars, filenames)
    found = sum(1 for p in image_paths if p is not None)
    log.info(f"Image paths: {found}/{len(image_paths)} found")

    fig, axes = plt.subplots(n_dims, K, figsize=(1.2 * K, 1.4 * n_dims))

    for dim in range(n_dims):
        top_idx = np.argsort(embedding[:, dim])[::-1][:K]
        for col, idx in enumerate(top_idx):
            ax = axes[dim, col]
            p = image_paths[idx]
            if p is not None and p.exists():
                img = Image.open(p).convert("RGB").resize((64, 64))
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[dim, 0].set_ylabel(f"D{dim+1}", fontsize=9, fontweight="bold",
                                 rotation=0, labelpad=15, va="center")

    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    out_path = OUTPUT_DIR / "topk_dimensions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
