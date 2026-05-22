"""Quick SRF on monkey 22k data with visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

from pysrf import SRF
from src.datasets.loaders import load_things_monkey
from src.tools.metrics import gaussian_kernel_similarity
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
THINGS_IMG_DIR = Path("/SSD/datasets/things/core")

RANK = 10
MIN_RELIAB = 0.3
N_TOP = 8


def load_image(filename: str) -> np.ndarray:
    """Load THINGS image by filename (e.g., 'aardvark_02s.jpg')."""
    obj_name = Path(filename).stem.rsplit("_", 1)[0]
    img_path = THINGS_IMG_DIR / obj_name / filename
    if img_path.exists():
        return imread(img_path)
    return np.zeros((100, 100, 3), dtype=np.uint8)


def plot_dimensions(
    embedding: np.ndarray,
    filenames: list[str],
    title: str,
    n_top: int = N_TOP,
) -> plt.Figure:
    """Plot top images per dimension."""
    rank = embedding.shape[1]
    fig, axes = plt.subplots(rank, n_top, figsize=(n_top * 1.5, rank * 1.5))

    for d in range(rank):
        top_idx = np.argsort(embedding[:, d])[-n_top:][::-1]
        top_weights = embedding[top_idx, d]

        for i, (idx, w) in enumerate(zip(top_idx, top_weights)):
            ax = axes[d, i]
            img = load_image(filenames[idx])
            ax.imshow(img)
            ax.set_title(f"{w:.2f}", fontsize=7)
            ax.axis("off")

        axes[d, 0].set_ylabel(f"D{d+1}", fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def main():
    log.info("Loading monkey 22k data (F, IT, reliab > 0.3)...")
    result = load_things_monkey(monkey_type="F", roi="it", min_reliab=MIN_RELIAB)
    data = result.data
    filenames = result.metadata["filenames"]
    log.info(f"Data shape: {data.shape}")

    log.info("Computing Gaussian kernel RSM...")
    rsm = gaussian_kernel_similarity(data, data)
    log.info(f"RSM shape: {rsm.shape}")

    log.info(f"Running SRF (rank={RANK}, max_outer=5, max_inner=10)...")
    model = SRF(rank=RANK, max_outer=5, max_inner=10, verbose=1)
    model.fit(rsm)
    embedding = model.w_
    log.info(f"Embedding shape: {embedding.shape}")

    # Save
    np.savez_compressed(
        OUTPUT_DIR / "srf_result.npz",
        embedding=embedding,
        filenames=filenames,
        rsm=rsm,
    )
    log.info(f"Saved: {OUTPUT_DIR / 'srf_result.npz'}")

    # Plot
    fig = plot_dimensions(embedding, filenames, f"Monkey F 22k (rank={RANK})")
    fig.savefig(OUTPUT_DIR / "dimensions.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved: {OUTPUT_DIR / 'dimensions.png'}")

    log.info("Done!")


if __name__ == "__main__":
    main()
