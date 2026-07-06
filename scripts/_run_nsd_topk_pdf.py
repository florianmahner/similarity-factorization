"""One-shot follow-up: combined topk_items.pdf (78 dims x k cols) for NSD."""
from __future__ import annotations
from pathlib import Path
import time

import numpy as np

from datasets import load_dataset
from experiments.datasets.visualize.run import _plot_topk_images

EMBEDDING = Path("experiments/datasets/consensus/outputs/nsd_subj01_sigma0.4/embedding.npy")
OUT_PDF = Path("experiments/datasets/visualize/outputs/nsd_subj01_sigma0.4/topk_items.pdf")
NSD_ROOT = Path("/data/labshare/_stachelschwein/LOCAL/LABSHARE/natural-scenes-dataset")
K = 20


def main() -> None:
    t0 = time.time()
    print(f"=== loading embedding ===", flush=True)
    embedding = np.load(EMBEDDING)
    print(f"  shape: {embedding.shape}", flush=True)

    print("=== loading NSD subj01 images ===", flush=True)
    dataset = load_dataset(
        "nsd",
        root=str(NSD_ROOT),
        subject_id=1,
        roi_name="streams",
        space="func1pt8mm",
        zscore_betas=True,
    )
    images = dataset.metadata["images"]
    n_imgs = len(images) if isinstance(images, list) else images.shape[0]
    print(f"  images: {n_imgs}", flush=True)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== writing {OUT_PDF} ({embedding.shape[1]} dims x {K} imgs) ===", flush=True)
    _plot_topk_images(OUT_PDF, embedding, images, k=K)
    print(f"=== done in {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
