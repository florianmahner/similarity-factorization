"""Mockup: Nature-style dimension showcase across datasets."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path("/LOCAL/fmahner/similarity-factorization")
CONSENSUS_DIR = PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs"


def _load_embedding(dataset_name, subject_id=None):
    d = CONSENSUS_DIR / dataset_name
    if subject_id is not None:
        d = d / f"subj{subject_id:02d}"
    return np.load(d / "embedding.npy")


def _load_images(dataset_name):
    from datasets import load_dataset

    roots = {
        "peterson-animals": "/SSD/datasets/similarity_datasets/peterson",
        "peterson-various": "/SSD/datasets/similarity_datasets/peterson",
        "mur92": "/SSD/datasets/similarity_datasets/mur92",
    }

    ds = load_dataset(dataset_name, root=roots[dataset_name])
    labels = list(ds.labels) if hasattr(ds, "labels") else [str(i) for i in range(ds.rsm.shape[0])]
    images = None
    if hasattr(ds, "metadata") and "images" in ds.metadata:
        imgs = ds.metadata["images"]
        images = imgs if isinstance(imgs, np.ndarray) else [Path(p) for p in imgs]
    return labels, images


def _load_things_images():
    meta = pd.read_csv(PROJECT_ROOT / "data" / "features" / "vgg16" / "metadata.csv")
    return list(meta["category"]), [Path(p) for p in meta["path"]]


def _load_img(images, idx):
    if isinstance(images, np.ndarray):
        return Image.fromarray(images[idx])
    p = Path(images[idx])
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def _pick_sparse_dim(embedding, rng):
    sparsity = (embedding == 0).mean(axis=0)
    good = np.where(sparsity > np.median(sparsity))[0]
    if len(good) == 0:
        good = np.arange(embedding.shape[1])
    return rng.choice(good)


def main():
    rng = np.random.default_rng(42)

    datasets = []

    # VGG16
    w = _load_embedding("vgg16")
    labels, images = _load_things_images()
    dim = _pick_sparse_dim(w, rng)
    datasets.append(("VGG-16", w, dim, images, labels))

    # Peterson Animals
    w = _load_embedding("peterson-animals")
    labels, images = _load_images("peterson-animals")
    dim = _pick_sparse_dim(w, rng)
    datasets.append(("Peterson (Animals)", w, dim, images, labels))

    # Peterson Various
    w = _load_embedding("peterson-various")
    labels, images = _load_images("peterson-various")
    dim = _pick_sparse_dim(w, rng)
    datasets.append(("Peterson (Various)", w, dim, images, labels))

    # Mur92
    w = _load_embedding("mur92")
    labels, images = _load_images("mur92")
    dim = _pick_sparse_dim(w, rng)
    datasets.append(("Mur et al.", w, dim, images, labels))

    # 2x2 card layout, each card = 2x3 image grid
    n_cols = 2
    n_rows = 2
    top_k = 6
    img_cols = 3
    img_rows = 2

    fig = plt.figure(figsize=(7.1, 5.0))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue"],
        "font.size": 7,
    })

    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                              wspace=0.25, hspace=0.35,
                              left=0.02, right=0.98, top=0.93, bottom=0.02)

    for i, (name, w, dim, images, labels) in enumerate(datasets):
        row, col = i // n_cols, i % n_cols
        inner = gridspec.GridSpecFromSubplotSpec(
            img_rows, img_cols, subplot_spec=outer[row, col],
            wspace=0.05, hspace=0.15,
        )

        top_idx = np.argsort(w[:, dim])[::-1][:top_k]

        for j in range(top_k):
            r, c = j // img_cols, j % img_cols
            ax = fig.add_subplot(inner[r, c])
            img = _load_img(images, top_idx[j])
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("#CCCCCC")

        card_ax = fig.add_subplot(outer[row, col])
        card_ax.set_axis_off()
        card_ax.set_title(name, fontsize=8, fontweight="bold", pad=8)

    fig.savefig(OUTPUT_DIR / "mockup_cards.pdf",
                bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
