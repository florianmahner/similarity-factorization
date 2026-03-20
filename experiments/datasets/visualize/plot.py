"""Visualize top-k items per embedding dimension.

Reads consensus embedding from ../consensus/outputs/ and dataset
labels/images, produces per-dimension image grids and combined overview.

Usage:
    ./scripts/submit experiments/datasets/visualize/plot.py -- --dataset mur92
    ./scripts/submit experiments/datasets/visualize/plot.py -- --dataset things_behavior --k 15
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets import load_dataset
from src.colors import CYCLE
from src.utils.figure_theme import despine, save_figure

log = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
CONSENSUS_DIR = TASK_DIR.parent / "consensus" / "outputs"
OUTPUT_DIR = TASK_DIR / "outputs"


def _get_item_labels(dataset, n_items: int) -> list[str]:
    if hasattr(dataset, "labels") and dataset.labels is not None:
        return list(dataset.labels)
    if hasattr(dataset, "items") and dataset.items is not None:
        return list(dataset.items)
    return [str(i) for i in range(n_items)]


def _get_images(dataset) -> np.ndarray | list[Path] | None:
    if hasattr(dataset, "metadata") and "images" in dataset.metadata:
        imgs = dataset.metadata["images"]
        if isinstance(imgs, np.ndarray):
            return imgs
        return [Path(p) for p in imgs]
    if hasattr(dataset, "image_paths"):
        return [Path(p) for p in dataset.image_paths]
    return None


def _plot_topk_text(
    output_path: Path,
    embedding: np.ndarray,
    labels: list[str],
    k: int = 10,
) -> None:
    n_dims = embedding.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(3 * n_dims, 6))
    if n_dims == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        top_idx = np.argsort(embedding[:, dim])[::-1][:k]
        top_labels = [labels[i] for i in top_idx]
        top_values = embedding[top_idx, dim]

        ax.barh(range(k), top_values[::-1], color=CYCLE[dim % len(CYCLE)])
        ax.set_yticks(range(k))
        ax.set_yticklabels(top_labels[::-1], fontsize=8)
        ax.set_xlabel("Loading")
        ax.set_title(f"Dimension {dim + 1}")
        despine(ax)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_topk_images(
    output_path: Path,
    embedding: np.ndarray,
    images: np.ndarray | list[Path],
    k: int = 10,
) -> None:
    n_dims = embedding.shape[1]
    is_array = isinstance(images, np.ndarray)

    fig, axes = plt.subplots(n_dims, k, figsize=(1.2 * k, 1.5 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    for dim in range(n_dims):
        top_idx = np.argsort(embedding[:, dim])[::-1][:k]
        for col, idx in enumerate(top_idx):
            ax = axes[dim, col]
            if is_array:
                ax.imshow(images[idx])
            else:
                img_path = images[idx]
                if img_path.exists():
                    ax.imshow(Image.open(img_path).convert("RGB"))
                else:
                    ax.text(0.5, 0.5, img_path.stem, ha="center", va="center", fontsize=6)
            ax.axis("off")
        axes[dim, 0].set_ylabel(f"D{dim + 1}", fontsize=10, fontweight="bold",
                                rotation=0, labelpad=20, va="center")

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_single_dimension_images(output_dir, dim, embedding, images, k=10):
    is_array = isinstance(images, np.ndarray)
    top_idx = np.argsort(embedding[:, dim])[::-1][:k]
    n_cols = int(np.ceil(np.sqrt(k)))
    n_rows = int(np.ceil(k / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(top_idx):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        if is_array:
            ax.imshow(images[idx])
        else:
            img_path = images[idx]
            if img_path.exists():
                ax.imshow(Image.open(img_path).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=12)
        ax.axis("off")

    for i in range(k, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    fig.suptitle(f"Dimension {dim + 1}", fontsize=12, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(output_dir / f"dim_{dim + 1:02d}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_single_dimension_text(output_dir, dim, embedding, labels, k=10):
    top_idx = np.argsort(embedding[:, dim])[::-1][:k]
    top_labels = [labels[i] for i in top_idx]
    top_values = embedding[top_idx, dim]

    fig, ax = plt.subplots(figsize=(6, 0.4 * k + 1))
    ax.barh(range(k), top_values[::-1], color=CYCLE[dim % len(CYCLE)])
    ax.set_yticks(range(k))
    ax.set_yticklabels(top_labels[::-1], fontsize=10)
    ax.set_xlabel("Loading")
    ax.set_title(f"Dimension {dim + 1}", fontsize=12, fontweight="bold")
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / f"dim_{dim + 1:02d}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_individual_dimensions(output_dir, embedding, images, labels, k=10):
    dims_dir = output_dir / "dims"
    dims_dir.mkdir(exist_ok=True)

    for dim in range(embedding.shape[1]):
        if images is not None:
            _plot_single_dimension_images(dims_dir, dim, embedding, images, k=k)
        else:
            _plot_single_dimension_text(dims_dir, dim, embedding, labels, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., mur92, things_behavior)")
    parser.add_argument("--k", type=int, default=10, help="Top-k items per dimension")
    parser.add_argument("--subject-id", type=int, default=None)
    args = parser.parse_args()

    ds_dir = CONSENSUS_DIR / args.dataset
    if args.subject_id:
        ds_dir = ds_dir / f"subj{args.subject_id:02d}"

    embedding_path = ds_dir / "embedding.npy"
    if not embedding_path.exists():
        log.error(f"Embedding not found: {embedding_path}")
        raise SystemExit(1)

    embedding = np.load(embedding_path)
    log.info(f"Embedding shape: {embedding.shape}")

    dataset = load_dataset(args.dataset)
    labels = _get_item_labels(dataset, embedding.shape[0])
    images = _get_images(dataset)

    out = OUTPUT_DIR / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    if images is not None:
        log.info(f"Plotting top-{args.k} images per dimension...")
        _plot_topk_images(out / "topk_images.pdf", embedding, images, k=args.k)
    else:
        log.info(f"Plotting top-{args.k} labels per dimension...")
        _plot_topk_text(out / "topk_items.pdf", embedding, labels, k=args.k)

    log.info("Plotting individual dimensions...")
    _plot_individual_dimensions(out, embedding, images, labels, k=args.k)
    log.info(f"Saved to {out}")
