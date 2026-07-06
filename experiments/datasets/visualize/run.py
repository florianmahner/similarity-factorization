"""Visualize top-k items per embedding dimension.

Reads consensus embedding from ../consensus/outputs/ and dataset
labels/images, produces per-dimension image grids and combined overview.

Usage:
    ./scripts/submit experiments/datasets/visualize/run.py dataset=mur92
    ./scripts/submit experiments/datasets/visualize/run.py dataset=peterson_animals k=15
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from datasets import load_dataset
from src.colors import CYCLE
from src.utils.figure_theme import despine, save_figure

log = logging.getLogger(__name__)

CONSENSUS_DIR = Path(__file__).resolve().parent.parent / "consensus" / "outputs"


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


def _plot_topk_text(output_path, embedding, labels, k=10):
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


def _plot_topk_images(output_path, embedding, images, k=10):
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


def _plot_single_dimension_images(output_dir, dim, embedding, images, k=10, dpi=200):
    """SVG output — each top-k image is a separate <image> element (ungroup/extract in Affinity)."""
    is_array = isinstance(images, np.ndarray)
    top_idx = np.argsort(embedding[:, dim])[::-1][:k]
    n_cols = int(np.ceil(np.sqrt(k)))
    n_rows = int(np.ceil(k / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), dpi=dpi,
    )
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(top_idx):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        if is_array:
            ax.imshow(images[idx], interpolation="lanczos")
        else:
            img_path = images[idx]
            if img_path.exists():
                ax.imshow(Image.open(img_path).convert("RGB"), interpolation="lanczos")
            else:
                ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=12)
        ax.axis("off")

    for i in range(k, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=1, bottom=0)
    fig.savefig(
        output_dir / f"dim_{dim + 1:02d}.svg",
        format="svg", bbox_inches="tight", facecolor="none", dpi=dpi,
    )
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


def _plot_individual_dimensions(output_dir, embedding, images, labels, k=10, dpi=200):
    dims_dir = output_dir / "dims"
    dims_dir.mkdir(exist_ok=True)
    for dim in range(embedding.shape[1]):
        if images is not None:
            _plot_single_dimension_images(dims_dir, dim, embedding, images, k=k, dpi=dpi)
        else:
            _plot_single_dimension_text(dims_dir, dim, embedding, labels, k=k)


def run(cfg: DictConfig) -> None:
    subject_id = cfg.get("subject_id")
    k = cfg.get("k", 10)
    ds_name = cfg.dataset.name
    # Override for new naming convention (e.g. "nsd_subj01_sigma0.4"). Falls
    # back to dataset.name + optional /subj{NN} nesting (legacy layout).
    consensus_dir_name = cfg.get("consensus_dir_name") or ds_name

    ds_dir = CONSENSUS_DIR / consensus_dir_name
    if subject_id is not None and (ds_dir / f"subj{subject_id:02d}").exists():
        ds_dir = ds_dir / f"subj{subject_id:02d}"

    embedding_path = ds_dir / "embedding.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding not found: {embedding_path}\n"
            f"Run consensus first: ./scripts/submit experiments/datasets/consensus/run.py dataset={ds_name}"
        )

    embedding = np.load(embedding_path)
    log.info(f"Embedding shape: {embedding.shape}")

    load_kwargs = {"root": cfg.dataset.get("path")}
    if subject_id is not None:
        load_kwargs["subject_id"] = subject_id
    if "loader_kwargs" in cfg.dataset:
        load_kwargs.update(dict(cfg.dataset.loader_kwargs))
    loader_name = cfg.dataset.get("loader_name", cfg.dataset.name)
    dataset = load_dataset(loader_name, **load_kwargs)
    labels = _get_item_labels(dataset, embedding.shape[0])
    images = _get_images(dataset)

    output_dir = Path(__file__).resolve().parent / "outputs" / consensus_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if images is None:
        log.info(f"Plotting top-{k} labels per dimension (no images available)...")
        _plot_topk_text(output_dir / "topk_items.pdf", embedding, labels, k=k)
    else:
        topk_pdf_k = cfg.get("topk_pdf_k", k)
        log.info(f"Plotting combined topk_items.pdf (k={topk_pdf_k})...")
        _plot_topk_images(output_dir / "topk_items.pdf", embedding, images, k=topk_pdf_k)

    dpi = cfg.get("dpi", 200)
    log.info(f"Plotting individual dimensions as SVG (k={k}, dpi={dpi})...")
    _plot_individual_dimensions(output_dir, embedding, images, labels, k=k, dpi=dpi)
    log.info(f"Saved to {output_dir}")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
