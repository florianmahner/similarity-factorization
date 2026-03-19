"""
Plot top items per embedding dimension.

Visualize what each dimension of a consensus embedding represents by showing
the top-k items (images or labels) with highest loading on each dimension.

Usage:
    ./scripts/submit experiments/plot_dimensions.py dataset=mur92
    ./scripts/submit experiments/plot_dimensions.py dataset=things_behavior k=15
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


def _get_consensus_path(cfg: DictConfig, subject_id: int | None) -> Path:
    """Get path to consensus outputs."""
    consensus_dir = Path(cfg.project_root) / "outputs" / "experiments" / "consensus" / cfg.dataset.name
    if subject_id is None:
        return consensus_dir
    primary = consensus_dir / f"subj{subject_id:02d}"
    fallback = consensus_dir / f"subject_{subject_id}"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def _load_embedding(consensus_path: Path) -> np.ndarray:
    """Load consensus embedding."""
    embedding_path = consensus_path / "embedding.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")
    return np.load(embedding_path)


def _get_item_labels(dataset, n_items: int) -> list[str]:
    """Extract item labels from dataset."""
    if hasattr(dataset, "labels") and dataset.labels is not None:
        return list(dataset.labels)
    if hasattr(dataset, "items") and dataset.items is not None:
        return list(dataset.items)
    return [str(i) for i in range(n_items)]


def _get_images(dataset) -> np.ndarray | list[Path] | None:
    """Extract images from dataset (as array or paths)."""
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
    """Plot top-k items per dimension as text."""
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
    """Plot top-k items per dimension as image grid (dims as rows, items as cols)."""
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
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, img_path.stem, ha="center", va="center", fontsize=6)

            ax.axis("off")

        axes[dim, 0].set_ylabel(f"D{dim + 1}", fontsize=10, fontweight="bold", rotation=0, labelpad=20, va="center")

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_single_dimension_images(
    output_dir: Path,
    dim: int,
    embedding: np.ndarray,
    images: np.ndarray | list[Path],
    labels: list[str] | None = None,
    k: int = 10,
) -> None:
    """Plot a single dimension as a square grid of top-k images."""
    is_array = isinstance(images, np.ndarray)
    top_idx = np.argsort(embedding[:, dim])[::-1][:k]

    # Square grid
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
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=12)
        ax.axis("off")

    # Hide unused axes
    for i in range(k, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    fig.suptitle(f"Dimension {dim + 1}", fontsize=12, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(output_dir / f"dim_{dim + 1:02d}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_single_dimension_text(
    output_dir: Path,
    dim: int,
    embedding: np.ndarray,
    labels: list[str],
    k: int = 10,
) -> None:
    """Plot a single dimension as horizontal bar chart."""
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


def _plot_individual_dimensions(
    output_dir: Path,
    embedding: np.ndarray,
    images: np.ndarray | list[Path] | None,
    labels: list[str],
    k: int = 10,
) -> None:
    """Plot each dimension separately in a dims/ subfolder."""
    dims_dir = output_dir / "dims"
    dims_dir.mkdir(exist_ok=True)

    n_dims = embedding.shape[1]
    for dim in range(n_dims):
        if images is not None:
            _plot_single_dimension_images(dims_dir, dim, embedding, images, k=k)
        else:
            _plot_single_dimension_text(dims_dir, dim, embedding, labels, k=k)


def run(cfg: DictConfig) -> None:
    """Plot top items per embedding dimension."""
    subject_id = cfg.get("subject_id")
    k = cfg.get("k", 10)
    output_dir = Path.cwd()

    # Load consensus embedding
    consensus_path = _get_consensus_path(cfg, subject_id)
    log.info(f"Loading embedding from {consensus_path}")
    embedding = _load_embedding(consensus_path)
    log.info(f"Embedding shape: {embedding.shape}")

    # Load dataset for labels/images
    log.info(f"Loading dataset {cfg.dataset.name}...")
    load_kwargs = {"root": cfg.dataset.get("path")}
    if subject_id is not None:
        load_kwargs["subject_id"] = subject_id
    dataset = load_dataset(cfg.dataset.name, **load_kwargs)

    n_items = embedding.shape[0]
    labels = _get_item_labels(dataset, n_items)
    images = _get_images(dataset)

    # Plot combined view
    if images is not None:
        log.info(f"Plotting top-{k} images per dimension (combined)...")
        _plot_topk_images(output_dir / "topk_images.pdf", embedding, images, k=k)
    else:
        log.info(f"Plotting top-{k} labels per dimension (combined)...")
        _plot_topk_text(output_dir / "topk_items.pdf", embedding, labels, k=k)

    # Plot individual dimensions
    log.info(f"Plotting individual dimensions to dims/...")
    _plot_individual_dimensions(output_dir, embedding, images, labels, k=k)

    log.info(f"Results saved to {output_dir}")
