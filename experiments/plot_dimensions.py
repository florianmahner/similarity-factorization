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
from src.utils.figure_theme import CMAP, despine, save_figure

log = logging.getLogger(__name__)


def _get_consensus_path(cfg: DictConfig, subject_id: int | None) -> Path:
    """Get path to consensus outputs."""
    consensus_dir = Path(cfg.project_root) / "outputs" / "experiments" / "consensus"
    if subject_id is not None:
        return consensus_dir / cfg.dataset.name / f"subject_{subject_id}"
    return consensus_dir / cfg.dataset.name


def _load_embedding(consensus_path: Path) -> np.ndarray:
    """Load consensus embedding."""
    embedding_path = consensus_path / "embedding.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")
    return np.load(embedding_path)


def _get_item_labels(dataset) -> list[str]:
    """Extract item labels from dataset."""
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if hasattr(dataset, "metadata") and "images" in dataset.metadata:
        return [Path(p).stem for p in dataset.metadata["images"]]
    if hasattr(dataset, "items"):
        return list(dataset.items)
    return [str(i) for i in range(len(dataset.rsm))]


def _get_image_paths(dataset) -> list[Path] | None:
    """Extract image paths from dataset if available."""
    if hasattr(dataset, "metadata") and "images" in dataset.metadata:
        return [Path(p) for p in dataset.metadata["images"]]
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

        ax.barh(range(k), top_values[::-1], color=CMAP[dim % len(CMAP)])
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
    image_paths: list[Path],
    k: int = 10,
    thumbnail_size: int = 64,
) -> None:
    """Plot top-k items per dimension as image grid (dims as rows, items as cols)."""
    n_dims = embedding.shape[1]

    fig, axes = plt.subplots(n_dims, k, figsize=(1.2 * k, 1.5 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    for dim in range(n_dims):
        top_idx = np.argsort(embedding[:, dim])[::-1][:k]

        for col, idx in enumerate(top_idx):
            ax = axes[dim, col]

            img_path = image_paths[idx]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumbnail_size * 2, thumbnail_size * 2))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, img_path.stem, ha="center", va="center", fontsize=6)

            ax.axis("off")

        # Label dimension on the left
        axes[dim, 0].set_ylabel(f"D{dim + 1}", fontsize=10, fontweight="bold", rotation=0, labelpad=20, va="center")

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


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
    dataset = load_dataset(cfg.dataset.name, root=cfg.dataset.get("path"))

    labels = _get_item_labels(dataset)
    image_paths = _get_image_paths(dataset)

    # Plot
    if image_paths is not None:
        log.info(f"Plotting top-{k} images per dimension...")
        _plot_topk_images(output_dir / "topk_images.pdf", embedding, image_paths, k=k)
    else:
        log.info(f"Plotting top-{k} labels per dimension...")
        _plot_topk_text(output_dir / "topk_items.pdf", embedding, labels, k=k)

    log.info(f"Results saved to {output_dir}")
