"""
Plot MDS embedding with images.

Visualize the 2D MDS embedding of a similarity matrix by placing item images
at their corresponding coordinates.

Usage:
    ./scripts/submit experiments/plot_mds.py dataset=peterson_animals
    ./scripts/submit experiments/plot_mds.py dataset=mur92
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from omegaconf import DictConfig
from PIL import Image
from sklearn.manifold import MDS

from datasets import load_dataset
from similarity import build_similarity
from src.colors import setup_style

log = logging.getLogger(__name__)


def _rsm_to_distance(rsm: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to distance matrix via min-max normalization."""
    rsm = rsm.copy()
    np.fill_diagonal(rsm, np.nan)

    # Normalize to [0, 1] using min-max scaling
    rmin = np.nanmin(rsm)
    rmax = np.nanmax(rsm)
    rsm_norm = (rsm - rmin) / (rmax - rmin)

    # Set diagonal to max similarity (0 distance)
    np.fill_diagonal(rsm_norm, 1.0)

    # Convert similarity to distance
    return 1 - rsm_norm


def _compute_mds(rsm: np.ndarray) -> np.ndarray:
    """Apply MDS to RSM to get 2D embedding."""
    distance = _rsm_to_distance(rsm)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    return mds.fit_transform(distance)


def _load_image(path: Path, size: int = 40) -> np.ndarray:
    """Load and resize image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img)


def _get_images(dataset) -> list[Path] | np.ndarray | None:
    """Extract images from dataset."""
    if hasattr(dataset, "metadata") and "images" in dataset.metadata:
        imgs = dataset.metadata["images"]
        if isinstance(imgs, np.ndarray):
            return imgs
        return [Path(p) for p in imgs]
    if hasattr(dataset, "image_paths"):
        return [Path(p) for p in dataset.image_paths]
    return None


def _plot_mds_with_images(
    coords: np.ndarray,
    images: list[Path] | np.ndarray,
    output_path: Path,
    title: str | None = None,
    img_size: int = 40,
    figsize: tuple[float, float] = (5, 5),
) -> None:
    """Plot MDS with images at coordinates."""
    setup_style()

    is_paths = isinstance(images, list)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    for i, (x, y) in enumerate(coords):
        if is_paths:
            img = _load_image(images[i], size=img_size)
        else:
            img = images[i]
            if img.shape[0] != img_size:
                pil_img = Image.fromarray(img).resize((img_size, img_size))
                img = np.array(pil_img)

        im = OffsetImage(img, zoom=0.6)
        ab = AnnotationBbox(im, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)

    # Set limits with padding for images
    pad = 0.12
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    ax.set_xlim(coords[:, 0].min() - pad * x_range, coords[:, 0].max() + pad * x_range)
    ax.set_ylim(coords[:, 1].min() - pad * y_range, coords[:, 1].max() + pad * y_range)

    ax.set_xlabel("MDS 1", fontsize=10)
    ax.set_ylabel("MDS 2", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_mds_with_labels(
    coords: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> None:
    """Plot MDS with text labels (fallback when no images)."""
    from src.colors import TEAL, GRAY_DARK

    setup_style()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.7, color=TEAL, edgecolors="white", linewidths=0.5)

    for i, (x, y) in enumerate(coords):
        ax.annotate(labels[i], (x, y), fontsize=6, ha="center", va="bottom", color=GRAY_DARK)

    ax.set_xlabel("MDS 1", fontsize=10)
    ax.set_ylabel("MDS 2", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _get_labels(dataset, n_items: int) -> list[str]:
    """Extract labels from dataset."""
    if hasattr(dataset, "labels") and dataset.labels is not None:
        return list(dataset.labels)
    if hasattr(dataset, "items") and dataset.items is not None:
        return list(dataset.items)
    return [str(i) for i in range(n_items)]


def run(cfg: DictConfig) -> None:
    """Plot MDS embedding with images."""
    subject_id = cfg.get("subject_id")
    output_dir = Path.cwd()
    img_size = cfg.get("img_size", 40)

    # Build similarity matrix
    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    rsm = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"RSM shape: {rsm.shape}")

    # Load dataset for images/labels
    log.info(f"Loading dataset {cfg.dataset.name}...")
    load_kwargs = {"root": cfg.dataset.get("path")}
    if subject_id is not None:
        load_kwargs["subject_id"] = subject_id
    dataset = load_dataset(cfg.dataset.name, **load_kwargs)

    # Compute MDS
    log.info("Computing MDS embedding...")
    coords = _compute_mds(rsm)
    log.info(f"MDS embedding shape: {coords.shape}")

    # Plot
    title = cfg.dataset.name.replace("-", " ").replace("_", " ").title()
    images = _get_images(dataset)
    if images is not None:
        log.info(f"Plotting MDS with {len(images)} images...")
        _plot_mds_with_images(coords, images, output_dir / "mds.png", title=title, img_size=img_size)
    else:
        labels = _get_labels(dataset, rsm.shape[0])
        log.info(f"Plotting MDS with {len(labels)} labels...")
        _plot_mds_with_labels(coords, labels, output_dir / "mds.png", title=title)

    # Save coordinates
    np.save(output_dir / "mds_coords.npy", coords)

    log.info(f"Results saved to {output_dir}")
