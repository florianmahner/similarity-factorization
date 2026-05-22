"""Datasets overview figure: interpretable dimensions across diverse data types.

Layout (180mm, 2 rows):
  Row 1 (a): Example dimension per dataset -- 4 image cards side by side
  Row 2 (b): Dimensionality (k*) bar chart  |  (c): Sparsity per dataset

Usage:
    poetry run python experiments/figures/plot_datasets/plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np
import pandas as pd
from PIL import Image

from src.colors import ROSE, TEAL, INDIGO, GRAY, GRAY_LIGHT, GRAY_DARK

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSENSUS_DIR = PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs"
KAPPA_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ranks" / "kappa" / "outputs"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
ROW_HEIGHT_MM = 35
ROW_HEIGHT_IN = ROW_HEIGHT_MM / 25.4
FONT_SIZE = 5


def _nature_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


# ── Data loading ──────────────────────────────────────────────────────────

def _load_embedding(name, subject_id=None):
    d = CONSENSUS_DIR / name
    if subject_id is not None:
        d = d / f"subj{subject_id:02d}"
    return np.load(d / "embedding.npy")


def _load_summary(name, subject_id=None):
    d = CONSENSUS_DIR / name
    if subject_id is not None:
        d = d / f"subj{subject_id:02d}"
    return json.loads((d / "summary.json").read_text())


def _load_kappa(name):
    path = KAPPA_DIR / f"{name}.json"
    return json.loads(path.read_text())


def _load_images_neural(dataset_name):
    from datasets import load_dataset
    roots = {
        "peterson-animals": "/SSD/datasets/similarity_datasets/peterson",
        "peterson-various": "/SSD/datasets/similarity_datasets/peterson",
        "mur92": "/SSD/datasets/similarity_datasets/mur92",
    }
    ds = load_dataset(dataset_name, root=roots[dataset_name])
    images = None
    if hasattr(ds, "metadata") and "images" in ds.metadata:
        imgs = ds.metadata["images"]
        images = imgs if isinstance(imgs, np.ndarray) else [Path(p) for p in imgs]
    return images


def _load_images_vgg16():
    meta = pd.read_csv(PROJECT_ROOT / "data" / "features" / "vgg16" / "metadata.csv")
    return [Path(p) for p in meta["path"]]


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


# ── Panel functions ───────────────────────────────────────────────────────

DATASETS_WITH_IMAGES = [
    ("Peterson\n(Animals)", "peterson-animals", "peterson_animals"),
    ("Peterson\n(Various)", "peterson-various", "peterson_various"),
    ("Mur et al.", "mur92", "mur92"),
    ("VGG-16", "vgg16", "vgg16"),
]


def panel_dimensions(fig, gs_slot, rng):
    """Row 1: one example dimension per dataset as image cards."""
    n_datasets = len(DATASETS_WITH_IMAGES)
    img_rows, img_cols = 2, 3
    top_k = img_rows * img_cols

    inner = gs_slot.subgridspec(1, n_datasets, wspace=0.3)

    for di, (label, consensus_name, kappa_name) in enumerate(DATASETS_WITH_IMAGES):
        w = _load_embedding(consensus_name)
        if consensus_name == "vgg16":
            images = _load_images_vgg16()
        else:
            images = _load_images_neural(consensus_name)

        dim = _pick_sparse_dim(w, rng)
        top_idx = np.argsort(w[:, dim])[::-1][:top_k]

        card = inner[di].subgridspec(img_rows, img_cols, wspace=0.04, hspace=0.04)

        for j in range(top_k):
            r, c = j // img_cols, j % img_cols
            ax = fig.add_subplot(card[r, c])
            img = _load_img(images, top_idx[j])
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color(GRAY_LIGHT)

        # Title above card
        title_ax = fig.add_subplot(inner[di])
        title_ax.set_axis_off()
        title_ax.set_title(label, fontsize=FONT_SIZE + 1, fontweight="bold",
                           pad=4, linespacing=1.1)


ALL_DATASETS = [
    ("Peterson\n(Animals)", "peterson_animals"),
    ("Peterson\n(Various)", "peterson_various"),
    ("Mur\net al.", "mur92"),
    ("VGG-16", "vgg16"),
]


def panel_dimensionality(ax):
    """Panel b: bar chart of estimated k* across datasets."""
    names = []
    k_stars = []
    for label, kappa_name in ALL_DATASETS:
        path = KAPPA_DIR / f"{kappa_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            names.append(label)
            k_stars.append(data["k_star"])

    x = np.arange(len(names))
    bars = ax.bar(x, k_stars, color=ROSE, width=0.6, edgecolor="white", linewidth=0.3)

    for bar, k in zip(bars, k_stars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(k), ha="center", va="bottom", fontsize=FONT_SIZE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FONT_SIZE, linespacing=0.9)
    ax.set_ylabel("Estimated rank ($k^*$)")
    ax.set_ylim(0, max(k_stars) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_sparsity(ax):
    """Panel c: sparsity per dataset."""
    names = []
    sparsities = []
    for label, kappa_name in ALL_DATASETS:
        consensus_name = kappa_name.replace("_", "-")
        path = CONSENSUS_DIR / consensus_name / "summary.json"
        if not path.exists():
            path = CONSENSUS_DIR / kappa_name / "summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            names.append(label)
            sparsities.append(data["sparsity"])

    x = np.arange(len(names))
    bars = ax.bar(x, sparsities, color=TEAL, width=0.6, edgecolor="white", linewidth=0.3)

    for bar, s in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{s:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FONT_SIZE, linespacing=0.9)
    ax.set_ylabel("Sparsity")
    ax.set_ylim(0, min(1.0, max(sparsities) * 1.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_nature_rc())

    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * 2.5))

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.4, 1.0],
        hspace=0.45,
        left=0.06, right=0.98, top=0.93, bottom=0.08,
    )

    # Row 1: dimension image cards
    panel_dimensions(fig, outer[0], rng)

    # Row 2: dimensionality + sparsity
    bottom = outer[1].subgridspec(1, 2, wspace=0.4)
    ax_k = fig.add_subplot(bottom[0])
    ax_s = fig.add_subplot(bottom[1])
    panel_dimensionality(ax_k)
    panel_sparsity(ax_s)

    # Panel labels
    fig.text(0.01, 0.96, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.01, 0.42, "b", fontsize=8, fontweight="bold", va="top")
    fig.text(0.52, 0.42, "c", fontsize=8, fontweight="bold", va="top")

    fig.savefig(OUTPUT / "datasets.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT / 'datasets.pdf'}")


if __name__ == "__main__":
    main()
