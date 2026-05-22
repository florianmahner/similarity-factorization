"""Mur92: compare SRF at k=2 (raw kappa) vs k=10 (centered kappa).

Visualize all dimensions to check if dims 3-10 are signal or noise.
Also compute reconstruction R^2 for both ranks.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pysrf import SRF
from pysrf.consensus import EnsembleEmbedding, AlignedConsensus
from sklearn.pipeline import Pipeline

from PIL import Image

from datasets import load_dataset
from similarity import build_similarity
from omegaconf import OmegaConf
from src.colors import ROSE, TEAL, INDIGO, GRAY_LIGHT, CYCLE
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_mur92():
    cfg_path = PROJECT_ROOT / "configs" / "dataset" / "mur92.yaml"
    raw = OmegaConf.load(cfg_path)
    parent = OmegaConf.create({
        "paths": {"data_dir": str(PROJECT_ROOT / "data")},
        "dataset": raw,
    })
    OmegaConf.resolve(parent)
    return build_similarity(parent.dataset)


def reconstruction_r2(s, w):
    s_hat = w @ w.T
    mask = ~np.eye(s.shape[0], dtype=bool)
    ss_res = np.sum((s[mask] - s_hat[mask]) ** 2)
    ss_tot = np.sum((s[mask] - s[mask].mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    s = load_mur92()
    n = s.shape[0]
    log.info(f"Mur92: {s.shape}")

    ranks = [2, 5, 10, 15, 20]
    n_runs = 20

    results = {}
    for k in ranks:
        log.info(f"Running consensus at k={k}...")
        pipe = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=k), n_runs=n_runs, n_jobs=-1)),
            ("consensus", AlignedConsensus(rank=k, aggregation="select")),
        ])
        w = pipe.fit_transform(s)
        r2 = reconstruction_r2(s, w)
        results[k] = {"w": w, "r2": r2}
        log.info(f"  k={k}: R^2={r2:.4f}")

    # --- Plot 1: R^2 vs rank ---
    fig, ax = create_figure("single")
    ks = sorted(results.keys())
    r2s = [results[k]["r2"] for k in ks]
    ax.plot(ks, r2s, "o-", color=ROSE, markersize=4, linewidth=0.8)
    ax.axvline(2, color=TEAL, linestyle="--", linewidth=0.7, label="kappa (raw) k*=2")
    ax.axvline(10, color=INDIGO, linestyle="--", linewidth=0.7, label="kappa (centered) k*=10")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Reconstruction $R^2$")
    ax.legend(fontsize=6)
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "r2_vs_rank.png")
    plt.close()

    # --- Plot 2: dimension profiles for k=10 ---
    w10 = results[10]["w"]
    fig, axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True)
    for d in range(10):
        ax = axes[d // 5, d % 5]
        vals = w10[:, d]
        order = np.argsort(-vals)
        ax.bar(range(n), vals[order], color=ROSE if d < 2 else GRAY_LIGHT, width=1.0,
               edgecolor="none")
        ax.set_title(f"Dim {d+1}", fontsize=8)
        ax.set_yticks([])
        if d // 5 == 1:
            ax.set_xlabel("Objects (sorted)", fontsize=7)
        despine(ax)
    fig.suptitle("Mur92 k=10: dimension profiles (sorted by loading)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dims_k10.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # --- Plot 3: dimension profiles for k=2 ---
    w2 = results[2]["w"]
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for d in range(2):
        ax = axes[d]
        vals = w2[:, d]
        order = np.argsort(-vals)
        ax.bar(range(n), vals[order], color=TEAL, width=1.0, edgecolor="none")
        ax.set_title(f"Dim {d+1}", fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel("Objects (sorted)", fontsize=7)
        despine(ax)
    fig.suptitle("Mur92 k=2: dimension profiles", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dims_k2.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # --- Plot 4: RSM comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    vmin, vmax = s[~np.eye(n, dtype=bool)].min(), s[~np.eye(n, dtype=bool)].max()

    axes[0].imshow(s, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Original RSM", fontsize=8)

    for i, k in enumerate([2, 10]):
        w = results[k]["w"]
        s_hat = w @ w.T
        axes[i + 1].imshow(s_hat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i + 1].set_title(f"SRF k={k} ($R^2$={results[k]['r2']:.3f})", fontsize=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rsm_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # --- Plot 5: top-k images per dimension (k=10) ---
    log.info("Loading Mur92 images...")
    ds = load_dataset("mur92", root="/SSD/datasets/similarity_datasets/mur92")
    image_paths = [Path(p) for p in ds.metadata["images"]]

    top_k = 8
    for rank_val in [2, 10]:
        w = results[rank_val]["w"]
        n_dims = w.shape[1]
        fig, axes = plt.subplots(n_dims, top_k, figsize=(1.2 * top_k, 1.4 * n_dims))
        if n_dims == 1:
            axes = axes.reshape(1, -1)

        for dim in range(n_dims):
            top_idx = np.argsort(w[:, dim])[::-1][:top_k]
            for col, idx in enumerate(top_idx):
                ax = axes[dim, col]
                img_path = image_paths[idx]
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                ax.axis("off")
                if dim == 0:
                    ax.set_title(f"{col+1}", fontsize=7, color="gray")
            axes[dim, 0].set_ylabel(
                f"D{dim+1}", fontsize=9, fontweight="bold",
                rotation=0, labelpad=18, va="center",
            )

        fig.suptitle(f"Mur92 k={rank_val}: top-{top_k} images per dimension", fontsize=10, y=1.01)
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        fig.savefig(
            OUTPUT_DIR / f"topk_images_k{rank_val}.png",
            dpi=200, bbox_inches="tight", facecolor="white",
        )
        plt.close()
        log.info(f"  Saved topk_images_k{rank_val}.png")

    log.info(f"\nSaved to {OUTPUT_DIR}")
    for k in ks:
        log.info(f"  k={k}: R^2={results[k]['r2']:.4f}")


if __name__ == "__main__":
    main()
