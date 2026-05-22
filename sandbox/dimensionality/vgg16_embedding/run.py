"""Quick VGG16 kernel embedding visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.colors import GRAY, INDIGO, ROSE, TEAL, setup_style
from src.utils.figure_theme import despine

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

VGG = ROOT / "experiments/datasets/dimensionality/outputs/cache/vgg16.npy"
THINGS = ROOT / "experiments/datasets/dimensionality/outputs/cache/things_behavior.npy"


def centered_kernel_embedding(k: np.ndarray, n_components: int = 2):
    k = np.asarray(k, dtype=np.float64)
    row_mean = np.nanmean(k, axis=1, keepdims=True)
    col_mean = np.nanmean(k, axis=0, keepdims=True)
    grand_mean = float(np.nanmean(k))
    kc = k - row_mean - col_mean + grand_mean
    vals, vecs = np.linalg.eigh(kc)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[:n_components], 0.0)
    coords = vecs[:, :n_components] * np.sqrt(pos)
    return coords, vals


def cosine_kernel(k: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.maximum(np.diag(k).astype(np.float64), 1e-12))
    return k / np.outer(diag, diag)


def scatter(ax, xy, color, title, colorbar_label=None):
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=color,
        s=10,
        cmap="viridis",
        linewidths=0,
        alpha=0.82,
    )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    despine(ax)
    if colorbar_label:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(colorbar_label)


def main() -> None:
    setup_style()
    vgg = np.asarray(np.load(VGG), dtype=np.float64)
    things = np.asarray(np.load(THINGS), dtype=np.float64)
    norm = np.diag(vgg)
    log_norm = np.log10(norm)

    raw_xy, raw_vals = centered_kernel_embedding(vgg)
    cos_xy, cos_vals = centered_kernel_embedding(cosine_kernel(vgg))
    things_xy, things_vals = centered_kernel_embedding(things)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    scatter(axes[0], raw_xy, log_norm, "VGG16 raw linear kernel", "log10 self-sim")
    scatter(axes[1], cos_xy, log_norm, "VGG16 cosine-normalized kernel", "log10 self-sim")
    scatter(axes[2], things_xy, np.full(things.shape[0], 0.5), "THINGS behavior RSM")
    axes[2].collections[0].set_color(TEAL)

    fig.savefig(OUT / "embedding_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "embedding_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].hist(norm, bins=60, color=INDIGO, alpha=0.85)
    axes[0].set_title("VGG16 self-similarity")
    axes[0].set_xlabel("diagonal K_ii")
    axes[0].set_ylabel("count")
    despine(axes[0])

    for vals, label, color in [
        (raw_vals, "raw linear", ROSE),
        (cos_vals, "cosine", TEAL),
        (things_vals, "THINGS", GRAY),
    ]:
        positive = np.maximum(vals[:100], 0.0)
        frac = positive / positive.sum()
        axes[1].plot(np.arange(1, len(frac) + 1), np.cumsum(frac), label=label, color=color)
    axes[1].set_title("Top-100 cumulative spectral mass")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].legend(frameon=False)
    despine(axes[1])

    fig.savefig(OUT / "norm_and_spectrum.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "norm_and_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT / 'embedding_comparison.png'}")
    print(f"wrote {OUT / 'norm_and_spectrum.png'}")


if __name__ == "__main__":
    main()
