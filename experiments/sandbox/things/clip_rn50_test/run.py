"""VGG16 penultimate layer + linear kernel / max, fit SRF k=11, visualize."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pysrf import SRF

from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

K = 11
TOP_K = 10


def main():
    features = np.load(OUTPUT_DIR.parent / "dev" / "vgg16_features.npy")
    meta_path = Path("/LOCAL/fmahner/similarity-factorization/data/features/clip_rn50/metadata.csv")
    meta = pd.read_csv(meta_path)
    image_paths = [Path(p) for p in meta["path"]]

    similarity = features @ features.T
    similarity = similarity / np.max(similarity)
    print(f"Similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")

    model = SRF(rank=K, random_state=42)
    model.fit(similarity)
    w = model.w_

    recon = w @ w.T
    mask = np.ones_like(similarity, dtype=bool)
    np.fill_diagonal(mask, False)
    r = np.corrcoef(similarity[mask].ravel(), recon[mask].ravel())[0, 1]
    sparsity = (w == 0).mean()
    print(f"Sparsity: {sparsity:.3f}, Reconstruction r: {r:.4f}")

    fig, axes = plt.subplots(K, TOP_K, figsize=(1.2 * TOP_K, 1.4 * K))
    for dim in range(K):
        top_idx = np.argsort(w[:, dim])[::-1][:TOP_K]
        for col, idx in enumerate(top_idx):
            ax = axes[dim, col]
            p = image_paths[idx]
            if p.exists():
                ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[dim, 0].set_ylabel(f"D{dim + 1}", fontsize=9, fontweight="bold",
                                rotation=0, labelpad=15, va="center")

    fig.suptitle(f"VGG16 penultimate + linear/max  (r={r:.3f}, sparsity={sparsity:.2f})",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(OUTPUT_DIR / "vgg16_linear.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
