import matplotlib.pyplot as plt
from .models.trifactor import TrifactorCD
import skimage.io as sio
import numpy as np


Array = np.ndarray


def plot_simulation(x: Array, m: Array, f: Array, s: Array):

    # get seaborn color palette
    cmaps = ["Blues", "Greens", "Purples", "Oranges"]

    fig, axes = plt.subplots(1, 4, figsize=(5.0 * 4, 3.0))
    im0 = axes[0].imshow(m, aspect="auto", cmap=cmaps[0])
    axes[0].set_xlabel("Component", fontsize=10)
    axes[0].set_ylabel("Image", fontsize=10)
    axes[0].set_xticks(np.arange(m.shape[1]))
    axes[0].set_xticklabels(np.arange(1, m.shape[1] + 1), fontsize=10)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].set_title("Membership Matrix (m)", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(f, aspect="auto", cmap=cmaps[1])
    axes[1].set_ylabel("Component", fontsize=10)
    axes[1].set_xlabel("Feature", fontsize=10)
    axes[1].set_title("Factor Matrix (f)", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(x, aspect="auto", cmap=cmaps[2])
    axes[2].set_ylabel("Image", fontsize=10)
    axes[2].set_xlabel("Feature", fontsize=10)
    axes[2].set_title("Data Matrix (x)", fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(s, aspect="auto", cmap=cmaps[3])
    axes[3].set_xlabel("Image", fontsize=10)
    axes[3].set_ylabel("Image", fontsize=10)
    axes[3].set_title("Correlation Matrix (s)", fontsize=10)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_images_from_embedding(w, images, top_k=12, dpi=300):
    plt.ioff()
    fig, axs = plt.subplots(w.shape[1], top_k, figsize=(top_k, w.shape[1]), dpi=dpi)
    for rank in range(w.shape[1]):
        top_k_images = np.argsort(-w[:, rank])[:top_k]
        for col, img_idx in enumerate(top_k_images):
            ax = axs[rank, col]
            # load the image path from the original images list
            img_path = images[img_idx]

            if isinstance(img_path, str):
                img = sio.imread(img_path)
            else:
                img = img_path
            ax.imshow(img)
            ax.axis("off")
    return fig


def plot_tri_factors(model: TrifactorCD):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].hist(model.w_.flatten(), bins=50)
    ax[0].set_title("w")
    ax[1].hist(model.h_.flatten(), bins=50)
    ax[1].set_title("h")
    ax[2].hist(model.a_.flatten(), bins=50)
    ax[2].set_title("a")
    return fig
