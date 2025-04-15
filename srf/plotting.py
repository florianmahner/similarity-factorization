import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


def plot_matrices(x: Array, m: Array, f: Array):

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

    s = np.corrcoef(x)
    im3 = axes[3].imshow(s, aspect="auto", cmap=cmaps[3])
    axes[3].set_xlabel("Image", fontsize=10)
    axes[3].set_ylabel("Image", fontsize=10)
    axes[3].set_title("Correlation Matrix (s)", fontsize=10)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
