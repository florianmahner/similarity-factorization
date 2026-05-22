"""Plot the saved embeddings from the consensus test.

Run with:
    poetry run python sandbox/nsd_consensus_test/plot.py
"""

from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from h5py import File

from datasets.nsd_utils import NSD_DIR_IRIS
from src.utils.figure_theme import save_figure

# Use the successful run outputs
LATEST = Path(__file__).parent / "outputs" / "260115" / "113230"
print(f"Loading from: {LATEST}")

# Load embeddings
consensus_embedding = np.load(LATEST / "consensus_embedding.npy")
individual_embeddings = np.load(LATEST / "individual_embeddings.npy")

print(f"Consensus embedding: {consensus_embedding.shape}")
print(f"Individual embeddings: {individual_embeddings.shape}")


def plot_topk_images(
    output_path: Path,
    embedding: np.ndarray,
    images: np.ndarray,
    k: int = 8,
) -> None:
    """Plot top-k images per dimension (dims as rows, items as cols)."""
    n_dims = embedding.shape[1]

    fig, axes = plt.subplots(n_dims, k, figsize=(1.2 * k, 1.5 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    for dim in range(n_dims):
        top_idx = np.argsort(embedding[:, dim])[::-1][:k]

        for col, idx in enumerate(top_idx):
            ax = axes[dim, col]
            ax.imshow(images[idx])
            ax.axis("off")

        axes[dim, 0].set_ylabel(
            f"D{dim + 1}",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=20,
            va="center",
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_individual_heatmaps(
    output_path: Path,
    embeddings: np.ndarray,
    n_runs: int,
    n_items: int = 500,
) -> None:
    """Plot heatmaps of individual embeddings (first n_items rows).
    
    embeddings: (n_items, n_runs * rank) - needs reshaping to (n_runs, n_items, rank)
    """
    rank = embeddings.shape[1] // n_runs
    # Reshape: (n_items, n_runs*rank) -> (n_runs, n_items, rank)
    embs_reshaped = embeddings.reshape(embeddings.shape[0], n_runs, rank).transpose(1, 0, 2)

    fig, axes = plt.subplots(1, n_runs, figsize=(3 * n_runs, 6))
    if n_runs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        emb = embs_reshaped[i]
        im = ax.imshow(emb[:n_items], aspect="auto", cmap="viridis")
        ax.set_title(f"Run {i + 1}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Item")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    # Load images directly from HDF5 (much faster than loading betas)
    print("\nGetting stimulus indices for subject 5...")
    experiment = loadmat(
        NSD_DIR_IRIS / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    )
    n_sessions = len(
        glob.glob(
            str(
                NSD_DIR_IRIS
                / "nsddata_betas"
                / "ppdata"
                / "subj05"
                / "func1pt8mm"
                / "betas_fithrf_GLMdenoise_RR"
                / "betas_session*.nii.gz"
            )
        )
    )
    trial_ordering = (
        experiment["subjectim"][4, experiment["masterordering"].squeeze() - 1] - 1
    )
    n_trials = n_sessions * 750
    trials_in_session = trial_ordering[:n_trials]
    stim_ids = np.unique(trials_in_session)

    print(f"  Subject 5 has {len(stim_ids)} stimuli")

    print("Loading images from HDF5...")
    with File(
        NSD_DIR_IRIS / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5", "r"
    ) as f:
        # Load only the images we need
        sorted_idx = np.argsort(stim_ids)
        sorted_stim_ids = stim_ids[sorted_idx]
        images_sorted = f["imgBrick"][sorted_stim_ids]
        # Unsort to match embedding order
        images = images_sorted[np.argsort(sorted_idx)]

    print(f"Loaded {len(images)} images")

    print("\nPlotting...")

    # Consensus embedding top-k images
    print("  - Top-k images per dimension (consensus)...")
    plot_topk_images(
        LATEST / "topk_images_consensus.pdf", consensus_embedding, images, k=8
    )

    # Individual embedding heatmaps
    n_runs = 5
    rank = 10
    print("  - Individual embedding heatmaps...")
    plot_individual_heatmaps(
        LATEST / "individual_embeddings_heatmap.pdf", individual_embeddings, n_runs
    )

    # Reshape for individual run plots: (n_items, n_runs*rank) -> (n_runs, n_items, rank)
    embs_reshaped = individual_embeddings.reshape(-1, n_runs, rank).transpose(1, 0, 2)

    # Top-k for each individual run
    for i in range(n_runs):
        plot_topk_images(
            LATEST / f"topk_images_run{i + 1}.pdf",
            embs_reshaped[i],
            images,
            k=8,
        )
    print(f"  - Top-k images for {n_runs} individual runs")

    print(f"\nSaved to: {LATEST}")


if __name__ == "__main__":
    main()
