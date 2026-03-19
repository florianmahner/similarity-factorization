"""Validate 2k preprocessed data by computing RSM and SRF embedding.

Uses thingsprimate format and z-scores before RSM computation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

from pysrf import SRF
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "things-monkey" / "2k" / "processed" / "final"
THINGS_IMG_DIR = Path("/SSD/datasets/things/behav1854")

RANK = 10
N_TOP = 10
SIGMA_SCALE = 0.5
MIN_RELIAB = 0.3


def load_data(recording: str, roi: str = "it") -> tuple[np.ndarray, np.ndarray, list]:
    """Load data in thingsprimate format."""
    npy_path = DATA_DIR / f"monkey{recording}_{roi}.npy"
    csv_path = DATA_DIR / f"monkey{recording}_{roi}_stiminfo.csv"

    monkey_data = np.load(npy_path, allow_pickle=True).item()
    data = monkey_data["train_MUA"].astype("float32").T  # (n_stim, n_channels)
    reliab = monkey_data["reliab"]

    stiminfo = pd.read_csv(csv_path)
    stims = stiminfo["exemplar"].tolist()

    return data, reliab, stims


def compute_rsm(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute RSM using Gaussian RBF kernel. Data is already z-scored in preprocessing."""
    dist = pairwise_distances(data, metric="euclidean")
    sigma = np.median(dist[np.triu_indices(len(dist), k=1)]) * SIGMA_SCALE
    gamma = 1 / (2 * sigma**2)
    rsm = pairwise_kernels(data, metric="rbf", gamma=gamma)
    return rsm, sigma


def load_image(obj_name: str) -> np.ndarray:
    """Load THINGS image."""
    img_path = THINGS_IMG_DIR / obj_name / f"{obj_name}_01b.jpg"
    if img_path.exists():
        return imread(img_path)
    return np.zeros((100, 100, 3), dtype=np.uint8)


def plot_embedding(W: np.ndarray, stimuli: list, recording: str) -> plt.Figure:
    """Plot top images per dimension."""
    fig, axes = plt.subplots(RANK, N_TOP, figsize=(N_TOP * 1.5, RANK * 1.5))

    for d in range(RANK):
        top_idx = np.argsort(W[:, d])[-N_TOP:][::-1]
        top_weights = W[top_idx, d]

        for i, (idx, w) in enumerate(zip(top_idx, top_weights)):
            ax = axes[d, i]
            img = load_image(stimuli[idx])
            ax.imshow(img)
            ax.set_title(f"{w:.2f}", fontsize=7)
            ax.axis("off")

        axes[d, 0].set_ylabel(f"D{d+1}", fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(f"{recording} - SRF Embedding (k={RANK})", fontsize=11)
    plt.tight_layout()
    return fig


def process_recording(recording: str):
    """Process one recording."""
    print(f"\n{'='*60}")
    print(f"Processing {recording}")
    print(f"{'='*60}")

    # Load data
    data, reliab, stims = load_data(recording)
    print(f"Raw data: {data.shape} (stimuli x channels)")

    # Filter by reliability
    reliab_mask = reliab >= MIN_RELIAB
    data_filtered = data[:, reliab_mask]
    print(f"After reliability filter (>={MIN_RELIAB}): {data_filtered.shape}")

    # Compute RSM (data already z-scored in preprocessing)
    print(f"\nComputing RSM (sigma_scale={SIGMA_SCALE})...")
    rsm, sigma = compute_rsm(data_filtered)
    triu = np.triu_indices(len(rsm), k=1)
    print(f"  Sigma: {sigma:.2f}")
    print(f"  RSM mean: {rsm[triu].mean():.3f}, std: {rsm[triu].std():.3f}")

    # Run SRF
    print(f"\nRunning SRF (rank={RANK})...")
    model = SRF(rank=RANK, max_outer=100, verbose=0, random_state=42)
    model.fit(rsm)
    W = model.components_

    sparsity = (W < 0.01).sum() / W.size
    print(f"  Sparsity (<0.01): {sparsity:.1%}")
    print(f"  Iterations: {model.n_iter_}")

    # Top items per dimension
    print(f"\nTop 5 items per dimension:")
    for d in range(RANK):
        top_idx = np.argsort(W[:, d])[-5:][::-1]
        top_items = [stims[i] for i in top_idx]
        print(f"  D{d+1}: {top_items}")

    # Plot
    fig = plot_embedding(W, stims, recording)
    out_path = OUTPUT_DIR / f"{recording}_embedding.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    for recording in ["F", "N1", "N2"]:
        process_recording(recording)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
