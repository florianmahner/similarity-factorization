"""
Plotting for NSD Visual Hierarchy analysis.

Visualizes:
1. RSM correlations across visual hierarchy
2. Reconstruction error by ROI
3. Effective dimensionality across hierarchy
4. Example RSMs for early vs late visual areas
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from h5py import File

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.colors import TEAL, ROSE, setup_style
from src.utils.figure_theme import create_figure, save_figure, despine
from src.datasets.nsd_utils import NSD_DIR_IRIS


# ROI ordering for visual hierarchy (early → late)
HIERARCHY_ORDER = [
    "V1", "V2", "V3", "hV4", "VO1", "VO2", "LO1", "LO2",
    "floc-faces", "floc-bodies", "floc-places", "floc-words"
]


def load_results(output_dir: Path) -> dict:
    """Load all results from output directory."""
    results = {
        "roi_results": pd.read_csv(output_dir / "roi_results.csv"),
        "rsm_correlations": pd.read_csv(output_dir / "rsm_correlations.csv", index_col=0),
    }

    # Load embeddings
    emb_dir = output_dir / "embeddings"
    results["embeddings"] = {
        f.stem: np.load(f) for f in emb_dir.glob("*.npy")
    }

    # Load RSMs
    rsm_dir = output_dir / "rsms"
    results["rsms"] = {
        f.stem: np.load(f) for f in rsm_dir.glob("*.npy")
    }

    return results


def order_rois(roi_list: list[str]) -> list[str]:
    """Order ROIs by visual hierarchy."""
    ordered = [r for r in HIERARCHY_ORDER if r in roi_list]
    remaining = [r for r in roi_list if r not in ordered]
    return ordered + remaining


def plot_rsm_correlation_heatmap(rsm_corr: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of RSM correlations between ROIs."""
    setup_style()

    # Reorder by hierarchy
    ordered = order_rois(rsm_corr.columns.tolist())
    rsm_corr = rsm_corr.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(rsm_corr.values, cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RSM correlation (r)", fontsize=8)

    # Add correlation values
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            val = rsm_corr.iloc[i, j]
            color = "white" if val > 0.7 or val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    ax.set_title("RSM similarity across visual hierarchy")
    plt.tight_layout()
    save_figure(fig, output_path, tight=True)


def plot_reconstruction_error(roi_results: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of reconstruction error by ROI."""
    setup_style()

    # Order by hierarchy
    ordered = order_rois(roi_results["roi"].tolist())
    roi_results = roi_results.set_index("roi").loc[ordered].reset_index()

    fig, ax = create_figure("wide", pad_right=0.2)

    # Color early visual vs category-selective differently
    colors = [TEAL if not r.startswith("floc") else ROSE
              for r in roi_results["roi"]]

    ax.bar(range(len(roi_results)), roi_results["recon_rmse"], color=colors, alpha=0.85)

    ax.set_xticks(range(len(roi_results)))
    ax.set_xticklabels(roi_results["roi"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Reconstruction RMSE")
    ax.set_title("SRF reconstruction error by ROI")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAL, alpha=0.85, label="Retinotopic"),
        Patch(facecolor=ROSE, alpha=0.85, label="Category-selective"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    despine(ax)
    save_figure(fig, output_path, tight=True)


def plot_effective_dimensionality(roi_results: pd.DataFrame, output_path: Path) -> None:
    """Plot effective dimensionality across hierarchy."""
    setup_style()

    ordered = order_rois(roi_results["roi"].tolist())
    roi_results = roi_results.set_index("roi").loc[ordered].reset_index()

    fig, ax = create_figure("wide", pad_right=0.2)

    colors = [TEAL if not r.startswith("floc") else ROSE
              for r in roi_results["roi"]]

    ax.bar(range(len(roi_results)), roi_results["effective_dim"], color=colors, alpha=0.85)

    ax.set_xticks(range(len(roi_results)))
    ax.set_xticklabels(roi_results["roi"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Effective dimensionality")
    ax.set_title("Representational complexity by ROI")

    despine(ax)
    save_figure(fig, output_path, tight=True)


def plot_example_rsms(rsms: dict, output_path: Path, n_stimuli: int = 200) -> None:
    """Show example RSMs for early vs late visual areas."""
    setup_style()

    # Pick representative ROIs
    rois_to_show = ["V1", "hV4", "floc-faces", "floc-places"]
    rois_to_show = [r for r in rois_to_show if r in rsms]

    if len(rois_to_show) < 2:
        print("Not enough ROIs to plot RSM examples")
        return

    fig, axes = plt.subplots(1, len(rois_to_show), figsize=(3 * len(rois_to_show), 3))
    if len(rois_to_show) == 1:
        axes = [axes]

    for ax, roi in zip(axes, rois_to_show):
        rsm = rsms[roi][:n_stimuli, :n_stimuli]
        im = ax.imshow(rsm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(roi, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Similarity")
    plt.tight_layout()
    save_figure(fig, output_path, tight=True)


def plot_sparsity_by_roi(roi_results: pd.DataFrame, output_path: Path) -> None:
    """Plot embedding sparsity across ROIs."""
    setup_style()

    ordered = order_rois(roi_results["roi"].tolist())
    roi_results = roi_results.set_index("roi").loc[ordered].reset_index()

    fig, ax = create_figure("wide", pad_right=0.2)

    colors = [TEAL if not r.startswith("floc") else ROSE
              for r in roi_results["roi"]]

    ax.bar(range(len(roi_results)), roi_results["sparsity"], color=colors, alpha=0.85)

    ax.set_xticks(range(len(roi_results)))
    ax.set_xticklabels(roi_results["roi"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sparsity")
    ax.set_title("Embedding sparsity by ROI")

    despine(ax)
    save_figure(fig, output_path, tight=True)


class NSDImageLoader:
    """Lazy loader for NSD stimulus images. Only loads images when indexed."""

    def __init__(self, trials: np.ndarray, nsd_dir: Path = NSD_DIR_IRIS):
        self.trials = trials
        self.hdf5_path = nsd_dir / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = File(self.hdf5_path, "r")["imgBrick"]

    def __getitem__(self, idx: int | list | np.ndarray) -> np.ndarray:
        """Load image(s) by embedding index (not trial index)."""
        self._open()
        if isinstance(idx, (int, np.integer)):
            trial_idx = self.trials[idx]
            return self._file[trial_idx]
        else:
            # Multiple indices
            trial_indices = self.trials[np.array(idx)]
            # HDF5 requires sorted indices for efficient access
            sort_order = np.argsort(trial_indices)
            sorted_trials = trial_indices[sort_order]
            images = self._file[sorted_trials.tolist()]
            # Restore original order
            return images[np.argsort(sort_order)]

    def __len__(self):
        return len(self.trials)

    def close(self):
        if self._file is not None:
            self._file.file.close()
            self._file = None


def load_trials_from_cache(output_dir: Path) -> np.ndarray:
    """Load trial indices from cache file."""
    trials_file = output_dir / "trials.npy"
    if trials_file.exists():
        return np.load(trials_file)
    raise FileNotFoundError(f"Trials not found: {trials_file}. Re-run run.py to generate.")


def get_nsd_trials_fast(subject_id: int, n_stimuli: int | None = None) -> np.ndarray:
    """Get NSD trial indices from experiment design (fast, no betas loading).

    Returns stimulus IDs in the order they first appeared in the experiment.
    """
    from scipy.io import loadmat

    experiment_design = loadmat(
        NSD_DIR_IRIS / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    )

    trial_ordering = (
        experiment_design["subjectim"][
            subject_id - 1, experiment_design["masterordering"].squeeze() - 1
        ]
        - 1
    )

    # Get unique trials in order of first appearance
    seen = set()
    trials = []
    for trial in trial_ordering:
        if trial not in seen:
            seen.add(trial)
            trials.append(trial)

    trials = np.array(trials)
    if n_stimuli is not None:
        trials = trials[:n_stimuli]

    return trials


def plot_top_images_per_dimension(
    embedding: np.ndarray,
    image_loader: NSDImageLoader,
    roi_name: str,
    output_path: Path,
    top_k: int = 8,
) -> None:
    """Plot top-k images for each dimension of an embedding.

    Args:
        embedding: (n_stimuli, rank) array
        image_loader: Lazy image loader
        roi_name: Name of the ROI for title
        output_path: Path to save figure
        top_k: Number of top images per dimension
    """
    n_stimuli, rank = embedding.shape

    fig, axes = plt.subplots(rank, top_k, figsize=(top_k * 1.2, rank * 1.2))

    for dim in range(rank):
        # Get top-k indices for this dimension
        loadings = embedding[:, dim]
        top_indices = np.argsort(loadings)[::-1][:top_k]

        # Load only the images we need for this dimension
        images = image_loader[top_indices]

        for k, img in enumerate(images):
            ax = axes[dim, k] if rank > 1 else axes[k]
            ax.imshow(img)
            ax.axis("off")

            if k == 0:
                ax.set_ylabel(f"Dim {dim+1}", fontsize=8, rotation=0, ha="right", va="center")

    fig.suptitle(f"{roi_name}: Top {top_k} images per dimension", fontsize=10, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path, tight=True)
    plt.close(fig)


def plot_all_roi_top_images(
    embeddings: dict[str, np.ndarray],
    image_loader: NSDImageLoader,
    output_dir: Path,
    top_k: int = 8,
) -> None:
    """Plot top-k images for all ROIs."""
    img_dir = output_dir / "top_images"
    img_dir.mkdir(exist_ok=True)

    for roi_name, embedding in embeddings.items():
        print(f"  Plotting {roi_name}...")
        plot_top_images_per_dimension(
            embedding, image_loader, roi_name,
            img_dir / f"{roi_name}_top_images.pdf",
            top_k=top_k,
        )


def main(metric: str = "gaussian"):
    output_dir = Path(__file__).parent / "outputs" / metric

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        print("Run run.py first to generate results.")
        return

    print("Loading results...")
    results = load_results(output_dir)

    print("Generating plots...")

    # 1. RSM correlation heatmap
    plot_rsm_correlation_heatmap(
        results["rsm_correlations"],
        output_dir / "rsm_correlation_heatmap.pdf"
    )
    print("  - rsm_correlation_heatmap.pdf")

    # 2. Reconstruction error
    plot_reconstruction_error(
        results["roi_results"],
        output_dir / "reconstruction_error.pdf"
    )
    print("  - reconstruction_error.pdf")

    # 3. Effective dimensionality
    plot_effective_dimensionality(
        results["roi_results"],
        output_dir / "effective_dimensionality.pdf"
    )
    print("  - effective_dimensionality.pdf")

    # 4. Example RSMs
    plot_example_rsms(
        results["rsms"],
        output_dir / "example_rsms.pdf"
    )
    print("  - example_rsms.pdf")

    # 5. Sparsity
    plot_sparsity_by_roi(
        results["roi_results"],
        output_dir / "sparsity.pdf"
    )
    print("  - sparsity.pdf")

    # 6. Top images per dimension for each ROI
    print("\nGenerating top images per dimension (lazy loading)...")
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    try:
        trials = load_trials_from_cache(output_dir)
    except FileNotFoundError:
        print("  trials.npy not found, computing from experiment design...")
        trials = get_nsd_trials_fast(metadata["subject_id"], metadata.get("n_stimuli_limit"))
        np.save(output_dir / "trials.npy", trials)

    image_loader = NSDImageLoader(trials)

    plot_all_roi_top_images(
        results["embeddings"],
        image_loader,
        output_dir,
        top_k=8,
    )
    print("  - top_images/*.pdf")

    image_loader.close()

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="gaussian",
                        choices=["gaussian", "shifted_cosine"],
                        help="Similarity metric")
    args = parser.parse_args()
    main(metric=args.metric)
