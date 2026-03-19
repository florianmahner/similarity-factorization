"""SRF consensus embedding for monkey 2k neural data.

Runs SRF with consensus (EnsembleEmbedding + AlignedConsensus) on F and N_concat
recordings to evaluate embedding stability and visualize dimensions.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.pipeline import Pipeline

from pysrf import SRF, AlignedConsensus, EnsembleEmbedding
from src.datasets.monkey import get_output_path, get_things_classes
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
THINGS_IMG_DIR = Path("/SSD/datasets/things/behav1854")

RANK = 10
N_RUNS = 50
N_TOP = 8
RECORDINGS = ["F", "N_concat"]


def load_rsm(recording: str, roi: str = "it") -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed RSM and stimulus names."""
    path = get_output_path(recording, roi)
    data = np.load(path)
    return data["rsm"], data["stimuli"]


def load_image(obj_name: str) -> np.ndarray:
    """Load THINGS image for an object."""
    img_path = THINGS_IMG_DIR / obj_name / f"{obj_name}_01b.jpg"
    if img_path.exists():
        return imread(img_path)
    return np.zeros((100, 100, 3), dtype=np.uint8)


def run_consensus(rsm: np.ndarray, rank: int, n_runs: int) -> dict:
    """Run SRF with consensus and return results."""
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(
            SRF(rank=rank, max_outer=100, tol=1e-5),
            n_runs=n_runs,
            n_jobs=-1,
        )),
        ("consensus", AlignedConsensus(rank=rank, aggregation="select")),
    ])

    embedding = pipeline.fit_transform(rsm)
    consensus = pipeline.named_steps["consensus"]

    return {
        "embedding": embedding,
        "agreement_scores": consensus.agreement_scores_,
        "centrality_scores": consensus.centrality_scores_,
        "selected_run": consensus.selected_run_idx_,
    }


def plot_dimensions(
    embedding: np.ndarray,
    classes: np.ndarray,
    recording: str,
    n_top: int = N_TOP,
) -> plt.Figure:
    """Plot top images per dimension."""
    rank = embedding.shape[1]
    fig, axes = plt.subplots(rank, n_top, figsize=(n_top * 1.2, rank * 1.2))

    for d in range(rank):
        top_idx = np.argsort(embedding[:, d])[-n_top:][::-1]
        top_weights = embedding[top_idx, d]

        for i, (idx, w) in enumerate(zip(top_idx, top_weights)):
            ax = axes[d, i]
            img = load_image(classes[idx])
            ax.imshow(img)
            ax.set_title(f"{w:.2f}", fontsize=6)
            ax.axis("off")

        axes[d, 0].set_ylabel(f"D{d+1}", fontsize=8, rotation=0, ha="right", va="center")

    fig.suptitle(f"{recording} - Top {n_top} per dimension", fontsize=10)
    plt.tight_layout()
    return fig


def plot_stability(results: dict, recording: str) -> plt.Figure:
    """Plot stability metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Agreement scores histogram
    ax = axes[0]
    ax.hist(results["agreement_scores"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(
        results["agreement_scores"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {results['agreement_scores'].mean():.3f}",
    )
    ax.set_xlabel("Agreement score")
    ax.set_ylabel("Count")
    ax.set_title(f"{recording}: Agreement scores")
    ax.legend()

    # Centrality scores
    ax = axes[1]
    ax.hist(results["centrality_scores"], bins=20, edgecolor="black", alpha=0.7)
    selected = results["selected_run"]
    ax.axvline(
        results["centrality_scores"][selected],
        color="green",
        linestyle="--",
        label=f"Selected run {selected}",
    )
    ax.set_xlabel("Centrality score (lower = more central)")
    ax.set_ylabel("Count")
    ax.set_title(f"{recording}: Centrality scores")
    ax.legend()

    plt.tight_layout()
    return fig


def main():
    classes = get_things_classes()
    log.info(f"Loaded {len(classes)} THINGS classes")

    for recording in RECORDINGS:
        log.info(f"\n{'='*50}")
        log.info(f"Processing {recording}")
        log.info(f"{'='*50}")

        rsm, stimuli = load_rsm(recording)
        log.info(f"RSM shape: {rsm.shape}")

        log.info(f"Running SRF consensus (rank={RANK}, n_runs={N_RUNS})...")
        results = run_consensus(rsm, RANK, N_RUNS)

        log.info(f"Selected run: {results['selected_run']}")
        log.info(f"Mean agreement: {results['agreement_scores'].mean():.3f}")
        log.info(f"Agreement range: [{results['agreement_scores'].min():.3f}, {results['agreement_scores'].max():.3f}]")

        # Save results
        np.savez_compressed(
            OUTPUT_DIR / f"{recording}_consensus.npz",
            embedding=results["embedding"],
            agreement_scores=results["agreement_scores"],
            centrality_scores=results["centrality_scores"],
            selected_run=results["selected_run"],
            classes=stimuli,
        )
        log.info(f"Saved: {OUTPUT_DIR / f'{recording}_consensus.npz'}")

        # Plot dimensions
        fig = plot_dimensions(results["embedding"], stimuli, recording)
        fig.savefig(
            OUTPUT_DIR / f"{recording}_dimensions.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        log.info(f"Saved: {OUTPUT_DIR / f'{recording}_dimensions.png'}")

        # Plot stability
        fig = plot_stability(results, recording)
        fig.savefig(
            OUTPUT_DIR / f"{recording}_stability.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        log.info(f"Saved: {OUTPUT_DIR / f'{recording}_stability.png'}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
