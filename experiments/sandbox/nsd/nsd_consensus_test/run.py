"""Quick test: NSD subject 5 consensus embedding with fixed rank.

Run with:
    ./scripts/submit sandbox/nsd/nsd_consensus_test/run.py --bg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import joblib

from pysrf import SRF
from pysrf.consensus import EnsembleEmbedding, AlignedConsensus
from sklearn.pipeline import Pipeline

from datasets import load_dataset
from tools.rsa import compute_similarity
from src.utils.figure_theme import save_figure
from src.utils import get_output_dir

# Output directory: set by submit, or default for direct runs
OUTPUT_DIR = get_output_dir()

# Config
SUBJECT_ID = 5
RANK = 10
N_RUNS = 5
MAX_OUTER = 5
MAX_INNER = 10
SEED = 42


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


def plot_individual_embeddings(
    output_path: Path,
    embeddings: list[np.ndarray],
    n_items: int = 500,
) -> None:
    """Plot heatmaps of individual embeddings (first n_items rows)."""
    n_runs = len(embeddings)

    fig, axes = plt.subplots(1, n_runs, figsize=(3 * n_runs, 6))
    if n_runs == 1:
        axes = [axes]

    for i, (emb, ax) in enumerate(zip(embeddings, axes)):
        im = ax.imshow(emb[:n_items], aspect="auto", cmap="viridis")
        ax.set_title(f"Run {i + 1}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Item")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    print(f"Loading NSD subject {SUBJECT_ID}...")
    ds = load_dataset(
        "nsd",
        root="/LOCAL/LABSHARE/natural-scenes-dataset",
        subject_id=SUBJECT_ID,
        roi_name="nsdgeneral",
        zscore_betas=True,
    )
    print(f"  Betas shape: {ds.data.shape}")

    print("Computing similarity matrix...")
    similarity = compute_similarity(ds.data, ds.data, "gaussian_kernel")
    print(f"  Similarity shape: {similarity.shape}")

    print(
        f"\nRunning {N_RUNS} SRF fits with rank={RANK}, max_outer={MAX_OUTER}, max_inner={MAX_INNER}..."
    )

    # Create ensemble and consensus pipeline
    base_model = SRF(
        rank=RANK,
        max_outer=MAX_OUTER,
        max_inner=MAX_INNER,
        random_state=SEED,
        verbose=False,
    )

    ensemble = EnsembleEmbedding(
        base_model,
        n_runs=N_RUNS,
        random_state=SEED,
        n_jobs=-1,
    )

    consensus = AlignedConsensus(
        rank=RANK,
        aggregation="refine",
    )

    pipeline = Pipeline(
        [
            ("ensemble", ensemble),
            ("consensus", consensus),
        ]
    )

    pipeline.fit(similarity)
    consensus_embedding = pipeline.transform(similarity)

    # Get individual embeddings from ensemble
    individual_embeddings = ensemble.embeddings_

    print(f"\nResults:")
    print(
        f"  Individual embeddings: {len(individual_embeddings)} x {individual_embeddings[0].shape}"
    )
    print(f"  Consensus embedding: {consensus_embedding.shape}")
    print(f"  Selected run index: {consensus.selected_run_idx_}")
    print(f"  Agreement scores: {consensus.agreement_scores_}")
    print(f"  Centrality scores: {consensus.centrality_scores_}")

    # Compute reconstruction error
    recon = consensus_embedding @ consensus_embedding.T
    recon_error = np.linalg.norm(similarity - recon, "fro") / np.linalg.norm(
        similarity, "fro"
    )
    print(f"  Reconstruction error: {recon_error:.4f}")

    # Save outputs
    np.save(OUTPUT_DIR / "consensus_embedding.npy", consensus_embedding)
    np.save(OUTPUT_DIR / "individual_embeddings.npy", np.array(individual_embeddings))
    joblib.dump(pipeline, OUTPUT_DIR / "pipeline.joblib")

    # Plot embeddings
    print("\nPlotting...")
    images = ds.metadata["images"]

    print("  - Top-k images per dimension (consensus)...")
    plot_topk_images(
        OUTPUT_DIR / "topk_images_consensus.pdf", consensus_embedding, images, k=8
    )

    print("  - Individual embedding heatmaps...")
    plot_individual_embeddings(
        OUTPUT_DIR / "individual_embeddings.pdf", individual_embeddings
    )

    # Also plot top-k for individual runs
    for i, emb in enumerate(individual_embeddings):
        plot_topk_images(OUTPUT_DIR / f"topk_images_run{i + 1}.pdf", emb, images, k=8)
    print(f"  - Top-k images for each of {N_RUNS} individual runs")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
