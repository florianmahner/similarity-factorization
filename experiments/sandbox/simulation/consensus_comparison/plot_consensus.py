"""
Visualization of consensus methods for symmetric NMF.

Creates publication-quality plots comparing:
1. Reconstruction error vs distance to consensus
2. Embedding visualizations (refine vs select vs median)
3. RSM reconstructions
4. Dimension stability across runs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pysrf import SRF, EnsembleEmbedding, AlignedConsensus
from src.datasets.loaders import load_mur92
from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure


def recon_error(emb: np.ndarray, rsm: np.ndarray) -> float:
    recon = emb @ emb.T
    return np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")


def main():
    output_dir = Path(__file__).parent / "outputs" / "consensus_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    # Use image filenames as labels
    if "images" in dataset.metadata:
        labels = [Path(p).stem for p in dataset.metadata["images"]]
    else:
        labels = [f"Item {i}" for i in range(rsm.shape[0])]
    n_samples = rsm.shape[0]
    rank = 3

    print(f"Fitting ensemble with rank={rank}...")
    pipeline = Pipeline([
        ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=50, n_jobs=-1)),
        ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
    ])
    emb_refine = pipeline.fit_transform(rsm)
    consensus = pipeline.named_steps["consensus"]

    # Get different embeddings
    emb_select = consensus.aligned_embeddings_[consensus.selected_run_idx_]
    emb_median = consensus.consensus_median_

    # Compute metrics
    err_refine = recon_error(emb_refine, rsm)
    err_select = recon_error(emb_select, rsm)
    err_median = recon_error(emb_median, rsm)

    dist_refine = np.linalg.norm(emb_refine - emb_median, "fro")
    dist_select = np.linalg.norm(emb_select - emb_median, "fro")

    print(f"Refine: error={err_refine:.4f}, dist={dist_refine:.4f}")
    print(f"Select: error={err_select:.4f}, dist={dist_select:.4f}")
    print(f"Median: error={err_median:.4f}")

    # =========================================================================
    # Plot 1: Scatter of all runs + methods (error vs distance to consensus)
    # =========================================================================
    fig, ax = create_figure("single")

    # Plot individual runs
    errors = [recon_error(e, rsm) for e in consensus.aligned_embeddings_]
    distances = [np.linalg.norm(e - emb_median, "fro") for e in consensus.aligned_embeddings_]

    ax.scatter(distances, errors, c=GRAY, s=30, alpha=0.6, label="Individual runs")

    # Highlight methods
    ax.scatter([dist_refine], [err_refine], c=TEAL, s=120, marker="*",
               zorder=10, label=f"Refine ({err_refine:.4f})")
    ax.scatter([dist_select], [err_select], c=CYAN, s=100, marker="s",
               zorder=10, label=f"Select ({err_select:.4f})")
    ax.scatter([0], [err_median], c=ROSE, s=100, marker="^",
               zorder=10, label=f"Median ({err_median:.4f})")

    ax.set_xlabel("Distance to consensus median")
    ax.set_ylabel("Reconstruction error")
    ax.legend(fontsize=7, loc="upper right")
    despine(ax)

    save_figure(fig, output_dir / "plot1_error_vs_distance.pdf")
    plt.close(fig)
    print("Saved plot1_error_vs_distance.pdf")

    # =========================================================================
    # Plot 2: Bar chart comparing methods
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    methods = ["Refine", "Select", "Median"]
    errs = [err_refine, err_select, err_median]
    dists = [dist_refine, dist_select, 0]
    colors = [TEAL, CYAN, ROSE]

    # Error bars
    bars = axes[0].bar(methods, errs, color=colors)
    axes[0].set_ylabel("Reconstruction error")
    axes[0].set_ylim(0.24, 0.26)
    for bar, e in zip(bars, errs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, e + 0.001, f"{e:.4f}",
                     ha="center", fontsize=8)
    despine(axes[0])

    # Distance bars
    bars = axes[1].bar(methods, dists, color=colors)
    axes[1].set_ylabel("Distance to consensus")
    for bar, d in zip(bars, dists):
        axes[1].text(bar.get_x() + bar.get_width() / 2, d + 0.01, f"{d:.2f}",
                     ha="center", fontsize=8)
    despine(axes[1])

    fig.suptitle("Consensus Methods Comparison", fontsize=10)
    save_figure(fig, output_dir / "plot2_method_comparison.pdf")
    plt.close(fig)
    print("Saved plot2_method_comparison.pdf")

    # =========================================================================
    # Plot 3: Embedding heatmaps (3 methods side by side)
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=1, ncols=3)

    # Sort by first dimension for visualization
    sort_idx = np.argsort(emb_refine[:, 0])[::-1]

    vmax = max(emb_refine.max(), emb_select.max(), emb_median.max())

    for ax, emb, title in zip(
        axes,
        [emb_refine, emb_select, emb_median],
        ["Refine (NNLS)", "Select (most central)", "Median"],
    ):
        im = ax.imshow(emb[sort_idx], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Items (sorted)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(rank))
        ax.set_xticklabels([f"D{i+1}" for i in range(rank)])

    fig.colorbar(im, ax=axes, shrink=0.8, label="Loading")
    save_figure(fig, output_dir / "plot3_embedding_heatmaps.pdf")
    plt.close(fig)
    print("Saved plot3_embedding_heatmaps.pdf")

    # =========================================================================
    # Plot 4: RSM reconstructions
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=1, ncols=4)

    rsms = [rsm, emb_refine @ emb_refine.T, emb_select @ emb_select.T, emb_median @ emb_median.T]
    titles = ["Original RSM", f"Refine ({err_refine:.4f})", f"Select ({err_select:.4f})", f"Median ({err_median:.4f})"]

    vmin, vmax = rsm.min(), rsm.max()
    for ax, r, title in zip(axes, rsms, titles):
        im = ax.imshow(r, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="Similarity")
    save_figure(fig, output_dir / "plot4_rsm_reconstruction.pdf")
    plt.close(fig)
    print("Saved plot4_rsm_reconstruction.pdf")

    # =========================================================================
    # Plot 5: Dimension stability across runs (violin plots)
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=rank)

    for j, ax in enumerate(axes):
        # Get all estimates for this dimension
        dim_data = consensus.aligned_embeddings_[:, :, j]  # (n_runs, n_samples)

        # Compute cosine similarity of each run to median
        median_dim = emb_median[:, j]
        median_norm = median_dim / (np.linalg.norm(median_dim) + 1e-10)

        sims = []
        for run_dim in dim_data:
            run_norm = run_dim / (np.linalg.norm(run_dim) + 1e-10)
            sims.append(np.dot(run_norm, median_norm))

        parts = ax.violinplot([sims], positions=[0], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(CYCLE[j % len(CYCLE)])
            pc.set_alpha(0.7)

        ax.set_ylabel("Cosine sim to median" if j == 0 else "")
        ax.set_title(f"Dim {j+1}", fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(0.95, 1.01)
        despine(ax)

    fig.suptitle("Dimension Stability Across Runs", fontsize=10)
    save_figure(fig, output_dir / "plot5_dimension_stability.pdf")
    plt.close(fig)
    print("Saved plot5_dimension_stability.pdf")

    # =========================================================================
    # Plot 6: 3D embedding scatter (if rank=3)
    # =========================================================================
    if rank == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 3.5))

        for idx, (emb, title, color) in enumerate([
            (emb_refine, "Refine", TEAL),
            (emb_select, "Select", CYAN),
            (emb_median, "Median", ROSE),
        ]):
            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=color, s=20, alpha=0.7)
            ax.set_xlabel("D1", fontsize=8)
            ax.set_ylabel("D2", fontsize=8)
            ax.set_zlabel("D3", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        fig.savefig(output_dir / "plot6_3d_embedding.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved plot6_3d_embedding.pdf")

    # =========================================================================
    # Plot 7: Top-k items per dimension comparison
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=rank, ncols=3, sharex=False, sharey=False)
    if rank == 1:
        axes = axes.reshape(1, -1)

    k = 10  # top-k items to show

    for j in range(rank):
        for col, (emb, title) in enumerate([
            (emb_refine, "Refine"),
            (emb_select, "Select"),
            (emb_median, "Median"),
        ]):
            ax = axes[j, col] if rank > 1 else axes[col]
            top_idx = np.argsort(emb[:, j])[-k:][::-1]
            top_vals = emb[top_idx, j]
            top_labels = [labels[i][:15] for i in top_idx]

            ax.barh(range(k), top_vals, color=CYCLE[j % len(CYCLE)])
            ax.set_yticks(range(k))
            ax.set_yticklabels(top_labels, fontsize=6)
            ax.invert_yaxis()
            if j == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Dim {j+1}", fontsize=8)
            despine(ax)

    save_figure(fig, output_dir / "plot7_topk_items.pdf")
    plt.close(fig)
    print("Saved plot7_topk_items.pdf")

    # =========================================================================
    # Plot 8: Summary figure - key insight
    # =========================================================================
    fig, ax = create_figure("single")

    # Arrow showing the refinement process
    ax.annotate(
        "",
        xy=(dist_refine, err_refine),
        xytext=(0, err_median),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
    )

    # Points
    ax.scatter([0], [err_median], c=ROSE, s=150, marker="^", zorder=10, label="Median (breaks factorization)")
    ax.scatter([dist_refine], [err_refine], c=TEAL, s=150, marker="*", zorder=10, label="Refine (NNLS projection)")
    ax.scatter([dist_select], [err_select], c=CYAN, s=100, marker="s", zorder=10, label="Select (most central run)")

    # Optimal line
    ax.axhline(err_select, color=GRAY_LIGHT, linestyle="--", label="Optimal reconstruction")

    ax.set_xlabel("Distance to consensus median")
    ax.set_ylabel("Reconstruction error")
    ax.set_xlim(-0.05, max(dist_select, dist_refine) + 0.1)
    ax.legend(fontsize=7, loc="upper right")

    ax.text(0.02, err_median + 0.001, "Start:\nMedian", fontsize=7, ha="left")
    ax.text(dist_refine + 0.02, err_refine, "End:\nRefined", fontsize=7, ha="left")

    ax.set_title("NNLS Refinement: Best of Both Worlds", fontsize=10)
    despine(ax)

    save_figure(fig, output_dir / "plot8_key_insight.pdf")
    plt.close(fig)
    print("Saved plot8_key_insight.pdf")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
