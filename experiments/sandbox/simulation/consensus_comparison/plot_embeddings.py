"""
Visualize embeddings from consensus methods across multiple seeds.

Shows how the refined consensus embedding is stable across different
random seeds for the ensemble.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from pysrf import SRF, EnsembleEmbedding, AlignedConsensus
from src.datasets.loaders import load_mur92
from src.colors import TEAL, CYAN, ROSE, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure


def recon_error(emb: np.ndarray, rsm: np.ndarray) -> float:
    recon = emb @ emb.T
    return np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")


def l2_norm_columns(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def align_to_reference(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target embedding to reference using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    ref_norm = l2_norm_columns(ref)
    target_norm = l2_norm_columns(target)
    sim = ref_norm.T @ target_norm
    _, col_ind = linear_sum_assignment(-sim)
    return target[:, col_ind]


def main():
    output_dir = Path(__file__).parent / "outputs" / "embedding_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_mur92("/SSD/datasets/similarity_datasets/mur92")
    rsm = dataset.rsm
    n_samples = rsm.shape[0]
    rank = 3
    n_runs = 50
    seeds = [0, 42, 123, 456, 789]

    print(f"Running {len(seeds)} seeds with {n_runs} runs each...")

    # Collect embeddings from different seeds
    refine_embeddings = []
    select_embeddings = []
    median_embeddings = []

    for seed in seeds:
        print(f"  Seed {seed}...")
        pipeline = Pipeline([
            ("ensemble", EnsembleEmbedding(SRF(rank=rank), n_runs=n_runs, n_jobs=-1, random_state=seed)),
            ("consensus", AlignedConsensus(rank=rank, aggregation="refine")),
        ])
        emb_refine = pipeline.fit_transform(rsm)
        consensus = pipeline.named_steps["consensus"]

        refine_embeddings.append(emb_refine)
        select_embeddings.append(consensus.aligned_embeddings_[consensus.selected_run_idx_])
        median_embeddings.append(consensus.consensus_median_)

    # Align all to first seed's refine embedding
    ref = refine_embeddings[0]
    for i in range(1, len(seeds)):
        refine_embeddings[i] = align_to_reference(ref, refine_embeddings[i])
        select_embeddings[i] = align_to_reference(ref, select_embeddings[i])
        median_embeddings[i] = align_to_reference(ref, median_embeddings[i])

    # =========================================================================
    # Plot 1: 3D embedding comparison across seeds
    # =========================================================================
    fig = plt.figure(figsize=(12, 4))

    for col, (embeddings, title) in enumerate([
        (refine_embeddings, "Refine (NNLS)"),
        (select_embeddings, "Select (most central)"),
        (median_embeddings, "Median"),
    ]):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")

        for i, emb in enumerate(embeddings):
            alpha = 0.8 if i == 0 else 0.4
            size = 25 if i == 0 else 15
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                      c=CYCLE[i % len(CYCLE)], s=size, alpha=alpha,
                      label=f"Seed {seeds[i]}")

        ax.set_xlabel("D1", fontsize=8)
        ax.set_ylabel("D2", fontsize=8)
        ax.set_zlabel("D3", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "plot1_3d_across_seeds.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved plot1_3d_across_seeds.pdf")

    # =========================================================================
    # Plot 2: Embedding stability - variance across seeds
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=1, ncols=3)

    for col, (embeddings, title) in enumerate([
        (refine_embeddings, "Refine"),
        (select_embeddings, "Select"),
        (median_embeddings, "Median"),
    ]):
        ax = axes[col]
        stacked = np.stack(embeddings, axis=0)  # (n_seeds, n_samples, rank)

        # Compute std across seeds for each item
        std_per_item = np.std(stacked, axis=0)  # (n_samples, rank)
        mean_std = std_per_item.mean(axis=0)  # (rank,)

        # Plot variance per dimension
        bars = ax.bar(range(rank), mean_std, color=[CYCLE[j % len(CYCLE)] for j in range(rank)])
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Mean std across seeds" if col == 0 else "")
        ax.set_title(f"{title}\n(total: {mean_std.sum():.4f})", fontsize=9)
        ax.set_xticks(range(rank))
        ax.set_xticklabels([f"D{j+1}" for j in range(rank)])
        despine(ax)

    save_figure(fig, output_dir / "plot2_variance_across_seeds.pdf")
    plt.close(fig)
    print("Saved plot2_variance_across_seeds.pdf")

    # =========================================================================
    # Plot 3: Heatmap of embeddings from each seed (refine only)
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=1, ncols=len(seeds))

    # Sort by first seed's first dimension
    sort_idx = np.argsort(refine_embeddings[0][:, 0])[::-1]
    vmax = max(e.max() for e in refine_embeddings)

    for i, (ax, emb) in enumerate(zip(axes, refine_embeddings)):
        im = ax.imshow(emb[sort_idx], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"Seed {seeds[i]}", fontsize=9)
        ax.set_xlabel("Dim")
        ax.set_ylabel("Items" if i == 0 else "")
        ax.set_xticks(range(rank))
        ax.set_xticklabels([f"D{j+1}" for j in range(rank)], fontsize=7)
        if i > 0:
            ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="Loading")
    fig.suptitle("Refine Embeddings Across Seeds", fontsize=10)
    save_figure(fig, output_dir / "plot3_refine_heatmaps.pdf")
    plt.close(fig)
    print("Saved plot3_refine_heatmaps.pdf")

    # =========================================================================
    # Plot 4: Pairwise cosine similarity between seeds
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=1, ncols=3)

    for col, (embeddings, title) in enumerate([
        (refine_embeddings, "Refine"),
        (select_embeddings, "Select"),
        (median_embeddings, "Median"),
    ]):
        ax = axes[col]
        n_seeds = len(embeddings)
        sim_matrix = np.zeros((n_seeds, n_seeds))

        for i in range(n_seeds):
            for j in range(n_seeds):
                # Flatten and compute cosine similarity
                e_i = embeddings[i].flatten()
                e_j = embeddings[j].flatten()
                sim_matrix[i, j] = np.dot(e_i, e_j) / (np.linalg.norm(e_i) * np.linalg.norm(e_j))

        im = ax.imshow(sim_matrix, cmap="viridis", vmin=0.95, vmax=1.0)
        ax.set_title(f"{title}\n(mean off-diag: {sim_matrix[np.triu_indices(n_seeds, 1)].mean():.4f})", fontsize=9)
        ax.set_xticks(range(n_seeds))
        ax.set_yticks(range(n_seeds))
        ax.set_xticklabels([f"S{s}" for s in seeds], fontsize=7)
        ax.set_yticklabels([f"S{s}" for s in seeds], fontsize=7)

        # Add text annotations
        for i in range(n_seeds):
            for j in range(n_seeds):
                ax.text(j, i, f"{sim_matrix[i,j]:.3f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine similarity")
    save_figure(fig, output_dir / "plot4_seed_similarity.pdf")
    plt.close(fig)
    print("Saved plot4_seed_similarity.pdf")

    # =========================================================================
    # Plot 5: Dimension-wise comparison across seeds
    # =========================================================================
    fig, axes = create_figure("full_width", nrows=rank, ncols=1, sharex=True)

    for j in range(rank):
        ax = axes[j]

        for i, (emb, seed) in enumerate(zip(refine_embeddings, seeds)):
            dim_vals = emb[sort_idx, j]
            ax.plot(dim_vals, label=f"Seed {seed}", color=CYCLE[i % len(CYCLE)],
                   alpha=0.7, linewidth=1.5)

        ax.set_ylabel(f"D{j+1}", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, ncol=len(seeds), loc="upper right")
        if j == rank - 1:
            ax.set_xlabel("Items (sorted)")
        despine(ax)

    fig.suptitle("Dimension Values Across Seeds (Refine)", fontsize=10)
    save_figure(fig, output_dir / "plot5_dimension_traces.pdf")
    plt.close(fig)
    print("Saved plot5_dimension_traces.pdf")

    # =========================================================================
    # Plot 6: Summary statistics
    # =========================================================================
    fig, axes = create_figure("wide", nrows=1, ncols=2)

    # Compute stats
    methods = ["Refine", "Select", "Median"]
    all_embeddings = [refine_embeddings, select_embeddings, median_embeddings]

    # Mean reconstruction error
    errors = []
    for embs in all_embeddings:
        errs = [recon_error(e, rsm) for e in embs]
        errors.append((np.mean(errs), np.std(errs)))

    ax = axes[0]
    x = range(len(methods))
    means = [e[0] for e in errors]
    stds = [e[1] for e in errors]
    bars = ax.bar(x, means, yerr=stds, color=[TEAL, CYAN, ROSE], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Error across seeds", fontsize=9)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f"{m:.4f}±{s:.4f}", ha="center", fontsize=7)
    despine(ax)

    # Mean pairwise similarity
    sims = []
    for embs in all_embeddings:
        pairwise = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                e_i = embs[i].flatten()
                e_j = embs[j].flatten()
                pairwise.append(np.dot(e_i, e_j) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)))
        sims.append((np.mean(pairwise), np.std(pairwise)))

    ax = axes[1]
    means = [s[0] for s in sims]
    stds = [s[1] for s in sims]
    bars = ax.bar(x, means, yerr=stds, color=[TEAL, CYAN, ROSE], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Pairwise cosine similarity")
    ax.set_title("Stability across seeds", fontsize=9)
    ax.set_ylim(0.98, 1.001)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f"{m:.4f}", ha="center", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "plot6_summary_stats.pdf")
    plt.close(fig)
    print("Saved plot6_summary_stats.pdf")

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSettings: rank={rank}, n_runs={n_runs}, seeds={seeds}")
    print(f"\nReconstruction errors (mean ± std across seeds):")
    for method, (mean, std) in zip(methods, errors):
        print(f"  {method}: {mean:.4f} ± {std:.6f}")
    print(f"\nPairwise similarity across seeds:")
    for method, (mean, std) in zip(methods, sims):
        print(f"  {method}: {mean:.4f} ± {std:.4f}")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
