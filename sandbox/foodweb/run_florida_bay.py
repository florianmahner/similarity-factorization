#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cleaned analysis script for Symmetric NMF on the Florida Bay food web.

This script focuses on the three meaningful analyses for this
one-class (Positive-Unlabeled) dataset:
1. Embedding Structure (by group)
2. Reconstructed Group Connectivity (imputation)
3. Prediction Distribution (known links vs. unknown pairs)

Removed all "Not Applicable" plots (e.g., ROC/PR).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Assuming pysrf is installed
from pysrf import SRF


def load_florida_bay_data(
    graph_path: Path, meta_path: Path
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Loads the graph and metadata."""
    meta_df = pd.read_csv(meta_path)

    edges = []
    with open(graph_path) as f:
        for line in f:
            line = line.strip().strip('"')
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    source = int(parts[0].strip('"'))
                    target = int(parts[1].strip('"'))
                    edges.append((source, target))
                except ValueError:
                    continue

    G = nx.DiGraph()
    G.add_edges_from(edges)

    return G, meta_df


def create_adjacency_with_missing(G: nx.DiGraph, n_nodes: int) -> np.ndarray:
    """Creates a symmetric adjacency matrix with 1.0 for links and NaN elsewhere."""
    # adj = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)

    for source, target in G.edges():
        if source < n_nodes and target < n_nodes:
            adj[source, target] = 1.0

    # Symmetrize: an edge exists if it's present in either direction
    adj_sym = np.fmax(adj, adj.T)

    # Set diagonal to NaN (no self-loops)
    np.fill_diagonal(adj_sym, np.nan)

    return adj_sym


def evaluate_reconstruction_on_links(
    adj: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """
    Evaluates reconstruction quality ONLY on the observed links (1.0s).
    """
    observed_mask = ~np.isnan(adj)

    y_true = adj[observed_mask]  # These are all 1.0
    y_pred = reconstruction[observed_mask]
    y_pred = np.clip(y_pred, 0, 1)

    metrics = {}

    # Report prediction statistics on the EDGES
    metrics["mean_prediction_on_edges"] = y_pred.mean()
    metrics["std_prediction_on_edges"] = y_pred.std()
    metrics["min_prediction_on_edges"] = y_pred.min()
    metrics["max_prediction_on_edges"] = y_pred.max()

    # Continuous metrics (reconstruction error for the 1.0s)
    metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))

    return metrics


def analyze_group_connectivity(W: np.ndarray, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes RECONSTRUCTED connectivity patterns within and between groups.
    This computes the mean *imputed* score from W@W.T for all pairs.
    """
    groups = meta_df.set_index("node_id")["group"].to_dict()
    unique_groups = sorted([g for g in meta_df["group"].unique() if pd.notna(g)])
    n_nodes = W.shape[0]

    results = []
    reconstruction = W @ W.T
    np.fill_diagonal(reconstruction, np.nan)  # Ignore self-loops

    for g1 in unique_groups:
        nodes_g1 = [i for i in range(n_nodes) if groups.get(i, "") == g1]
        if not nodes_g1:
            continue

        for g2 in unique_groups:
            if unique_groups.index(g1) > unique_groups.index(g2):
                continue  # Only compute upper triangle

            nodes_g2 = [i for i in range(n_nodes) if groups.get(i, "") == g2]
            if not nodes_g2:
                continue

            # Efficiently select the sub-matrix of reconstructed scores
            if g1 == g2:
                if len(nodes_g1) < 2:
                    continue
                sub_matrix = reconstruction[np.ix_(nodes_g1, nodes_g1)]
                recon_scores = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
            else:
                sub_matrix = reconstruction[np.ix_(nodes_g1, nodes_g2)]
                recon_scores = sub_matrix.flatten()

            valid_scores = recon_scores[~np.isnan(recon_scores)]
            if valid_scores.size == 0:
                continue

            results.append(
                {
                    "group_1": g1,
                    "group_2": g2,
                    "n_nodes_1": len(nodes_g1),
                    "n_nodes_2": len(nodes_g2),
                    "n_pairs": valid_scores.size,
                    "reconstructed_density": valid_scores.mean(),
                    "within_group": g1 == g2,
                }
            )
    return pd.DataFrame(results)


def analyze_dimension_group_loading(
    W: np.ndarray, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """Analyzes how each dimension loads on different ecological groups."""
    groups = meta_df.set_index("node_id")["group"].to_dict()
    unique_groups = sorted([g for g in meta_df["group"].unique() if pd.notna(g)])
    n_nodes = W.shape[0]

    results = []
    for dim in range(W.shape[1]):
        dim_values = W[:, dim]
        for group in unique_groups:
            nodes = [i for i in range(n_nodes) if groups.get(i, "") == group]
            if not nodes:
                continue

            group_values = dim_values[nodes]
            valid_group_values = group_values[~np.isnan(group_values)]
            if valid_group_values.size == 0:
                continue

            results.append(
                {
                    "dimension": dim,
                    "group": group,
                    "mean_loading": valid_group_values.mean(),
                    "std_loading": valid_group_values.std(),
                    "max_loading": valid_group_values.max(),
                    "n_nodes": len(valid_group_values),
                }
            )
    return pd.DataFrame(results)


def plot_group_connectivity_matrix(
    connectivity_df: pd.DataFrame, output_path: Path
) -> None:
    """Visualizes RECONSTRUCTED within and between group connectivity."""
    if connectivity_df.empty:
        print("  - WARNING: Connectivity DataFrame is empty. Skipping plot.")
        return

    unique_groups = sorted(
        set(connectivity_df["group_1"]) | set(connectivity_df["group_2"])
    )
    n_groups = len(unique_groups)
    recon_matrix = np.full((n_groups, n_groups), np.nan)

    for _, row in connectivity_df.iterrows():
        i = unique_groups.index(row["group_1"])
        j = unique_groups.index(row["group_2"])
        recon_matrix[i, j] = row["reconstructed_density"]
        recon_matrix[j, i] = row["reconstructed_density"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(recon_matrix, cmap="YlOrRd", vmin=0, vmax=np.nanmax(recon_matrix))
    ax.set_title("Reconstructed (Imputed) Connectivity Between Groups", fontsize=14)
    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(unique_groups, rotation=45, ha="right")
    ax.set_yticklabels(unique_groups)
    plt.colorbar(im, ax=ax, label="Mean Reconstructed Score")

    # Add text annotations
    for i in range(n_groups):
        for j in range(n_groups):
            if not np.isnan(recon_matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{recon_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dimension_group_heatmap(dimension_df: pd.DataFrame, output_path: Path) -> None:
    """Visualizes how dimensions load on different ecological groups."""
    if dimension_df.empty:
        print("  - WARNING: Dimension DataFrame is empty. Skipping plot.")
        return

    pivot = dimension_df.pivot(
        index="group", columns="dimension", values="mean_loading"
    )
    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1]), max(4, pivot.shape[0] * 0.5))
    )
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Dimension (Factor)")
    ax.set_ylabel("Ecological Group")
    ax.set_title("Mean Loading of Each Dimension on Ecological Groups")
    plt.colorbar(im, ax=ax, label="Mean Loading")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_prediction_distribution(
    adj: np.ndarray, reconstruction: np.ndarray, output_path: Path
) -> None:
    """Plots distribution of predicted probabilities for observed vs unobserved."""
    observed_mask = ~np.isnan(adj)
    missing_mask = np.isnan(adj)
    np.fill_diagonal(missing_mask, False)  # Ignore diagonal

    observed_preds = reconstruction[observed_mask]
    unobserved_preds = reconstruction[missing_mask]

    # Filter any NaNs from the predictions themselves
    observed_preds = observed_preds[~np.isnan(observed_preds)]
    unobserved_preds = unobserved_preds[~np.isnan(unobserved_preds)]

    if observed_preds.size == 0 and unobserved_preds.size == 0:
        print("  - WARNING: No valid predictions to plot. Skipping distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        unobserved_preds,
        bins=50,
        alpha=0.6,
        label=f"Unobserved Pairs (NaNs)\n(n={unobserved_preds.size})",
        color="gray",
        density=True,
    )
    axes[0].hist(
        observed_preds,
        bins=50,
        alpha=0.6,
        label=f"Observed Edges (1.0s)\n(n={observed_preds.size})",
        color="green",
        density=True,
    )
    axes[0].set_xlabel("Reconstructed Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    data_to_plot = []
    labels = []
    if unobserved_preds.size > 0:
        data_to_plot.append(unobserved_preds)
        labels.append("Unobserved\nPairs")
    if observed_preds.size > 0:
        data_to_plot.append(observed_preds)
        labels.append("Observed\nEdges")

    if data_to_plot:
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel("Reconstructed Score")
        axes[1].set_title("Prediction Statistics")
        axes[1].grid(alpha=0.3, axis="y")

        stats_text = ""
        if observed_preds.size > 0:
            stats_text += f"Observed: μ={observed_preds.mean():.3f}, σ={observed_preds.std():.3f}\n"
        if unobserved_preds.size > 0:
            stats_text += f"Unobserved: μ={unobserved_preds.mean():.3f}, σ={unobserved_preds.std():.3f}"
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_embedding_by_group(
    W: np.ndarray, meta_df: pd.DataFrame, output_path: Path
) -> None:
    """Plots pairwise combinations of first few dimensions, colored by group."""
    n_dims = min(4, W.shape[1])
    if n_dims < 2:
        print("  - WARNING: Need at least 2 dimensions to plot embedding. Skipping.")
        return

    fig, axes = plt.subplots(n_dims - 1, n_dims - 1, figsize=(14, 14), squeeze=False)
    groups_series = meta_df.set_index("node_id")["group"]
    unique_groups = sorted([g for g in groups_series.unique() if pd.notna(g)])

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
    group_to_color = {g: colors[i] for i, g in enumerate(unique_groups)}

    for i in range(n_dims - 1):  # y-axis dimension
        for j in range(n_dims - 1):  # x-axis dimension
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            dim_x = j
            dim_y = i + 1

            for node_id in range(W.shape[0]):
                x, y = W[node_id, dim_x], W[node_id, dim_y]
                if np.isnan(x) or np.isnan(y):
                    continue  # Skip if embedding has NaNs

                group = groups_series.get(node_id, np.nan)
                color = group_to_color.get(group, "lightgray")
                ax.scatter(x, y, c=[color], s=30, alpha=0.6)

            ax.set_xlabel(f"Dimension {dim_x}")
            ax.set_ylabel(f"Dimension {dim_y}")
            ax.grid(alpha=0.3)

    legend_elements = [
        Patch(facecolor=group_to_color[g], label=g, alpha=0.6) for g in unique_groups
    ]
    legend_elements.append(Patch(facecolor="lightgray", label="Unlabeled", alpha=0.6))

    fig.legend(handles=legend_elements, loc="center right", fontsize=10)
    fig.suptitle("Pairwise Embeddings (Factors) Colored by Group", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    data_dir = Path("data/florida-bay")
    # This assumes your data is in 'data/foodweb/graph.csv' etc.
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"WARNING: Data directory {data_dir} was not found.")
        print(f"Please place 'graph.csv' and 'meta.csv' in that folder.")
        # As a fallback, try loading from the current directory
        data_dir = Path(".")

    graph_path = data_dir / "graph.csv"
    meta_path = data_dir / "meta.csv"

    if not graph_path.exists() or not meta_path.exists():
        print(f"ERROR: Could not find 'graph.csv' or 'meta.csv' in {data_dir}")
        print("Please make sure the data files are in the correct location.")
        return

    print("Loading Florida Bay food web data...")
    G, meta_df = load_florida_bay_data(graph_path, meta_path)
    n_nodes = len(meta_df)
    print(f"Network: {n_nodes} nodes, {G.number_of_edges()} directed edges")

    print("\nCreating symmetric adjacency matrix with missing values...")
    adj = create_adjacency_with_missing(G, n_nodes)

    observed_mask = ~np.isnan(adj)
    n_observed = observed_mask.sum()
    total_pairs = n_nodes * (n_nodes - 1)  # Off-diagonal

    if total_pairs <= 0:
        print("Adjacency matrix is empty.")
        return

    observed_pct = 100 * n_observed / total_pairs
    print(f"Adjacency matrix: {adj.shape}, {n_observed} observed links")
    print(f"Sparsity: {observed_pct:.2f}% of off-diagonal pairs are known links.")

    n_edges = np.sum(adj[observed_mask] == 1)
    n_non_edges = np.sum(adj[observed_mask] == 0)

    print(f"  - Observed Edges (1.0s): {n_edges}")
    print(f"  - Observed Non-Edges (0.0s): {n_non_edges}")

    print("\nIMPORTANT: This is a ONE-CLASS (PU) imputation task.")
    print("We will evaluate by imputation and embedding structure.")

    output_dir = Path(
        "experiments/development/foodweb/outputs"
    ) / datetime.now().strftime("%y%m%d/%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print("\nFitting SRF model...")
    rank = 8
    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=2000,
        max_inner=50,
        tol=0.0,
        verbose=1,
        init="random_sqrt",
        random_state=42,
        missing_values=np.nan,
        loss="frobenius",
    )

    W = model.fit_transform(adj)
    reconstruction = model.reconstruct()

    print(f"\nFactorization complete: W shape = {W.shape}")
    print(f"Iterations: {model.n_iter_}")

    print("\nEvaluating reconstruction performance (on observed 1.0s)...")
    metrics = evaluate_reconstruction_on_links(adj, reconstruction)

    print("\nPrediction Metrics (on observed 1.0s only):")
    print(
        f"  Mean prediction: {metrics.get('mean_prediction_on_edges', 0):.4f} (Ideal: 1.0)"
    )
    print(
        f"  Range: [{metrics.get('min_prediction_on_edges', 0):.4f}, {metrics.get('max_prediction_on_edges', 0):.4f}]"
    )
    print("\nReconstruction Error on Observed Edges:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")

    print("\nAnalyzing RECONSTRUCTED group connectivity patterns...")
    connectivity_df = analyze_group_connectivity(W, meta_df)
    connectivity_df.to_csv(
        output_dir / "reconstructed_group_connectivity.csv", index=False
    )

    print("\nAnalyzing dimension-group loadings...")
    dimension_df = analyze_dimension_group_loading(W, meta_df)
    dimension_df.to_csv(output_dir / "dimension_group_loadings.csv", index=False)

    print("\nSaving results...")
    np.save(output_dir / "W.npy", W)
    np.save(output_dir / "reconstruction.npy", reconstruction)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    print("\nCreating meaningful visualizations...")

    print("  - Prediction distributions (Observed vs. Unobserved)...")
    plot_prediction_distribution(
        adj, reconstruction, output_dir / "prediction_distribution.png"
    )

    print("  - Reconstructed group connectivity matrix...")
    plot_group_connectivity_matrix(
        connectivity_df, output_dir / "reconstructed_group_connectivity.png"
    )

    print("  - Dimension-group heatmap...")
    plot_dimension_group_heatmap(
        dimension_df, output_dir / "dimension_group_heatmap.png"
    )

    print("  - Embedding by group...")
    plot_embedding_by_group(W, meta_df, output_dir / "embedding_by_group.png")

    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*60}")
    print("\n1. Embedding by group (plot):")
    print("   - LOOK FOR: Clear clusters of colors.")
    print("   - MEANING: Your NMF model has successfully learned 'roles' that")
    print("     correspond to the known ecological groups.")

    print("\n2. Reconstructed group connectivity (plot):")
    print("   - LOOK FOR: High-value (yellow) cells, both on and off-diagonal.")
    print("   - MEANING: The model *predicts* high interaction rates between")
    print("     these groups, based on the patterns it learned from the sparse data.")

    print("\n3. Dimension-group heatmap (plot):")
    print("   - LOOK FOR: Dimensions (columns) that are 'hot' (yellow) for")
    print("     only one or two groups (rows).")
    print("   - MEANING: This shows a 'basis vector' for that group. E.g.,")
    print("     Dimension 2 might be the 'Phytoplankton' dimension.")

    print("\n4. Prediction distribution (plot):")
    print("   - LOOK FOR: A green distribution (Observed 1s) shifted far to the")
    print("     right (near 1.0), and a gray distribution (Unobserved NaNs)")
    print("     shifted to the left (near 0.0).")
    print("   - MEANING: The model is confident, assigning high scores to known")
    print("     links and low scores to unknown pairs.")

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
