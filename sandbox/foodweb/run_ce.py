#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis script for Symmetric NMF on the Florida Bay food web.

VERSION 3: Implements the 'classification' approach.
- Assumes all unobserved (NaN) links are non-links (0.0).
- Uses loss='cross-entropy'.
- Evaluates the model as a binary classifier using AUC-ROC and AUC-PR.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Imports for classification metrics are needed again
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

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


def create_adjacency_dense(G: nx.DiGraph, n_nodes: int) -> np.ndarray:
    """
    Creates a symmetric, DENSE adjacency matrix.
    - 1.0 for known links.
    - 0.0 for all other pairs (assumed non-links).
    """
    adj = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    for source, target in G.edges():
        if source < n_nodes and target < n_nodes:
            adj[source, target] = 1.0

    # Symmetrize: an edge exists if it's present in either direction
    adj_sym = np.fmax(adj, adj.T)

    # *** THIS IS THE KEY CHANGE ***
    # Fill all remaining NaNs (unobserved) with 0.0
    adj_sym[np.isnan(adj_sym)] = 0.0

    # Set diagonal to 0.0 (no self-loops)
    np.fill_diagonal(adj_sym, 0.0)

    return adj_sym


def evaluate_binary_reconstruction(
    adj: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """
    Evaluate reconstruction quality for binary classification.
    This is valid now because 'adj' contains both 0s and 1s.
    """
    # We evaluate on all pairs (links and non-links)
    y_true = adj.flatten()
    y_pred = reconstruction.flatten()

    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0, 1)

    metrics = {}

    # Binary classification metrics
    metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
    metrics["auc_pr"] = average_precision_score(y_true, y_pred)

    # Threshold-based metrics at 0.5
    y_pred_binary = (y_pred > 0.5).astype(float)
    metrics["accuracy"] = np.mean(y_true == y_pred_binary)

    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    metrics["precision"] = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["recall"] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Continuous metrics (overall reconstruction error)
    metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))

    return metrics


def analyze_group_connectivity(W: np.ndarray, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes RECONSTRUCTED connectivity patterns within and between groups.
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
                continue

            nodes_g2 = [i for i in range(n_nodes) if groups.get(i, "") == g2]
            if not nodes_g2:
                continue

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


def plot_roc_pr_curves(
    adj: np.ndarray, reconstruction: np.ndarray, output_path: Path
) -> None:
    """
    Plots ROC and Precision-Recall curves. This is now a key evaluation plot.
    """
    y_true = adj.flatten()
    y_pred = np.clip(reconstruction.flatten(), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    baseline = y_true.mean()  # Baseline = prevalence of 1s

    axes[1].plot(recall, precision, linewidth=2, label=f"AP = {auc_pr:.3f}")
    axes[1].axhline(
        baseline,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Baseline (AP = {baseline:.3f})",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    ax.set_title("Reconstructed Connectivity Between Groups", fontsize=14)
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
    """Plots distribution of predictions for 0s vs 1s."""

    # Select true edges (1s) and assumed non-edges (0s)
    edges = reconstruction[adj == 1]
    non_edges = reconstruction[adj == 0]

    # Filter any NaNs from the predictions themselves
    edges = edges[~np.isnan(edges)]
    non_edges = non_edges[~np.isnan(non_edges)]

    if edges.size == 0 and non_edges.size == 0:
        print("  - WARNING: No valid predictions to plot. Skipping distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(
        non_edges,
        bins=50,
        alpha=0.6,
        label=f"Assumed Non-Edges (0.0)\n(n={non_edges.size})",
        color="blue",
        density=True,
    )
    axes[0].hist(
        edges,
        bins=50,
        alpha=0.6,
        label=f"True Edges (1.0)\n(n={edges.size})",
        color="red",
        density=True,
    )
    axes[0].set_xlabel("Reconstructed Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction Distribution (Classifier Output)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    data_to_plot = []
    labels = []
    if non_edges.size > 0:
        data_to_plot.append(non_edges)
        labels.append("Non-Edges (0.0)")
    if edges.size > 0:
        data_to_plot.append(edges)
        labels.append("Edges (1.0)")

    if data_to_plot:
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel("Reconstructed Score")
        axes[1].set_title("Prediction Statistics")
        axes[1].grid(alpha=0.3, axis="y")

        stats_text = ""
        if edges.size > 0:
            stats_text += f"Edges: μ={edges.mean():.3f}, σ={edges.std():.3f}\n"
        if non_edges.size > 0:
            stats_text += (
                f"Non-Edges: μ={non_edges.mean():.3f}, σ={non_edges.std():.3f}"
            )
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
                    continue

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
    data_dir = Path("data/foodweb")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"WARNING: Data directory {data_dir} was not found.")
        print(f"Please place 'graph.csv' and 'meta.csv' in that folder.")
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

    print("\nCreating DENSE symmetric adjacency matrix (0.0 for non-links)...")
    adj = create_adjacency_dense(G, n_nodes)

    n_ones = np.sum(adj == 1)
    n_zeros = np.sum(adj == 0)
    total_off_diag = n_ones + n_zeros

    if total_off_diag > 0:
        print(f"Adjacency matrix: {adj.shape}")
        print(
            f"  - Known Links (1.0s): {n_ones} ({100 * n_ones / total_off_diag:.2f}%)"
        )
        print(
            f"  - Assumed Non-Links (0.0s): {n_zeros} ({100 * n_zeros / total_off_diag:.2f}%)"
        )
    else:
        print("Adjacency matrix is empty.")
        return

    print("\nIMPORTANT: This is now a BINARY CLASSIFICATION task.")

    output_dir = Path(
        "experiments/development/foodweb/outputs"
    ) / datetime.now().strftime("%y%m%d/%H%M%S_classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print("\nFitting SRF model (loss=cross-entropy)...")
    rank = 8
    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=1000,
        max_inner=30,
        tol=0.0,
        verbose=1,
        init="random_sqrt",
        random_state=42,
        missing_values=np.nan,  # This no longer has an effect, but is good practice
        bounds=(0.0, 1.0),  # Clip predictions to [0, 1]
        loss="cross-entropy",  # Use cross-entropy as requested
    )

    W = model.fit_transform(adj)
    reconstruction = model.reconstruct()

    print(f"\nFactorization complete: W shape = {W.shape}")
    print(f"Iterations: {model.n_iter_}")

    print("\nEvaluating binary classification performance...")
    metrics = evaluate_binary_reconstruction(adj, reconstruction)

    print("\nClassification Metrics:")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f} (0.5=random, 1.0=perfect)")
    print(
        f"  AUC-PR: {metrics.get('auc_pr', 0):.4f} (baseline={n_ones/total_off_diag:.3f})"
    )
    print(f"\nMetrics at 0.5 Threshold:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")

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

    print("  - ROC and PR curves...")
    plot_roc_pr_curves(adj, reconstruction, output_dir / "roc_pr_curves.png")

    print("  - Prediction distributions (0s vs 1s)...")
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
    print("INTERPRETATION GUIDE (CLASSIFICATION):")
    print(f"{'='*60}")
    print("\n1. AUC-ROC & AUC-PR (plots):")
    print("   - LOOK FOR: Curves that are high and to the top-left.")
    print("   - 'AUC-ROC > 0.5' means your model is better than random guessing.")
    print(
        "   - 'AUC-PR > baseline' means your model is better than just guessing 'yes'"
    )
    print("     for the most common class. This is the *key metric*.")

    print("\n2. Prediction Distribution (plot):")
    print(
        "   - LOOK FOR: A clear separation between the blue (0s) and red (1s) distributions."
    )
    print(
        "   - MEANING: A good model predicts low scores for 0s and high scores for 1s."
    )

    print("\n3. Embedding by group (plot):")
    print("   - LOOK FOR: Clear clusters of colors.")
    print("   - MEANING: The model's learned factors (dimensions) are able to separate")
    print("     the different ecological groups.")

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
