#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caribbean Reef Food Web Analysis using Symmetric NMF.

Focus: Understanding what each latent dimension represents ecologically.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pysrf import SRF


def load_weboflife_network(
    network_id: str, data_dir: Path
) -> tuple[nx.DiGraph, np.ndarray]:
    """Load a food web adjacency matrix from Web of Life format."""
    network_file = data_dir / f"{network_id}.csv"

    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")

    raw_matrix = pd.read_csv(network_file, header=None).values.astype(np.float32)
    n_prey, n_pred = raw_matrix.shape

    print(f"Loaded raw matrix: {n_prey} rows (prey) × {n_pred} columns (predators)")

    n = max(n_prey, n_pred)
    adj_matrix = np.zeros((n, n), dtype=np.float32)
    adj_matrix[:n_prey, :n_pred] = raw_matrix

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for prey in range(n):
        for predator in range(n):
            if adj_matrix[prey, predator] > 0:
                G.add_edge(prey, predator, weight=adj_matrix[prey, predator])

    isolated = list(nx.isolates(G))
    if isolated:
        print(f"Removing {len(isolated)} isolated nodes")
        G.remove_nodes_from(isolated)
        mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)
        active_nodes = sorted([node for node in range(n) if node not in isolated])
        adj_matrix = adj_matrix[np.ix_(active_nodes, active_nodes)]

    return G, adj_matrix


def create_adjacency_matrix(adj_matrix: np.ndarray, binary: bool = False) -> np.ndarray:
    """Convert adjacency matrix to NaN format for SRF."""
    n = adj_matrix.shape[0]
    adj = np.full((n, n), np.nan, dtype=np.float32)

    mask = adj_matrix > 0
    if binary:
        adj[mask] = 1.0
    else:
        adj[mask] = adj_matrix[mask]

    adj = np.fmax(adj, adj.T)
    np.fill_diagonal(adj, np.nan)

    return adj


def compute_trophic_levels(G: nx.DiGraph) -> dict[int, float]:
    """Compute trophic level for each species."""
    levels = {}

    for node in G.nodes():
        if G.in_degree(node) == 0:
            levels[node] = 1.0

    for _ in range(100):
        changed = False
        for node in G.nodes():
            if node not in levels:
                prey = list(G.predecessors(node))
                if prey and all(p in levels for p in prey):
                    levels[node] = 1.0 + np.mean([levels[p] for p in prey])
                    changed = True
        if not changed:
            break

    for node in G.nodes():
        if node not in levels:
            levels[node] = 0.0

    return levels


def plot_top_species_per_dimension(
    W: np.ndarray, trophic_levels: dict[int, float], output_path: Path, top_k: int = 10
) -> pd.DataFrame:
    """Show which species load highest on each dimension."""
    n_species, n_dims = W.shape

    results = []
    for dim in range(n_dims):
        loadings = W[:, dim]
        top_indices = np.argsort(loadings)[::-1][:top_k]

        for rank, species_id in enumerate(top_indices, 1):
            results.append(
                {
                    "dimension": dim,
                    "rank": rank,
                    "species_id": species_id,
                    "loading": loadings[species_id],
                    "trophic_level": trophic_levels[species_id],
                }
            )

    df = pd.DataFrame(results)

    # Plot
    n_cols = min(4, n_dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for dim in range(n_dims):
        ax = axes[dim]
        dim_data = df[df["dimension"] == dim].sort_values("rank")

        bars = ax.barh(
            range(top_k),
            dim_data["loading"].values,
            color=plt.cm.viridis(
                dim_data["trophic_level"].values / dim_data["trophic_level"].max()
            ),
        )
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"Sp {int(s)}" for s in dim_data["species_id"].values])
        ax.invert_yaxis()
        ax.set_xlabel("Loading")
        ax.set_title(f"Dimension {dim}")
        ax.grid(alpha=0.3, axis="x")

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df


def plot_dimension_summary(
    W: np.ndarray, trophic_levels: dict[int, float], output_path: Path
) -> None:
    """Summary statistics for each dimension."""
    n_species, n_dims = W.shape
    trophic_array = np.array([trophic_levels[i] for i in range(n_species)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Mean loading per dimension
    mean_loadings = W.mean(axis=0)
    axes[0, 0].bar(range(n_dims), mean_loadings, color="steelblue", alpha=0.7)
    axes[0, 0].set_xlabel("Dimension")
    axes[0, 0].set_ylabel("Mean loading")
    axes[0, 0].set_title("Average importance per dimension")
    axes[0, 0].set_xticks(range(n_dims))

    # Top-right: Sparsity (% near-zero) per dimension
    sparsity = (W < 0.01).sum(axis=0) / n_species * 100
    axes[0, 1].bar(range(n_dims), sparsity, color="coral", alpha=0.7)
    axes[0, 1].set_xlabel("Dimension")
    axes[0, 1].set_ylabel("% species with loading < 0.01")
    axes[0, 1].set_title("Dimension sparsity")
    axes[0, 1].set_xticks(range(n_dims))

    # Bottom-left: Correlation with trophic level
    correlations = []
    for dim in range(n_dims):
        rho, _ = spearmanr(W[:, dim], trophic_array)
        correlations.append(rho)

    colors = ["green" if abs(r) > 0.3 else "gray" for r in correlations]
    axes[1, 0].bar(range(n_dims), correlations, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Dimension")
    axes[1, 0].set_ylabel("Spearman correlation")
    axes[1, 0].set_title("Correlation with trophic level")
    axes[1, 0].set_xticks(range(n_dims))

    # Bottom-right: Heatmap of dimension loadings (top 30 species)
    top_species = np.argsort(W.sum(axis=1))[::-1][:30]
    W_subset = W[top_species, :]

    im = axes[1, 1].imshow(W_subset.T, cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_xlabel("Species (top 30 by total loading)")
    axes[1, 1].set_ylabel("Dimension")
    axes[1, 1].set_title("Loading heatmap")
    axes[1, 1].set_yticks(range(n_dims))
    plt.colorbar(im, ax=axes[1, 1], label="Loading")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reconstruction_quality(
    adj: np.ndarray, reconstruction: np.ndarray, output_path: Path
) -> None:
    """Visualize how well the model reconstructs observed interactions."""
    observed_mask = ~np.isnan(adj)

    observed_true = adj[observed_mask]
    observed_pred = reconstruction[observed_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Scatter plot
    axes[0].scatter(observed_true, observed_pred, alpha=0.3, s=20, edgecolors="none")

    # Add diagonal line
    min_val = min(observed_true.min(), observed_pred.min())
    max_val = max(observed_true.max(), observed_pred.max())
    axes[0].plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit"
    )

    axes[0].set_xlabel("True interaction strength")
    axes[0].set_ylabel("Reconstructed interaction strength")
    axes[0].set_title("Reconstruction accuracy on observed interactions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Add correlation text
    corr = np.corrcoef(observed_true, observed_pred)[0, 1]
    axes[0].text(
        0.05,
        0.95,
        f"Pearson r = {corr:.3f}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Right: Residuals
    residuals = observed_pred - observed_true
    axes[1].hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual (predicted - true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Reconstruction errors")
    axes[1].grid(alpha=0.3, axis="y")

    # Add statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    axes[1].text(
        0.95,
        0.95,
        f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    network_id = "FW_008"
    data_dir = Path("data/weboflife")
    rank = 8

    print("=" * 80)
    print("CARIBBEAN REEF FOOD WEB - SYMMETRIC NMF ANALYSIS")
    print("=" * 80)

    # Load
    print("\nLoading network...")
    G, adj_matrix_raw = load_weboflife_network(network_id, data_dir)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"Network: {n_nodes} species, {n_edges} interactions")

    # Check if weighted
    weights = adj_matrix_raw[adj_matrix_raw > 0]
    is_weighted = len(np.unique(weights)) > 1
    if is_weighted:
        print(f"Weighted network: [{weights.min():.4f}, {weights.max():.4f}]")

    # Trophic levels
    print("\nComputing trophic levels...")
    trophic_levels = compute_trophic_levels(G)
    trophic_array = np.array([trophic_levels[i] for i in range(n_nodes)])
    print(f"Trophic range: {trophic_array.min():.2f} - {trophic_array.max():.2f}")

    # Create matrix
    adj = create_adjacency_matrix(adj_matrix_raw, binary=False)

    # Normalize for SRF
    adj_norm = adj.copy()
    valid_mask = ~np.isnan(adj)
    if valid_mask.any():
        min_val = adj[valid_mask].min()
        max_val = adj[valid_mask].max()
        if max_val > min_val:
            adj_norm[valid_mask] = (adj[valid_mask] - min_val) / (max_val - min_val)

    # Output directory
    output_dir = Path(
        "experiments/development/caribbean/outputs"
    ) / datetime.now().strftime("%y%m%d/%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit SRF
    print(f"\nFitting SRF (rank={rank})...")
    model = SRF(
        rank=rank,
        rho=3.0,
        bounds=(0, 1),
        max_outer=1000,
        max_inner=30,
        tol=1e-4,
        verbose=1,
        init="random_sqrt",
        random_state=42,
        missing_values=np.nan,
        loss="frobenius",
    )

    W = model.fit_transform(adj_norm)
    reconstruction = model.reconstruct()

    # Rescale back
    if max_val > min_val:
        reconstruction = reconstruction * (max_val - min_val) + min_val

    print(f"\nComplete: {model.n_iter_} iterations")

    # Save
    np.save(output_dir / "embedding.npy", W)
    np.save(output_dir / "reconstruction.npy", reconstruction)

    pd.DataFrame(
        {
            "species_id": range(n_nodes),
            "trophic_level": [trophic_levels[i] for i in range(n_nodes)],
        }
    ).to_csv(output_dir / "species_info.csv", index=False)

    # Create plots
    print("\nCreating visualizations...")

    print("  1. Top species per dimension...")
    top_species_df = plot_top_species_per_dimension(
        W, trophic_levels, output_dir / "top_species_per_dimension.png", top_k=10
    )
    top_species_df.to_csv(output_dir / "top_species_per_dimension.csv", index=False)

    print("  2. Dimension summary...")
    plot_dimension_summary(W, trophic_levels, output_dir / "dimension_summary.png")

    print("  3. Reconstruction quality...")
    plot_reconstruction_quality(
        adj_norm, reconstruction, output_dir / "reconstruction_quality.png"
    )

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nEmbedding: {n_nodes} species × {rank} dimensions")
    print(f"Output: {output_dir}")

    # Interpretation hints
    print("\nINTERPRETATION GUIDE:")
    print("1. Top species per dimension: Which species define each latent niche?")
    print("2. Dimension summary: How do dimensions relate to trophic structure?")
    print(
        "3. Reconstruction quality: How well does rank-{} capture interactions?".format(
            rank
        )
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
