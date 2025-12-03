#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Serengeti food web analysis: Demonstrating that SRF dimensions are useful.

Goal: Show that learned dimensions capture meaningful biological structure
by predicting external metadata (trophic levels, groups, habitats).
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

try:
    from pysrf import SRF

    SRF_AVAILABLE = True
except ImportError:
    SRF_AVAILABLE = False
    print("Error: pysrf not available")
    exit(1)


def load_serengeti_data(
    data_dir: Path,
) -> tuple[nx.DiGraph, pd.DataFrame, dict[str, int]]:
    """Load Serengeti food web data."""
    species_path = data_dir / "species.csv"
    links_path = data_dir / "feeding-links.csv"

    if not species_path.exists() or not links_path.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    species_df = pd.read_csv(species_path, comment="%")
    links_df = pd.read_csv(links_path, comment="%")

    code_to_idx = {code: idx for idx, code in enumerate(species_df["code"].values)}
    n_species = len(species_df)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_species))

    for _, row in links_df.iterrows():
        pred_code = row["pred_code"]
        prey_code = row["prey_code"]
        if pred_code in code_to_idx and prey_code in code_to_idx:
            pred_idx = code_to_idx[pred_code]
            prey_idx = code_to_idx[prey_code]
            G.add_edge(prey_idx, pred_idx)

    print(f"Loaded {n_species} species, {G.number_of_edges()} links")
    return G, species_df, code_to_idx


def load_consensus_partition(data_dir: Path, code_to_idx: dict[str, int]) -> np.ndarray:
    """Load consensus partition group IDs."""
    partition_path = data_dir / "consensus-partition.csv"
    if not partition_path.exists():
        raise FileNotFoundError(f"Consensus partition not found: {partition_path}")

    partition_df = pd.read_csv(partition_path)
    n_nodes = len(code_to_idx)
    group_ids = np.full(n_nodes, -1, dtype=int)

    for _, row in partition_df.iterrows():
        code = row["code"]
        group = int(row["group"])
        if code in code_to_idx:
            idx = code_to_idx[code]
            group_ids[idx] = group

    valid_mask = group_ids >= 0
    print(
        f"Loaded partition: {valid_mask.sum()}/{n_nodes} species in {len(np.unique(group_ids[valid_mask]))} groups"
    )
    return group_ids


def create_adjacency_with_missing(G: nx.DiGraph, n_nodes: int) -> np.ndarray:
    """Create symmetric adjacency matrix with NaN for missing values."""
    adj = np.full((n_nodes, n_nodes), np.nan, dtype=float)

    for source, target in G.edges():
        if source < n_nodes and target < n_nodes:
            adj[source, target] = 1.0

    adj_sym = np.fmax(adj, adj.T)
    np.fill_diagonal(adj_sym, np.nan)
    return adj_sym


def compute_trophic_levels(G: nx.DiGraph) -> np.ndarray:
    """Compute trophic levels using bottom-up algorithm."""
    n_nodes = len(G.nodes())
    levels = {}

    for node in G.nodes():
        if G.in_degree(node) == 0:
            levels[node] = 1.0

    changed = True
    max_iter = 100
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        for node in G.nodes():
            if node not in levels:
                predecessors = list(G.predecessors(node))
                if all(p in levels for p in predecessors):
                    if predecessors:
                        levels[node] = 1.0 + np.mean([levels[p] for p in predecessors])
                    else:
                        levels[node] = 1.0
                    changed = True

    trophic_array = np.array([levels.get(i, 1.0) for i in range(n_nodes)])
    return trophic_array


def infer_feeding_types(G: nx.DiGraph, trophic_levels: np.ndarray) -> np.ndarray:
    """Infer feeding type from network structure."""
    feeding_types = []
    n_nodes = len(G.nodes())

    for i in range(n_nodes):
        in_degree = G.in_degree(i)
        out_degree = G.out_degree(i)
        trophic_level = trophic_levels[i]

        if trophic_level <= 1.5:
            feeding_types.append("plant")
        elif out_degree == 0 and trophic_level > 2.5:
            feeding_types.append("carnivore")
        elif trophic_level <= 2.5:
            feeding_types.append("herbivore")
        else:
            feeding_types.append("omnivore")

    return np.array(feeding_types)


def predict_trophic_level(
    X: np.ndarray, y: np.ndarray, cv_folds: int = 5
) -> dict[str, Any]:
    """Predict trophic level using regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=42)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error"
    )

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "cv_mae": -cv_scores.mean(),
        "model": model,
    }


def predict_groups(
    X: np.ndarray, group_ids: np.ndarray, cv_folds: int = 5
) -> dict[str, Any]:
    """Predict consensus partition group membership."""
    valid_mask = group_ids >= 0
    X_valid = X[valid_mask]
    y_valid = group_ids[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))

    cv_scores = cross_val_score(
        model, X_scaled, y_valid, cv=cv_folds, scoring="accuracy"
    )

    model.fit(X_scaled, y_valid)
    y_pred = model.predict(X_scaled)

    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1_macro": f1_score(y_valid, y_pred, average="macro", zero_division=0),
        "cv_accuracy": cv_scores.mean(),
    }


def plot_dimension_trophic_correlation(
    W: np.ndarray, trophic_levels: np.ndarray, output_path: Path
) -> pd.DataFrame:
    """Plot correlation between dimensions and trophic level."""
    n_dims = W.shape[1]
    correlations = []

    for dim in range(n_dims):
        rho, p_value = spearmanr(W[:, dim], trophic_levels)
        correlations.append({"dimension": dim, "spearman_rho": rho, "p_value": p_value})

    df = pd.DataFrame(correlations)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red" if p < 0.05 else "gray" for p in df["p_value"]]
    ax.barh(df["dimension"], df["spearman_rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman correlation with trophic level")
    ax.set_ylabel("Dimension")
    ax.set_title("Dimension-trophic level correlations")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return df


def plot_dimension_group_heatmap(
    W: np.ndarray, group_ids: np.ndarray, output_path: Path
) -> None:
    """Plot mean dimension loadings by consensus group."""
    valid_mask = group_ids >= 0
    W_valid = W[valid_mask]
    groups_valid = group_ids[valid_mask]

    unique_groups = sorted(np.unique(groups_valid))
    n_dims = W.shape[1]

    heatmap_data = np.zeros((n_dims, len(unique_groups)))
    for i, group in enumerate(unique_groups):
        group_mask = groups_valid == group
        heatmap_data[:, i] = W_valid[group_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels([f"G{g}" for g in unique_groups])
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels([f"D{i}" for i in range(n_dims)])
    ax.set_xlabel("Consensus group")
    ax.set_ylabel("Dimension")
    ax.set_title("Mean dimension loadings by consensus group")

    plt.colorbar(im, ax=ax, label="Mean loading")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_loading_species(
    W: np.ndarray,
    species_df: pd.DataFrame,
    feeding_types: np.ndarray,
    group_ids: np.ndarray,
    trophic_levels: np.ndarray,
    output_path: Path,
    top_n_dims: int = 6,
    top_k: int = 15,
) -> None:
    """Plot top-loading species for most correlated dimensions."""
    correlations = []
    for dim in range(W.shape[1]):
        rho, _ = spearmanr(W[:, dim], trophic_levels)
        correlations.append((dim, abs(rho)))

    correlations.sort(key=lambda x: x[1], reverse=True)
    top_dims = [d for d, _ in correlations[:top_n_dims]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    feeding_colors = {
        "plant": "green",
        "herbivore": "orange",
        "carnivore": "red",
        "omnivore": "purple",
    }

    for plot_idx, dim in enumerate(top_dims):
        ax = axes[plot_idx]
        loadings = W[:, dim]
        top_indices = np.argsort(loadings)[::-1][:top_k]

        top_species = species_df.iloc[top_indices]["code"].values
        top_loadings = loadings[top_indices]
        top_types = feeding_types[top_indices]
        top_groups = group_ids[top_indices]

        colors = [feeding_colors.get(ft, "gray") for ft in top_types]

        y_pos = np.arange(len(top_species))
        ax.barh(y_pos, top_loadings, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [
                f"{sp} (G{g})" if g >= 0 else sp
                for sp, g in zip(top_species, top_groups)
            ],
            fontsize=8,
        )
        ax.set_xlabel("Loading")
        ax.set_title(f"Dimension {dim}")
        ax.grid(alpha=0.3, axis="x")

    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=4, label=ft.capitalize())
        for ft, c in feeding_colors.items()
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    data_dir = Path("data/serengeti")
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} not found")
        return

    print("=" * 60)
    print("SERENGETI FOOD WEB: SRF Dimension Utility")
    print("=" * 60)

    print("\nLoading data...")
    G, species_df, code_to_idx = load_serengeti_data(data_dir)
    n_nodes = len(G.nodes())
    group_ids = load_consensus_partition(data_dir, code_to_idx)

    print("\nCreating symmetric adjacency matrix...")
    adj = create_adjacency_with_missing(G, n_nodes)
    observed_mask = ~np.isnan(adj)
    n_observed = observed_mask.sum()
    print(
        f"Matrix: {adj.shape}, {n_observed} observed links ({100*n_observed/(n_nodes*(n_nodes-1)):.2f}% density)"
    )

    print("\nComputing metadata...")
    trophic_levels = compute_trophic_levels(G)
    feeding_types = infer_feeding_types(G, trophic_levels)
    print(f"Trophic range: {trophic_levels.min():.2f} - {trophic_levels.max():.2f}")
    for ft in np.unique(feeding_types):
        print(f"  {ft}: {(feeding_types == ft).sum()}")

    output_dir = Path(
        "experiments/development/foodweb/outputs"
    ) / datetime.now().strftime("%y%m%d/%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    print("\n" + "=" * 60)
    print("FITTING SRF MODEL")
    print("=" * 60)

    rank = 16
    model = SRF(
        rank=rank,
        rho=3.0,
        bounds=(0.0, 1.0),
        max_outer=1000,
        max_inner=30,
        tol=1e-4,
        verbose=1,
        init="random_sqrt",
        random_state=42,
        missing_values=np.nan,
        loss="frobenius",
    )

    W = model.fit_transform(adj)
    print(f"\nFactorization complete: W shape = {W.shape}")
    print(f"W range: [{W.min():.4f}, {W.max():.4f}], mean: {W.mean():.4f}")

    print("\n" + "=" * 60)
    print("PREDICTION TASKS")
    print("=" * 60)

    print("\n1. Trophic Level Prediction (Regression)")
    reg_srf = predict_trophic_level(W, trophic_levels)
    print(f"   R²: {reg_srf['r2']:.4f}, MAE: {reg_srf['mae']:.4f}")

    np.random.seed(42)
    W_random = np.random.randn(n_nodes, rank)
    reg_random = predict_trophic_level(W_random, trophic_levels)
    print(f"   Random R²: {reg_random['r2']:.4f}, MAE: {reg_random['mae']:.4f}")
    print(
        f"   → SRF is {100*(reg_srf['r2']-reg_random['r2'])/abs(reg_random['r2']):.1f}% better"
    )

    print("\n2. Consensus Group Prediction (14 groups)")
    group_srf = predict_groups(W, group_ids)
    print(f"   Accuracy: {group_srf['accuracy']:.4f}, F1: {group_srf['f1_macro']:.4f}")

    group_random = predict_groups(W_random, group_ids)
    print(
        f"   Random Accuracy: {group_random['accuracy']:.4f}, F1: {group_random['f1_macro']:.4f}"
    )
    print(
        f"   → SRF is {100*(group_srf['accuracy']-group_random['accuracy'])/group_random['accuracy']:.1f}% better"
    )

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    print("\n1. Dimension-trophic correlations...")
    corr_df = plot_dimension_trophic_correlation(
        W, trophic_levels, output_dir / "dimension_trophic_correlations.png"
    )
    corr_df.to_csv(output_dir / "dimension_trophic_correlations.csv", index=False)

    print("2. Dimension-group heatmap...")
    plot_dimension_group_heatmap(
        W, group_ids, output_dir / "dimension_group_heatmap.png"
    )

    print("3. Top-loading species (top 6 dimensions)...")
    plot_top_loading_species(
        W,
        species_df,
        feeding_types,
        group_ids,
        trophic_levels,
        output_dir / "top_loading_species.png",
        top_n_dims=6,
    )

    np.save(output_dir / "W.npy", W)
    np.save(output_dir / "trophic_levels.npy", trophic_levels)

    results_df = pd.DataFrame(
        {
            "method": ["SRF", "Random"],
            "trophic_r2": [reg_srf["r2"], reg_random["r2"]],
            "trophic_mae": [reg_srf["mae"], reg_random["mae"]],
            "group_accuracy": [group_srf["accuracy"], group_random["accuracy"]],
            "group_f1": [group_srf["f1_macro"], group_random["f1_macro"]],
        }
    )
    results_df.to_csv(output_dir / "summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSRF dimensions encode meaningful biological structure:")
    print(
        f"  • Trophic levels: R² = {reg_srf['r2']:.3f} ({100*(reg_srf['r2']-reg_random['r2'])/abs(reg_random['r2']):.0f}% > random)"
    )
    print(
        f"  • Consensus groups: Acc = {group_srf['accuracy']:.3f} ({100*(group_srf['accuracy']-group_random['accuracy'])/group_random['accuracy']:.0f}% > random)"
    )
    print(
        f"  • Top correlated dimensions: {', '.join(map(str, corr_df.nlargest(3, 'spearman_rho')['dimension'].values))}"
    )
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
