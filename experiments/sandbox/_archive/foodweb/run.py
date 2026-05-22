#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grand Caricaie Marsh Food Web Analysis using Symmetric NMF.
"""

from datetime import datetime
from pathlib import Path
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from pysrf import SRF
from analyses.ppi.graph_utils import build_adjacency_with_nan
from analyses.ppi.metrics import compute_link_prediction_metrics


def load_gateway_data(
    data_dir: Path,
) -> tuple[nx.Graph, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load preprocessed Gateway food web data."""
    adj_sym_file = data_dir / "adjacency_symmetric.npy"
    adj_dir_file = data_dir / "adjacency_directed.npy"
    species_file = data_dir / "species.csv"

    if not adj_sym_file.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_sym_file}")
    if not adj_dir_file.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_dir_file}")
    if not species_file.exists():
        raise FileNotFoundError(f"Species file not found: {species_file}")

    adj_matrix = np.load(adj_sym_file).astype(np.float32)
    adj_directed = np.load(adj_dir_file).astype(np.float32)
    species_df = pd.read_csv(species_file)

    n = adj_matrix.shape[0]
    print(f"Loaded adjacency matrix: {n} × {n}")
    print(f"Edges: {int(adj_matrix.sum() / 2)}")
    print(f"Density: {adj_matrix.sum() / (n * n):.4f}")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)

    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, adj_matrix, adj_directed, species_df


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


def compute_trophic_levels(
    adj_matrix: np.ndarray, species_df: pd.DataFrame, max_iter: int = 100
) -> np.ndarray:
    """Compute trophic level using Lindeman formula: TL = 1 + mean(prey TL)."""
    n = len(species_df)
    levels = np.full(n, np.nan)

    out_degree = species_df["out_degree"].values
    basal_mask = out_degree == 0
    levels[basal_mask] = 1.0

    for iteration in range(max_iter):
        changed = False
        for idx in range(n):
            if not np.isnan(levels[idx]):
                continue

            prey_indices = np.where(adj_matrix[idx, :] > 0)[0]
            if len(prey_indices) == 0:
                continue

            prey_levels = levels[prey_indices]
            if np.all(~np.isnan(prey_levels)):
                levels[idx] = 1.0 + np.mean(prey_levels)
                changed = True

        if not changed:
            break

    levels[np.isnan(levels)] = 1.0

    return levels


def run_link_prediction_cv(
    adj_binary: np.ndarray,
    rank: int,
    n_repeats: int = 5,
    sampling_fraction: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """Run link prediction cross-validation with explicit positive/negative pairs."""
    print(
        f"\nRunning link prediction CV (n_repeats={n_repeats}, sampling_fraction={sampling_fraction})..."
    )

    n = adj_binary.shape[0]
    nodes = list(range(n))

    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_mask = adj_binary[triu_i, triu_j] > 0

    positives = np.column_stack((triu_i[edge_mask], triu_j[edge_mask]))
    negatives = np.column_stack((triu_i[~edge_mask], triu_j[~edge_mask]))

    validation_fraction = 1.0 - sampling_fraction
    n_validation = max(1, int(np.round(len(positives) * validation_fraction)))

    rng = np.random.RandomState(random_state)

    auc_values: list[float] = []
    ap_values: list[float] = []

    for split_idx in range(n_repeats):
        val_indices = rng.choice(len(positives), size=n_validation, replace=False)
        val_edges = positives[val_indices]

        train_mask = np.ones(len(positives), dtype=bool)
        train_mask[val_indices] = False
        train_edges = positives[train_mask]

        neg_indices = rng.choice(len(negatives), size=n_validation, replace=False)
        neg_edges = negatives[neg_indices]

        g_train = nx.Graph()
        g_train.add_nodes_from(nodes)
        g_train.add_edges_from((int(u), int(v)) for u, v in train_edges)

        g_full = nx.Graph()
        g_full.add_nodes_from(nodes)
        g_full.add_edges_from((int(u), int(v)) for u, v in positives)

        adj_train = build_adjacency_with_nan(g_train, g_full, nodes).astype(np.float32)

        model = SRF(
            rank=rank,
            rho=3.0,
            bounds=(0, 1),
            max_outer=2000,
            max_inner=50,
            tol=1e-4,
            verbose=0,
            init="random_sqrt",
            random_state=random_state + split_idx,
            missing_values=np.nan,
            loss="frobenius",
        )

        embedding = model.fit_transform(adj_train)

        test_pairs = np.vstack([val_edges, neg_edges])
        test_labels = np.concatenate(
            [
                np.ones(len(val_edges), dtype=np.int32),
                np.zeros(len(neg_edges), dtype=np.int32),
            ]
        )

        scores = np.sum(
            embedding[test_pairs[:, 0]] * embedding[test_pairs[:, 1]], axis=1
        )

        metrics = compute_link_prediction_metrics(scores, test_labels)

        auc_values.append(metrics["auroc"])
        ap_values.append(metrics["auprc"])

    auc_mean = float(np.mean(auc_values)) if auc_values else 0.0
    auc_std = float(np.std(auc_values)) if auc_values else 0.0
    ap_mean = float(np.mean(ap_values)) if ap_values else 0.0
    ap_std = float(np.std(ap_values)) if ap_values else 0.0

    print(f"Link prediction AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"Link prediction AP: {ap_mean:.3f} ± {ap_std:.3f}")

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_values": auc_values,
        "ap_mean": ap_mean,
        "ap_std": ap_std,
        "ap_values": ap_values,
    }


def main() -> None:
    data_dir = Path("data/gateway/grand_caricaie")
    rank = 15
    normalize_by_degree = True

    print("=" * 80)
    print("GRAND CARICAIE MARSH FOOD WEB - SYMMETRIC NMF ANALYSIS")
    print(f"Rank: {rank}, Degree normalization: {normalize_by_degree}")
    print("=" * 80)

    print("\nLoading network...")
    G, adj_matrix_raw, adj_directed, species_df = load_gateway_data(data_dir)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"Network: {n_nodes} species, {n_edges} interactions")

    print("\nComputing trophic levels...")
    trophic_array = compute_trophic_levels(adj_directed, species_df)
    print(f"Trophic range: {trophic_array.min():.2f} - {trophic_array.max():.2f}")
    n_basal = np.sum(trophic_array == 1.0)
    print(f"Basal species: {n_basal}, Max TL = {trophic_array.max():.2f}")

    if normalize_by_degree:
        print("\nApplying degree normalization (symmetric normalized Laplacian)...")
        degrees = adj_matrix_raw.sum(axis=0) + adj_matrix_raw.sum(axis=1)
        degrees = np.maximum(degrees, 1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        adj_normalized = D_inv_sqrt @ adj_matrix_raw @ D_inv_sqrt
        adj = create_adjacency_matrix(adj_normalized, binary=False)
        print(
            f"Normalization applied: degree range {degrees.min():.0f}-{degrees.max():.0f}"
        )
    else:
        adj = create_adjacency_matrix(adj_matrix_raw, binary=True)

    adj_norm = adj.copy()
    valid_mask = ~np.isnan(adj)
    if valid_mask.any():
        min_val = adj[valid_mask].min()
        max_val = adj[valid_mask].max()
        if max_val > min_val:
            adj_norm[valid_mask] = (adj[valid_mask] - min_val) / (max_val - min_val)

    output_dir = Path(
        "experiments/development/foodweb/outputs"
    ) / datetime.now().strftime("%y%m%d/%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFitting SRF (rank={rank})...")
    model = SRF(
        rank=rank,
        rho=3.0,
        bounds=(0, 1),
        max_outer=2000,
        max_inner=50,
        tol=1e-4,
        verbose=1,
        init="random_sqrt",
        random_state=42,
        missing_values=np.nan,
        loss="frobenius",
    )

    W = model.fit_transform(adj_norm)
    reconstruction = model.reconstruct()

    if max_val > min_val:
        reconstruction = reconstruction * (max_val - min_val) + min_val

    print(f"\nComplete: {model.n_iter_} iterations")

    np.save(output_dir / "embedding.npy", W)
    np.save(output_dir / "reconstruction.npy", reconstruction)

    link_pred_stats = run_link_prediction_cv(adj_matrix_raw, rank)

    with open(output_dir / "link_prediction_results.pkl", "wb") as f:
        pickle.dump(link_pred_stats, f)

    species_with_factors = species_df.copy()
    for dim in range(W.shape[1]):
        species_with_factors[f"factor_{dim}"] = W[:, dim]
    species_with_factors["trophic_level"] = trophic_array
    species_with_factors.to_csv(output_dir / "species_with_factors.csv", index=False)

    observed_mask = ~np.isnan(adj_norm)
    observed_true = adj_norm[observed_mask]
    observed_pred = reconstruction[observed_mask]

    if np.std(observed_true) > 0 and np.std(observed_pred) > 0:
        corr = np.corrcoef(observed_true, observed_pred)[0, 1]
    else:
        corr = 0.0

    residuals = observed_pred - observed_true
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    explained_variance = (
        1.0 - (np.var(residuals) / np.var(observed_true))
        if np.var(observed_true) > 0
        else 0.0
    )

    stats = {
        "metric": [
            "n_species",
            "n_factors",
            "explained_variance",
            "reconstruction_rmse",
            "reconstruction_mae",
            "reconstruction_correlation",
            "link_prediction_auc_mean",
            "link_prediction_auc_std",
            "link_prediction_ap_mean",
            "link_prediction_ap_std",
        ],
        "value": [
            n_nodes,
            rank,
            explained_variance,
            rmse,
            mae,
            corr,
            link_pred_stats["auc_mean"],
            link_pred_stats["auc_std"],
            link_pred_stats["ap_mean"],
            link_pred_stats["ap_std"],
        ],
    }

    pd.DataFrame(stats).to_csv(output_dir / "statistics.csv", index=False)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nEmbedding: {n_nodes} species × {rank} dimensions")
    print(
        f"Link Prediction AUC: {link_pred_stats['auc_mean']:.3f} ± {link_pred_stats['auc_std']:.3f}"
    )
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
