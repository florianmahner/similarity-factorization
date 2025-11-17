import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import json


def kfold_cv(g: nx.Graph, n_folds: int, seed: int) -> list[tuple[nx.Graph, nx.Graph]]:
    """Create k-fold cross-validation splits.
    
    Parameters
    ----------
    g : nx.Graph
        Full graph
    n_folds : int
        Number of folds
    seed : int
        Random seed
    
    Returns
    -------
    list[tuple[nx.Graph, nx.Graph]]
        List of (g_train, g_full) tuples for each fold
    """
    edges = np.array(list(g.edges()))
    rng = np.random.RandomState(seed)
    rng.shuffle(edges)
    
    fold_size = len(edges) // n_folds
    folds = []
    
    for i in range(n_folds):
        test_edges = edges[i * fold_size : (i + 1) * fold_size]
        train_edges = np.concatenate(
            [edges[: i * fold_size], edges[(i + 1) * fold_size :]]
        )
        
        g_train = nx.Graph()
        g_train.add_nodes_from(g.nodes())
        g_train.add_edges_from([tuple(e) for e in train_edges])
        
        if any("weight" in d for _, _, d in g.edges(data=True)):
            for u, v in train_edges:
                if g.has_edge(u, v):
                    g_train[u][v]["weight"] = g[u][v]["weight"]
        
        folds.append((g_train, g))
    
    return folds


def build_test_set(
    g_train: nx.Graph, g_full: nx.Graph
) -> tuple[np.ndarray, np.ndarray]:
    """Build test set of all node pairs not in training set.
    
    Parameters
    ----------
    g_train : nx.Graph
        Training graph
    g_full : nx.Graph
        Full graph
    
    Returns
    -------
    test_pairs : np.ndarray
        Test pairs (n_pairs, 2) with node indices
    test_labels : np.ndarray
        Binary labels (1 for positive, 0 for negative)
    """
    all_nodes = sorted(g_full.nodes())
    n = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    train_edges = {
        tuple(sorted([node_to_idx[u], node_to_idx[v]])) for u, v in g_train.edges()
    }
    full_edges = {
        tuple(sorted([node_to_idx[u], node_to_idx[v]])) for u, v in g_full.edges()
    }
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    all_pairs = set(zip(i_idx, j_idx))
    
    test_pairs = np.array(sorted(all_pairs - train_edges))
    test_labels = np.array(
        [1 if tuple(p) in full_edges else 0 for p in test_pairs], dtype=np.int8
    )
    
    return test_pairs, test_labels


def load_splits_from_csv(splits_dir: Path, fold_idx: int) -> dict:
    """Load k-fold split data from CSV files.
    
    Parameters
    ----------
    splits_dir : Path
        Directory containing split CSV files
    fold_idx : int
        Fold index to load
    
    Returns
    -------
    dict
        Dictionary with keys:
        - train_edges: list of (source, target) tuples
        - test_edges: list of (source, target) tuples (positive only)
        - test_pairs: np.ndarray (n_pairs, 2) with node indices
        - test_labels: np.ndarray (n_pairs,) with binary labels
        - nodes: list of node IDs (sorted)
    """
    nodes_df = pd.read_csv(splits_dir / "nodes.csv")
    nodes = nodes_df["node_id"].tolist()
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    train_edges_df = pd.read_csv(splits_dir / f"fold{fold_idx}_train_edges.csv")
    train_edges = [
        (row["source"], row["target"]) 
        for _, row in train_edges_df.iterrows()
        if row["source"] != row["target"]  # Filter self-loops
    ]
    
    test_edges_df = pd.read_csv(splits_dir / f"fold{fold_idx}_test_edges.csv")
    test_edges = [
        (row["source"], row["target"]) 
        for _, row in test_edges_df.iterrows()
        if row["source"] != row["target"]  # Filter self-loops
    ]
    
    test_pairs_df = pd.read_csv(splits_dir / f"fold{fold_idx}_test_pairs.csv")
    test_pairs = test_pairs_df[["node_i", "node_j"]].values
    test_labels = test_pairs_df["label"].values
    
    # Filter self-loops from test_pairs
    valid_mask = test_pairs[:, 0] != test_pairs[:, 1]
    test_pairs = test_pairs[valid_mask]
    test_labels = test_labels[valid_mask]
    
    return {
        "train_edges": train_edges,
        "test_edges": test_edges,
        "test_pairs": test_pairs,
        "test_labels": test_labels,
        "nodes": nodes,
        "node_to_idx": node_to_idx,
    }

