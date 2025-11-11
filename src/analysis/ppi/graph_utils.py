import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from pathlib import Path


def load_network(csv_path: Path | str) -> tuple[nx.Graph, bool]:
    """Load network from CSV file.

    Parameters
    ----------
    csv_path : Path or str
        Path to CSV file with source, target columns (optionally weight)

    Returns
    -------
    g : nx.Graph
        NetworkX graph (largest connected component)
    has_weights : bool
        Whether the graph has edge weights
    """
    df = pd.read_csv(csv_path)

    # has_weights = "weight" in df.columns
    print("Disabled weighting for STRING data")
    has_weights = False

    if has_weights:
        df["u"] = np.minimum(df["source"], df["target"])
        df["v"] = np.maximum(df["source"], df["target"])
        df = df.groupby(["u", "v"], as_index=False)["weight"].max()

        weights = df["weight"].values
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())

        g = nx.Graph()
        g.add_weighted_edges_from(zip(df["u"], df["v"], weights))
    else:
        df["u"] = np.minimum(df["source"], df["target"])
        df["v"] = np.maximum(df["source"], df["target"])
        df = df.drop_duplicates(subset=["u", "v"])
        g = nx.from_pandas_edgelist(df, "u", "v")

    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()

    return g, has_weights


def build_sparse_adjacency(g: nx.Graph, all_nodes: list) -> csr_matrix:
    """Build sparse adjacency matrix from graph.

    Parameters
    ----------
    g : nx.Graph
        NetworkX graph
    all_nodes : list
        Sorted list of all nodes

    Returns
    -------
    csr_matrix
        Sparse adjacency matrix
    """
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n = len(all_nodes)

    a = lil_matrix((n, n), dtype=np.float32)

    for u, v, data in g.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get("weight", 1.0)
        a[i, j] = weight
        a[j, i] = weight

    return a.tocsr()


def build_adjacency_with_nan(
    g_train: nx.Graph,
    g_full: nx.Graph,
    all_nodes: list,
    fill_missing_with_nan: bool = False,
) -> np.ndarray:
    """Build adjacency matrix with NaN for test edges.

    Parameters
    ----------
    g_train : nx.Graph
        Training graph
    g_full : nx.Graph
        Full graph (for test edge positions)
    all_nodes : list
        Sorted list of all nodes
    fill_missing_with_nan : bool
        If True, fill non-train edges with NaN, otherwise with zeros

    Returns
    -------
    np.ndarray
        Adjacency matrix with train edges as 1, rest as NaN or 0, diagonal as NaN
    """
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n = len(all_nodes)

    fill_value = np.nan if fill_missing_with_nan else 0.0
    adj = np.full((n, n), fill_value, dtype=np.float32)

    for u, v in g_train.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    np.fill_diagonal(adj, np.nan)

    return adj
