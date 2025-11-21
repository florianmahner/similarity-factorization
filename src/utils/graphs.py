from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import lil_matrix, csr_matrix


def build_directed_graph(
    sources: list[str], targets: list[str], weights: list[float]
) -> nx.DiGraph:
    """Build directed graph from edge lists with weights."""
    g = nx.DiGraph()
    for src, tgt, weight in zip(sources, targets, weights):
        g.add_edge(src, tgt, weight=weight)
    return g


def clean_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Remove self-loops, nodes without outgoing edges, keep largest SCC."""
    g = g.copy()
    g.remove_edges_from(nx.selfloop_edges(g))
    g.remove_nodes_from([n for n in g.nodes() if g.out_degree(n) == 0])
    largest_scc = max(nx.strongly_connected_components(g), key=len)
    return g.subgraph(largest_scc).copy()


def graph_to_matrix(g: nx.DiGraph, nodes: list[str]) -> np.ndarray:
    """Extract weighted adjacency matrix from graph."""
    return nx.to_numpy_array(g, nodelist=nodes, weight="weight")


def _geometric_mean(a: float, b: float) -> float:
    return np.sqrt(a * b) if a > 0 and b > 0 else max(a, b)


def _arithmetic_mean(a: float, b: float) -> float:
    return (a + b) / 2


def _max(a: float, b: float) -> float:
    return max(a, b)


# def symmetrize_matrix(
#     matrix: np.ndarray, method: str = "geometric_mean", bidirectional_only: bool = False
# ) -> np.ndarray:
#     """Symmetrize matrix, optionally keeping only bidirectional edges.

#     Args:
#         matrix: Directed adjacency matrix (may be asymmetric)
#         method: How to combine bidirectional edges (geometric_mean, arithmetic_mean, max)
#         bidirectional_only: If True, only keep edges where both i→j and j→i exist
#     """
#     n = matrix.shape[0]
#     symmetric_matrix = np.full((n, n), np.nan, dtype=np.float32)

#     func = {
#         "geometric_mean": _geometric_mean,
#         "arithmetic_mean": _arithmetic_mean,
#         "max": _max,
#     }[method]

#     for i in range(n):
#         for j in range(i + 1, n):
#             has_forward = not np.isnan(matrix[i, j])
#             has_backward = not np.isnan(matrix[j, i])

#             if bidirectional_only and not (has_forward and has_backward):
#                 continue

#             average = func(matrix[i, j], matrix[j, i])
#             symmetric_matrix[i, j] = symmetric_matrix[j, i] = average


#     return symmetric_matrix
def symmetrize_matrix(
    matrix: np.ndarray,
    method: str = "sum",  # Recommended: 'sum' or 'arithmetic_mean'
    bidirectional_only: bool = False,
) -> np.ndarray:
    """
    Vectorized matrix symmetrization.

    Args:
        matrix: Directed adjacency matrix (NaN indicates missing edge)
        method: 'sum', 'arithmetic_mean', 'geometric_mean', or 'max'
        bidirectional_only: If True, only keep edges where BOTH directions exist.
                            (WARNING: This destroys semantic data)
    """
    # 1. Create a mask of where edges exist
    # (Assuming your input uses NaN for missing edges)
    mask_forward = ~np.isnan(matrix)
    mask_backward = ~np.isnan(matrix.T)

    # 2. Handle "Bidirectional Only" logic
    # If we only want edges that exist both ways (AND logic)
    if bidirectional_only:
        valid_mask = mask_forward & mask_backward
        # Zero out unidirectional edges so they don't contribute
        # We use a temporary copy to treat NaNs/invalid as 0
        mat_clean = np.where(valid_mask, matrix, 0.0)
        mat_clean_T = np.where(valid_mask, matrix.T, 0.0)
    else:
        # Standard: Treat NaN as 0, keep everything (OR logic)
        valid_mask = mask_forward | mask_backward
        mat_clean = np.nan_to_num(matrix, nan=0.0)
        mat_clean_T = np.nan_to_num(matrix.T, nan=0.0)

    # 3. Apply Symmetrization Method
    if method == "sum":
        # Math: A + A.T is always symmetric
        # 50 + 0 = 50 (Preserves unidirectional)
        symmetric = mat_clean + mat_clean_T

    elif method == "arithmetic_mean":
        # Math: (A + A.T) / 2
        symmetric = (mat_clean + mat_clean_T) / 2.0

    elif method == "max":
        # Take the stronger association
        symmetric = np.maximum(mat_clean, mat_clean_T)

    elif method == "geometric_mean":
        # Requires both to be non-zero.
        # sqrt(50 * 0) = 0. (Destroys unidirectional)
        symmetric = np.sqrt(np.multiply(mat_clean, mat_clean_T))

    else:
        raise ValueError(f"Unknown method: {method}")

    # 4. Restore NaNs (Optional but recommended)
    # If an edge didn't exist in EITHER direction, it should remain NaN
    # rather than 0, depending on your PPMI function's needs.
    # 'valid_mask' tracks where at least one edge existed (or both if bidirectional_only)

    final_matrix = np.full_like(matrix, np.nan)
    final_matrix[valid_mask] = symmetric[valid_mask]

    # Ensure diagonal is NaN (no self-loops)
    np.fill_diagonal(final_matrix, np.nan)

    return final_matrix


def build_undirected_graph(edges: list[tuple], weighted: bool = False) -> nx.Graph:
    g = nx.Graph()
    if weighted:
        g.add_weighted_edges_from(edges)
    else:
        g.add_edges_from(edges)
    return g


def largest_component(g: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    if isinstance(g, nx.DiGraph):
        components = nx.strongly_connected_components(g)
    else:
        components = nx.connected_components(g)
    largest = max(components, key=len)
    return g.subgraph(largest).copy()


def build_sparse_adjacency(g: nx.Graph, nodes: list) -> csr_matrix:
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    a = lil_matrix((n, n), dtype=np.float32)

    for u, v, data in g.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get("weight", 1.0)
        a[i, j] = a[j, i] = weight

    return a.tocsr()


def load_edgelist_csv(
    csv_path: Path | str,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = None,
    deduplicate: bool = True,
    normalize_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(csv_path)
    sources, targets = df[source_col].values, df[target_col].values

    if weight_col and weight_col in df.columns:
        weights = df[weight_col].values
        if deduplicate:
            edge_df = pd.DataFrame(
                {
                    "u": np.minimum(sources, targets),
                    "v": np.maximum(sources, targets),
                    "w": weights,
                }
            )
            edge_df = edge_df.groupby(["u", "v"], as_index=False)["w"].max()
            sources, targets, weights = (
                edge_df["u"].values,
                edge_df["v"].values,
                edge_df["w"].values,
            )
        if normalize_weights and weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        return sources, targets, weights
    else:
        if deduplicate:
            u, v = np.minimum(sources, targets), np.maximum(sources, targets)
            # Create a structured array to allow np.unique to work on rows of strings
            dtype = [("f0", u.dtype), ("f1", v.dtype)]
            edges_structured = np.array(list(zip(u, v)), dtype=dtype)
            unique_edges_structured = np.unique(edges_structured)
            sources, targets = (
                unique_edges_structured["f0"],
                unique_edges_structured["f1"],
            )
        return sources, targets, None
