from __future__ import annotations

import networkx as nx
import numpy as np


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


def symmetrize_matrix(
    matrix: np.ndarray, method: str = "geometric_mean", bidirectional_only: bool = True
) -> np.ndarray:
    """Symmetrize matrix, optionally keeping only bidirectional edges."""
    n = matrix.shape[0]
    symmetric_matrix = np.full((n, n), np.nan, dtype=np.float32)

    func = {
        "geometric_mean": _geometric_mean,
        "arithmetic_mean": _arithmetic_mean,
        "max": _max,
    }[method]

    for i in range(n):
        for j in range(i + 1, n):
            average = func(matrix[i, j], matrix[j, i])
            symmetric_matrix[i, j] = symmetric_matrix[j, i] = average

    return symmetric_matrix
