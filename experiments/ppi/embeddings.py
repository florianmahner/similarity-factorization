"""
Embedding methods for node classification benchmark.

Provides unified interface for different embedding methods:
- SRF with Adamic-Adar similarity transform
- Spectral embedding via normalized Laplacian
- OpenNE methods (DeepWalk, Node2Vec, LINE)
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from pysrf import SRF
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root / "third_party/OpenNE/src"))

log = logging.getLogger(__name__)


def fit_srf_aa(
    nodes: list, edges: list, rank: int, seed: int, max_outer: int = 300
) -> dict[str, np.ndarray]:
    """Fit SRF with Adamic-Adar similarity transform."""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    degree = adj.sum(axis=1)
    safe_degree = np.maximum(degree, 2.0)
    inv_log_degree = 1.0 / np.log(safe_degree)
    similarity = adj @ np.diag(inv_log_degree) @ adj

    np.fill_diagonal(similarity, np.nan)

    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=max_outer,
        max_inner=50,
        tol=1e-5,
        verbose=0,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss="frobenius",
    )

    W = model.fit_transform(similarity)
    return {node: W[idx] for node, idx in node_to_idx.items()}


def fit_spectral(
    nodes: list, edges: list, rank: int, seed: int
) -> dict[str, np.ndarray]:
    """Spectral embedding using normalized Laplacian eigenvectors."""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    rows, cols = [], []
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            rows.extend([i, j])
            cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    L_sym = diags(np.ones(n_nodes)) - D_inv_sqrt @ A @ D_inv_sqrt

    np.random.seed(seed)
    _, eigvecs = eigsh(L_sym, k=rank, which="SM", tol=1e-6)

    return {node: eigvecs[idx] for node, idx in node_to_idx.items()}


def fit_openne(
    method: str,
    g_nx: nx.Graph,
    rank: int,
    seed: int,
    line_params: dict | None = None,
    walks_params: dict | None = None,
) -> dict[str, np.ndarray]:
    """Fit OpenNE embedding methods (deepwalk, node2vec, line)."""
    try:
        import tensorflow as tf

        tf.compat.v1.disable_v2_behavior()
        from openne.graph import Graph
        from openne.line import LINE
        from openne.node2vec import Node2vec
    except ImportError:
        log.warning("OpenNE not available")
        return {}

    random.seed(seed)
    np.random.seed(seed)

    g_directed = g_nx.to_directed() if not g_nx.is_directed() else g_nx.copy()
    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    g = Graph()
    g.read_g(g_directed)

    walks = walks_params or {}
    num_paths = walks.get("num_paths", 20)
    path_length = walks.get("path_length", 80)

    if method == "deepwalk":
        model = Node2vec(
            g, path_length=path_length, num_paths=num_paths, dim=rank, workers=1, dw=True
        )
        return model.vectors

    if method == "node2vec":
        model = Node2vec(
            g, path_length=path_length, num_paths=num_paths, dim=rank, workers=1, p=1, q=1
        )
        return model.vectors

    if method == "line":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        params = line_params or {}
        model = LINE(
            g,
            rep_size=rank,
            epoch=params.get("epoch", 1000),
            batch_size=params.get("batch_size", 512),
            order=params.get("order", 3),
            negative_ratio=params.get("negative_ratio", 5),
        )
        return model.vectors

    return {}


def embeddings_to_matrix(embeddings: dict[str, np.ndarray], nodes: list) -> np.ndarray:
    """Convert embedding dict to matrix aligned with nodes list."""
    if not embeddings:
        return np.zeros((len(nodes), 1), dtype=np.float32)

    sample_emb = next(iter(embeddings.values()))
    dim = len(sample_emb)
    W = np.zeros((len(nodes), dim), dtype=np.float32)

    for i, node in enumerate(nodes):
        if node in embeddings:
            W[i] = embeddings[node]

    return W
