"""
Node Classification Development Task (Self-Contained Parallel).
Runs multiple configurations in parallel using Joblib and generates plots.
"""

from __future__ import annotations

import logging
import sys
import json
import random
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, ListConfig
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up TensorFlow compatibility BEFORE importing OpenNE
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

# Project Imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "third_party/OpenNE/src"))

from experiments.ppi.utils import load_network
from experiments.ppi.node_classification import (
    fetch_go_annotations,
    build_label_matrix,
    filter_terms,
)
from pysrf import SRF

log = logging.getLogger(__name__)
sns.set_theme(style="ticks", context="talk")

# --- Plotting Constants ---
PALETTE_METHOD = "viridis"
PALETTE_SCALING = "magma"


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


def fit_pysrf_aa(
    nodes: list, edges: list, rank: int, seed: int, max_outer: int = 150
) -> dict[str, np.ndarray]:
    """Fit SRF with Adamic-Adar similarity transform (instead of binary adjacency)."""
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


def get_subgraph(g, n_nodes):
    if n_nodes >= len(g):
        return g
    top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    top_nodes = [n for n, d in top_nodes]
    sub = g.subgraph(top_nodes).copy()
    largest_cc = max(nx.connected_components(sub), key=len)
    return sub.subgraph(largest_cc).copy()


def fit_openne(method, g_nx, rank, seed, n_jobs=1, line_params=None):
    try:
        import openne
        from openne.graph import Graph
        from openne.node2vec import Node2vec
        from openne.line import LINE
    except ImportError:
        return {}

    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

    g_directed = g_nx.to_directed() if not g_nx.is_directed() else g_nx.copy()
    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    g = Graph()
    g.read_g(g_directed)

    if method == "node2vec":
        model = Node2vec(
            g, path_length=16, num_paths=10, dim=rank, workers=n_jobs, p=4, q=1
        )
        return model.vectors
    elif method == "deepwalk":
        model = Node2vec(
            g, path_length=40, num_paths=10, dim=rank, workers=n_jobs, dw=True
        )
        return model.vectors
    elif method == "line":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        params = line_params or {}
        model = LINE(
            g,
            rep_size=rank,
            epoch=params.get("epoch", 20),
            batch_size=params.get("batch_size", 500),
            order=params.get("order", 3),
            negative_ratio=params.get("negative_ratio", 5),
            table_size=params.get("table_size", 1_000_000),
        )
        log.info(f"LINE returned {len(model.vectors)} embeddings")
        if model.vectors:
            sample_keys = list(model.vectors.keys())[:3]
            log.info(f"LINE sample keys: {sample_keys}")
            log.info(f"LINE sample key types: {[type(k) for k in sample_keys]}")
        return model.vectors
    return {}


def run_single_task(task):
    """
    Worker function for a single configuration.
    task: dict with keys (size, rank, method, bin_range, seed, data_dir, g_sub)
    """
    size = task["size"]
    rank = task["rank"]
    method = task["method"]
    bin_min, bin_max = task["bin_range"]
    seed = task["seed"]
    data_dir = task["data_dir"]

    g_sub = task["g_sub"]
    nodes = sorted(g_sub.nodes())

    # Labels
    proteins_clean = np.array([p.split(".")[-1] if "." in p else p for p in nodes])
    cache_path = Path(data_dir) / "go_annotations_human.pkl"
    annotations = fetch_go_annotations(proteins_clean, cache_path)

    Y, terms, valid_indices = build_label_matrix(proteins_clean, annotations)
    Y_valid = Y[valid_indices]

    Y_filt, terms_filt = filter_terms(Y_valid, terms, bin_min, bin_max)

    if Y_filt.shape[1] == 0:
        return None

    labeled_nodes_list = [nodes[i] for i in valid_indices]
    label_dict = {
        prot: [terms_filt[j] for j in range(Y_filt.shape[1]) if Y_filt[i, j]]
        for i, prot in enumerate(labeled_nodes_list)
    }
    valid_nodes = [n for n, labs in label_dict.items() if labs]

    if len(valid_nodes) < 10:
        return None

    # Embedding
    embeddings = {}
    try:
        if method == "srf":
            embeddings = fit_pysrf_aa(
                nodes, list(g_sub.edges()), rank=rank, seed=seed, max_outer=300
            )
        elif method == "spectral":
            embeddings = fit_spectral(nodes, list(g_sub.edges()), rank=rank, seed=seed)
        elif method in ["node2vec", "deepwalk", "line"]:
            embeddings = fit_openne(
                method, g_sub, rank, seed, n_jobs=1, line_params=task.get("line_params")
            )
    except Exception as e:
        log.error(f"Error in embedding {method}: {e}")
        return None

    if not embeddings:
        log.warning(f"{method}: No embeddings returned")
        return None

    log.info(
        f"{method}: Got {len(embeddings)} embeddings, need {len(valid_nodes)} valid nodes"
    )

    # Check overlap
    emb_nodes = set(embeddings.keys())
    valid_set = set(valid_nodes)
    overlap = emb_nodes & valid_set
    log.info(
        f"{method}: Overlap between embeddings and valid nodes: {len(overlap)}/{len(valid_nodes)}"
    )

    # Classification
    X = np.array([embeddings.get(n, np.zeros(rank)) for n in valid_nodes])
    y_labels = [label_dict[n] for n in valid_nodes]

    # Check if X has any non-zero rows
    non_zero_rows = np.any(X != 0, axis=1).sum()
    log.info(f"{method}: Non-zero embedding rows: {non_zero_rows}/{len(X)}")

    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(y_labels)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y_bin, test_size=0.2, random_state=seed
        )
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200, solver="liblinear"))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return {
            "Method": method,
            "Nodes": size,
            "Rank": rank,
            "Bin": f"{bin_min}-{bin_max}",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Micro-F1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    except Exception as e:
        log.error(f"Error in classification: {e}")
        return None


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info(f"--- Parallel Development Run (Joblib) ---")

    # Parse Lists from Config
    sizes = (
        list(cfg.sizes) if isinstance(cfg.sizes, (list, ListConfig)) else [cfg.sizes]
    )
    ranks = (
        list(cfg.ranks) if isinstance(cfg.ranks, (list, ListConfig)) else [cfg.ranks]
    )
    methods = (
        list(cfg.methods)
        if isinstance(cfg.methods, (list, ListConfig))
        else [cfg.methods]
    )
    bins_keys = list(cfg.bins.keys())

    log.info(
        f"Sweeping: Sizes={sizes} | Ranks={ranks} | Methods={methods} | Bins={bins_keys}"
    )

    # Data Loading (Once for the entire sweep)
    data_path = Path(cfg.data_dir) / "STRING_human_min900_v12.csv"
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return
    g_full = load_network(data_path)

    # Prepare Tasks (Config Dictionary for each combination)
    line_cfg = cfg.line if "line" in cfg else None
    line_params = {
        "epoch": getattr(line_cfg, "epoch", 20),
        "batch_size": getattr(line_cfg, "batch_size", 500),
        "order": getattr(line_cfg, "order", 3),
        "negative_ratio": getattr(line_cfg, "negative_ratio", 5),
        "table_size": getattr(line_cfg, "table_size", 1_000_000),
    }
    tasks = []
    # Only create subgraphs once per size
    subgraphs_by_size = {size: get_subgraph(g_full, size) for size in sizes}

    for size in sizes:
        g_sub = subgraphs_by_size[size]
        for rank in ranks:
            for method in methods:
                for bin_key in bins_keys:
                    bin_range = cfg.bins[bin_key]
                    tasks.append(
                        {
                            "size": size,
                            "rank": rank,
                            "method": method,
                            "bin_range": bin_range,
                            "seed": cfg.seed,
                            "data_dir": cfg.data_dir,
                            "g_sub": g_sub,  # Pass subgraph object
                            "line_params": line_params,
                        }
                    )

    log.info(f"Launching {len(tasks)} tasks with n_jobs={cfg.n_jobs}...")

    # Run in parallel using Joblib
    all_results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_task)(task) for task in tasks
    )

    # Cleanup and Aggregate Results
    clean_results = [r for r in all_results if r is not None]
    if not clean_results:
        log.warning("No successful results to plot.")
        return

    df = pd.DataFrame(clean_results)

    # --- Save Results (Once) ---
    output_dir = Path.cwd()  # Current working directory (timestamped run folder)
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    log.info(f"Saved results to {output_dir / 'benchmark_results.csv'}")

    log.info("Run complete.")


if __name__ == "__main__":
    main()
