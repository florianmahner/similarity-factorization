"""
Node Classification Benchmark: SRF vs Graph Embedding Methods.

Compares SRF with different similarity transforms against
DeepWalk, LINE, and Node2Vec on multiple benchmark datasets.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import warnings
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, ListConfig
from pysrf import SRF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "third_party/OpenNE/src"))

log = logging.getLogger(__name__)


def load_blogcatalog(data_dir: Path) -> tuple[nx.Graph, dict[str, list[int]]]:
    edge_file = data_dir / "blogCatalog" / "bc_edgelist.txt"
    label_file = data_dir / "blogCatalog" / "bc_labels.txt"

    g = nx.Graph()
    with open(edge_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])

    labels = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = [int(l) for l in parts[1:]]

    return g, labels


def load_wikipedia(data_dir: Path) -> tuple[nx.Graph, dict[str, list[int]]]:
    edge_file = data_dir / "wiki" / "Wiki_edgelist.txt"
    label_file = data_dir / "wiki" / "wiki_labels.txt"

    g = nx.Graph()
    with open(edge_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])

    labels = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = [int(parts[1])]

    return g, labels


def get_subgraph(g: nx.Graph, n_nodes: int) -> nx.Graph:
    if n_nodes >= len(g):
        return g
    top_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    top_nodes = [n for n, _ in top_nodes]
    sub = g.subgraph(top_nodes).copy()
    largest_cc = max(nx.connected_components(sub), key=len)
    return sub.subgraph(largest_cc).copy()


def build_binary_adjacency(nodes: list, edges: list) -> np.ndarray:
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def transform_to_similarity(adj: np.ndarray, method: str) -> np.ndarray:
    if method == "binary":
        return adj.copy()
    if method == "CN":
        return adj @ adj
    if method == "AA":
        degree = adj.sum(axis=1)
        safe_degree = np.maximum(degree, 2.0)
        inv_log_degree = 1.0 / np.log(safe_degree)
        return adj @ np.diag(inv_log_degree) @ adj
    if method == "RA":
        degree = adj.sum(axis=1)
        safe_degree = np.maximum(degree, 1.0)
        inv_degree = 1.0 / safe_degree
        return adj @ np.diag(inv_degree) @ adj
    raise ValueError(f"Unknown transform: {method}")


def fit_srf(similarity: np.ndarray, rank: int, seed: int) -> np.ndarray:
    sim = similarity.copy()
    np.fill_diagonal(sim, np.nan)
    model = SRF(
        rank=rank, rho=3.0, max_outer=100, max_inner=30, tol=1e-4,
        verbose=0, init="random_sqrt", random_state=seed,
        missing_values=np.nan, loss="frobenius",
    )
    return model.fit_transform(sim)


def fit_graph_method(g_nx: nx.Graph, method: str, rank: int, seed: int) -> dict:
    try:
        import tensorflow as tf
        tf.compat.v1.disable_v2_behavior()
        from openne.graph import Graph
        from openne.line import LINE
        from openne.node2vec import Node2vec
    except ImportError:
        return {}

    random.seed(seed)
    np.random.seed(seed)

    g_directed = g_nx.to_directed()
    for u, v in g_directed.edges():
        g_directed[u][v]["weight"] = 1.0

    g = Graph()
    g.read_g(g_directed)

    if method == "deepwalk":
        model = Node2vec(g, path_length=40, num_paths=10, dim=rank, workers=1, dw=True)
        return model.vectors
    elif method == "node2vec":
        model = Node2vec(g, path_length=40, num_paths=10, dim=rank, workers=1, p=1, q=1)
        return model.vectors
    elif method == "line":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        model = LINE(g, rep_size=rank, epoch=10, batch_size=512, order=3)
        return model.vectors
    return {}


def embeddings_to_matrix(embeddings: dict, nodes: list) -> np.ndarray:
    sample_emb = next(iter(embeddings.values()))
    W = np.zeros((len(nodes), len(sample_emb)), dtype=np.float32)
    for i, node in enumerate(nodes):
        if node in embeddings:
            W[i] = embeddings[node]
    return W


def evaluate_classification(W, nodes, labels, train_ratio, seed, multi_label=True):
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    labeled_nodes = [n for n in nodes if n in labels]

    X = np.array([W[node_to_idx[n]] for n in labeled_nodes])
    y = [labels[n] for n in labeled_nodes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=seed
    )

    if multi_label:
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train)
        y_test_bin = mlb.transform(y_test)
    else:
        lb = LabelBinarizer()
        y_train_bin = lb.fit_transform([l[0] for l in y_train])
        y_test_bin = lb.transform([l[0] for l in y_test])

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
    )
    clf.fit(X_train, y_train_bin)
    y_pred = clf.predict(X_test)

    return {
        "micro_f1": f1_score(y_test_bin, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test_bin, y_pred, average="macro", zero_division=0),
    }


def run_single_method(method, g, adj_binary, nodes, labels, rank, seed, train_ratios, multi_label, dataset):
    """Run a single method and return results for all train ratios."""
    results = []

    if method in ["deepwalk", "node2vec", "line"]:
        emb_dict = fit_graph_method(g, method, rank, seed)
        if not emb_dict:
            return []
        W = embeddings_to_matrix(emb_dict, nodes)
        method_name = method
    else:
        similarity = transform_to_similarity(adj_binary, method)
        W = fit_srf(similarity, rank, seed)
        method_name = f"SRF+{method}"

    for train_ratio in train_ratios:
        metrics = evaluate_classification(W, nodes, labels, train_ratio, seed, multi_label)
        results.append({
            "dataset": dataset,
            "method": method_name,
            "train_ratio": train_ratio,
            "n_nodes": len(nodes),
            "rank": rank,
            **metrics,
        })

    return results


def run(cfg: DictConfig) -> None:
    dataset = cfg.dataset
    log.info(f"Loading dataset: {dataset}")

    if dataset == "blogcatalog":
        data_dir = project_root / "third_party/OpenNE/data"
        g_full, labels = load_blogcatalog(data_dir)
        g = get_subgraph(g_full, cfg.get("size", len(g_full)))
        labels = {n: labels[n] for n in g.nodes() if n in labels}
        multi_label = True
    elif dataset == "wikipedia":
        data_dir = project_root / "third_party/OpenNE/data"
        g_full, labels = load_wikipedia(data_dir)
        g = get_subgraph(g_full, cfg.get("size", len(g_full)))
        labels = {n: labels[n] for n in g.nodes() if n in labels}
        multi_label = False
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    nodes = sorted(g.nodes())
    edges = list(g.edges())

    log.info(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    log.info(f"Labeled nodes: {len([n for n in nodes if n in labels])}")

    adj_binary = build_binary_adjacency(nodes, edges)

    methods = list(cfg.methods) if isinstance(cfg.methods, (list, ListConfig)) else [cfg.methods]
    train_ratios = list(cfg.train_ratios) if isinstance(cfg.train_ratios, (list, ListConfig)) else [cfg.train_ratios]

    log.info(f"Running {len(methods)} methods in parallel...")

    all_results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_single_method)(
            method, g, adj_binary, nodes, labels,
            cfg.rank, cfg.seed, train_ratios, multi_label, dataset
        )
        for method in methods
    )

    flat_results = [r for res_list in all_results for r in res_list]
    df = pd.DataFrame(flat_results)
    df.to_csv(Path.cwd() / "results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {dataset.upper()}")
    print("=" * 70)

    pivot = df.pivot_table(index="method", columns="train_ratio", values="micro_f1")
    print("\nMicro-F1:")
    print(pivot.round(3).to_string())

    pivot_macro = df.pivot_table(index="method", columns="train_ratio", values="macro_f1")
    print("\nMacro-F1:")
    print(pivot_macro.round(3).to_string())


if __name__ == "__main__":
    hydra.main(config_path=".", config_name="config", version_base=None)(run)()
