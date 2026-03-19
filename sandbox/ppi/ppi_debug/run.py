"""
Debug SRF node classification on PPI.

Key hypotheses to test:
1. Rho value: sparse graphs may need lower rho (0.1-0.5 vs 3.0)
2. Input: raw adjacency vs Adamic-Adar similarity
3. Loss function: frobenius vs kl vs bce
4. Embedding normalization: L2 norm before classification
5. Rank selection: optimal rank may differ from baselines
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pysrf import SRF

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_ppi_subgraph(data_dir: Path, n_nodes: int = 500) -> tuple[nx.Graph, dict]:
    """Load PPI network and extract top-degree subgraph."""
    df = pd.read_csv(data_dir / "STRING_human_min900_v12.csv")

    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row["source"], row["target"])

    largest_cc = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()

    top = sorted(g.degree, key=lambda x: x[1], reverse=True)[:n_nodes]
    sub = g.subgraph([n for n, _ in top]).copy()
    cc = max(nx.connected_components(sub), key=len)
    g = sub.subgraph(cc).copy()

    nodes = sorted(g.nodes())
    proteins = [p.split(".")[-1] if "." in p else p for p in nodes]

    cache = data_dir / "go_annotations_human.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            annotations = pickle.load(f)
    else:
        raise FileNotFoundError(f"GO cache not found: {cache}")

    labels = {}
    for node, prot in zip(nodes, proteins):
        if prot in annotations:
            labels[node] = annotations[prot]

    return g, labels


def filter_go_terms(labels: dict, nodes: list, min_c: int = 11, max_c: int = 100) -> dict:
    """Filter GO terms by frequency."""
    counts = {}
    for n in nodes:
        if n in labels:
            for t in labels[n]:
                counts[t] = counts.get(t, 0) + 1
    valid = {t for t, c in counts.items() if min_c <= c <= max_c}
    return {
        n: [t for t in labels[n] if t in valid]
        for n in nodes if n in labels and any(t in valid for t in labels[n])
    }


def build_adjacency(nodes: list, edges: list) -> np.ndarray:
    """Build binary adjacency matrix."""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def build_adamic_adar(adj: np.ndarray) -> np.ndarray:
    """Transform adjacency to Adamic-Adar similarity."""
    degree = adj.sum(axis=1)
    safe_degree = np.maximum(degree, 2.0)
    inv_log_degree = 1.0 / np.log(safe_degree)
    return adj @ np.diag(inv_log_degree) @ adj


def build_common_neighbors(adj: np.ndarray) -> np.ndarray:
    """Transform adjacency to common neighbors similarity."""
    return adj @ adj


def fit_srf(
    similarity: np.ndarray,
    rank: int,
    rho: float,
    loss: str,
    seed: int,
    max_outer: int = 100,
) -> np.ndarray:
    """Fit SRF and return embeddings."""
    S = similarity.copy()
    np.fill_diagonal(S, np.nan)

    model = SRF(
        rank=rank,
        rho=rho,
        max_outer=max_outer,
        max_inner=50,
        tol=1e-5,
        verbose=0,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss=loss,
    )
    return model.fit_transform(S)


def fit_deepwalk(g: nx.Graph, nodes: list, rank: int, seed: int) -> np.ndarray:
    """Fit DeepWalk embeddings."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        from openne.graph import Graph
        from openne.node2vec import Node2vec
    except ImportError:
        log.warning("OpenNE not available, skipping DeepWalk")
        return None

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    g_directed = g.to_directed()
    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    openne_g = Graph()
    openne_g.read_g(g_directed)

    model = Node2vec(openne_g, path_length=80, num_paths=20, dim=rank, workers=1, dw=True)

    W = np.zeros((len(nodes), rank), dtype=np.float32)
    for node, idx in node_to_idx.items():
        if node in model.vectors:
            W[idx] = model.vectors[node]
    return W


def fit_node2vec(g: nx.Graph, nodes: list, rank: int, seed: int, p: float = 1.0, q: float = 1.0) -> np.ndarray:
    """Fit Node2Vec embeddings."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        from openne.graph import Graph
        from openne.node2vec import Node2vec
    except ImportError:
        log.warning("OpenNE not available, skipping Node2Vec")
        return None

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    g_directed = g.to_directed()
    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    openne_g = Graph()
    openne_g.read_g(g_directed)

    model = Node2vec(openne_g, path_length=80, num_paths=20, dim=rank, workers=1, p=p, q=q)

    W = np.zeros((len(nodes), rank), dtype=np.float32)
    for node, idx in node_to_idx.items():
        if node in model.vectors:
            W[idx] = model.vectors[node]
    return W


def fit_spectral(nodes: list, edges: list, rank: int, seed: int) -> np.ndarray:
    """Spectral embedding using normalized Laplacian eigenvectors."""
    from scipy.sparse.linalg import eigsh

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

    return eigvecs


def evaluate(
    W: np.ndarray,
    nodes: list,
    labels: dict,
    train_ratio: float,
    seed: int,
) -> dict:
    """Evaluate node classification using logistic regression."""
    idx = {n: i for i, n in enumerate(nodes)}
    labeled = [n for n in nodes if n in labels]

    if len(labeled) < 10:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "n_labeled": 0}

    X = np.array([W[idx[n]] for n in labeled])
    y = [labels[n] for n in labeled]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_ratio, random_state=seed)

    mlb = MultiLabelBinarizer()
    y_tr_bin = mlb.fit_transform(y_tr)
    y_te_bin = mlb.transform(y_te)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed))
    clf.fit(X_tr, y_tr_bin)
    y_pred = clf.predict(X_te)

    return {
        "micro_f1": f1_score(y_te_bin, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_te_bin, y_pred, average="macro", zero_division=0),
        "n_labeled": len(labeled),
    }


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison plots."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, bin_name in enumerate(["rare", "medium", "frequent"]):
        ax = axes[i]
        bin_df = df[df["go_bin"] == bin_name]

        for method in bin_df["method"].unique():
            method_df = bin_df[bin_df["method"] == method].sort_values("rank")
            ax.plot(method_df["rank"], method_df["micro_f1"], "o-", label=method)

        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Micro F1")
        ax.set_title(f"{bin_name.upper()} GO Terms")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "rank_comparison.pdf")
    plt.close()
    log.info(f"Saved plot to {output_dir / 'rank_comparison.pdf'}")


def run_experiments(args):
    """Run systematic experiments."""
    log.info(f"Loading PPI subgraph with {args.n_nodes} nodes...")
    data_dir = project_root / "data" / "ppi"
    g, all_labels = load_ppi_subgraph(data_dir, args.n_nodes)

    nodes = sorted(g.nodes())
    edges = list(g.edges())
    n = len(nodes)

    # GO term frequency bins (matching benchmark protocol)
    go_bins = {
        "rare": (11, 30),
        "medium": (31, 100),
        "frequent": (101, 300),
    }

    log.info(f"Graph: {n} nodes, {len(edges)} edges")
    for bin_name, (min_c, max_c) in go_bins.items():
        bin_labels = filter_go_terms(all_labels, nodes, min_c, max_c)
        log.info(f"  {bin_name} ({min_c}-{max_c}): {len(bin_labels)} proteins")

    adj = build_adjacency(nodes, edges)
    aa_sim = build_adamic_adar(adj)

    density = adj.sum() / (n * (n - 1))
    log.info(f"Graph density: {density:.4f}")

    results = []
    ranks = [16, 32, 64, 128]

    for rank in ranks:
        log.info(f"\n=== Rank {rank} ===")
        embeddings = {}

        # SRF with Adamic-Adar
        for rho in [0.5, 3.0]:
            try:
                start = time.time()
                W = fit_srf(aa_sim, rank, rho, "frobenius", args.seed)
                runtime = time.time() - start
                embeddings[f"srf_rho{rho}"] = {"W": W, "runtime": runtime}
                log.info(f"  SRF(rho={rho}): {runtime:.1f}s")
            except Exception as e:
                log.warning(f"  SRF(rho={rho}) failed: {e}")

        # DeepWalk
        try:
            start = time.time()
            W = fit_deepwalk(g, nodes, rank, args.seed)
            runtime = time.time() - start
            if W is not None:
                embeddings["deepwalk"] = {"W": W, "runtime": runtime}
                log.info(f"  DeepWalk: {runtime:.1f}s")
        except Exception as e:
            log.warning(f"  DeepWalk failed: {e}")

        # Evaluate on each GO bin
        for bin_name, (min_c, max_c) in go_bins.items():
            bin_labels = filter_go_terms(all_labels, nodes, min_c, max_c)
            if len(bin_labels) < 10:
                continue

            for method_name, emb_data in embeddings.items():
                W = emb_data["W"]
                metrics = evaluate(W, nodes, bin_labels, args.train_ratio, args.seed)
                results.append({
                    "method": method_name,
                    "go_bin": bin_name,
                    "rank": rank,
                    "runtime": emb_data["runtime"],
                    **metrics,
                })
                log.info(f"  {method_name} | {bin_name}: micro={metrics['micro_f1']:.3f}")

    # Save results
    df = pd.DataFrame(results)
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)

    # Create plots
    plot_results(df, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=500, help="Number of nodes in subgraph")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_experiments(args)


if __name__ == "__main__":
    main()
