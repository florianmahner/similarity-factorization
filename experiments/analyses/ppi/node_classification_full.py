"""
Node Classification on Full STRING Human Network.

Run: poetry run python experiments/ppi/node_classification_full.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

RANK = 32
METHODS = ["srf", "deepwalk", "line"]
GO_BINS = {
    "rare": (11, 30),
    "medium": (31, 100),
    "frequent": (101, 300),
}
TRAIN_RATIO = 0.8
SEED = 42
N_JOBS = -1

# SRF settings
SRF_RHO = 0.5
SRF_MAX_OUTER = 1000

# Output paths
DATA_DIR = PROJECT_ROOT / "data" / "ppi"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "ppi" / "data" / "node_classification"

# =============================================================================


def load_full_network() -> tuple[nx.Graph, list[str], dict]:
    """Load full STRING human network."""
    log.info("Loading STRING human network...")
    df = pd.read_csv(DATA_DIR / "STRING_human_min900_v12.csv")

    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row["source"], row["target"], weight=row["combined_score"])

    # Keep largest connected component
    largest_cc = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()

    nodes = sorted(g.nodes())
    log.info(f"Network: {len(nodes)} nodes, {g.number_of_edges()} edges")

    # Extract protein IDs (remove species prefix)
    proteins = [n.split(".")[-1] if "." in n else n for n in nodes]

    # Load GO annotations
    cache = DATA_DIR / "go_annotations_human.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            annotations = pickle.load(f)
        log.info(f"Loaded {len(annotations)} GO annotations from cache")
    else:
        raise FileNotFoundError(f"GO cache not found: {cache}")

    # Map annotations to node names
    node_labels = {}
    for node, prot in zip(nodes, proteins):
        if prot in annotations:
            node_labels[node] = annotations[prot]

    log.info(f"Nodes with GO annotations: {len(node_labels)}")

    return g, nodes, node_labels


def filter_go_terms(labels: dict, nodes: list, min_count: int, max_count: int) -> dict:
    """Filter GO terms by frequency."""
    term_counts = {}
    for n in nodes:
        if n in labels:
            for term in labels[n]:
                term_counts[term] = term_counts.get(term, 0) + 1

    valid_terms = {t for t, c in term_counts.items() if min_count <= c <= max_count}

    filtered = {}
    for n in nodes:
        if n in labels:
            valid = [t for t in labels[n] if t in valid_terms]
            if valid:
                filtered[n] = valid

    return filtered


def build_adjacency(g: nx.Graph, nodes: list) -> np.ndarray:
    """Build adjacency matrix."""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)

    for u, v in g.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def build_adamic_adar(adj: np.ndarray) -> np.ndarray:
    """Compute Adamic-Adar similarity from adjacency."""
    log.info("Computing Adamic-Adar similarity...")
    degree = adj.sum(axis=1)
    safe_degree = np.maximum(degree, 2.0)
    inv_log_degree = 1.0 / np.log(safe_degree)
    return adj @ np.diag(inv_log_degree) @ adj


def fit_srf(similarity: np.ndarray, rank: int) -> np.ndarray:
    """Fit SRF embeddings."""
    log.info(f"Fitting SRF (rank={rank}, rho={SRF_RHO})...")
    S = similarity.copy()
    np.fill_diagonal(S, np.nan)

    model = SRF(
        rank=rank,
        rho=SRF_RHO,
        max_outer=SRF_MAX_OUTER,
        max_inner=50,
        tol=1e-5,
        verbose=1,
        init="random_sqrt",
        random_state=SEED,
        missing_values=np.nan,
        loss="frobenius",
    )
    return model.fit_transform(S)


def fit_deepwalk(g: nx.Graph, nodes: list, rank: int) -> np.ndarray:
    """Fit DeepWalk embeddings."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    log.info(f"Fitting DeepWalk (rank={rank})...")

    from openne.graph import Graph
    from openne.node2vec import Node2vec

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    g_directed = g.to_directed()

    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    openne_g = Graph()
    openne_g.read_g(g_directed)

    model = Node2vec(openne_g, path_length=80, num_paths=10, dim=rank, workers=8, dw=True)

    W = np.zeros((len(nodes), rank), dtype=np.float32)
    for node, idx in node_to_idx.items():
        if node in model.vectors:
            W[idx] = model.vectors[node]

    return W


def fit_line(g: nx.Graph, nodes: list, rank: int) -> np.ndarray:
    """Fit LINE embeddings."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    log.info(f"Fitting LINE (rank={rank})...")

    from openne.graph import Graph
    from openne.line import LINE

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    g_directed = g.to_directed()

    for u, v in g_directed.edges():
        if "weight" not in g_directed[u][v]:
            g_directed[u][v]["weight"] = 1.0

    openne_g = Graph()
    openne_g.read_g(g_directed)

    model = LINE(openne_g, rep_size=rank, order=2)

    W = np.zeros((len(nodes), rank), dtype=np.float32)
    for node, idx in node_to_idx.items():
        if node in model.vectors:
            W[idx] = model.vectors[node]

    return W


def evaluate(
    W: np.ndarray,
    nodes: list,
    labels: dict,
    train_ratio: float,
) -> dict:
    """Evaluate node classification."""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    labeled_nodes = [n for n in nodes if n in labels]

    if len(labeled_nodes) < 50:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "n_labeled": 0}

    X = np.array([W[node_to_idx[n]] for n in labeled_nodes])
    y = [labels[n] for n in labeled_nodes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=SEED
    )

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED, n_jobs=N_JOBS)
    )
    clf.fit(X_train, y_train_bin)
    y_pred = clf.predict(X_test)

    return {
        "micro_f1": f1_score(y_test_bin, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test_bin, y_pred, average="macro", zero_division=0),
        "n_labeled": len(labeled_nodes),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_result(method: str, go_bin: str, result: dict):
    """Save result to JSON."""
    method_dir = OUTPUT_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)

    out_file = method_dir / f"rank{RANK}_{go_bin}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    g, nodes, all_labels = load_full_network()
    adj = build_adjacency(g, nodes)

    # Compute embeddings
    embeddings = {}

    if "srf" in METHODS:
        aa_sim = build_adamic_adar(adj)
        embeddings["srf"] = fit_srf(aa_sim, RANK)
        np.save(OUTPUT_DIR / "srf" / f"embedding_rank{RANK}.npy", embeddings["srf"])

    if "deepwalk" in METHODS:
        embeddings["deepwalk"] = fit_deepwalk(g, nodes, RANK)
        (OUTPUT_DIR / "deepwalk").mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_DIR / "deepwalk" / f"embedding_rank{RANK}.npy", embeddings["deepwalk"])

    if "line" in METHODS:
        embeddings["line"] = fit_line(g, nodes, RANK)
        (OUTPUT_DIR / "line").mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_DIR / "line" / f"embedding_rank{RANK}.npy", embeddings["line"])

    # Save node list
    with open(OUTPUT_DIR / "nodes.json", "w") as f:
        json.dump(nodes, f)

    # Evaluate on each GO bin
    log.info("\n" + "=" * 60)
    log.info("EVALUATION")
    log.info("=" * 60)

    all_results = []

    for bin_name, (min_c, max_c) in GO_BINS.items():
        bin_labels = filter_go_terms(all_labels, nodes, min_c, max_c)
        log.info(f"\n{bin_name.upper()} GO terms ({min_c}-{max_c}): {len(bin_labels)} proteins")

        for method, W in embeddings.items():
            result = evaluate(W, nodes, bin_labels, TRAIN_RATIO)
            result.update({
                "method": method,
                "go_bin": bin_name,
                "rank": RANK,
                "n_nodes": len(nodes),
            })

            save_result(method, bin_name, result)
            all_results.append(result)

            log.info(f"  {method}: micro_f1={result['micro_f1']:.3f}, macro_f1={result['macro_f1']:.3f}")

    # Save summary
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    log.info(f"\nSummary saved to {OUTPUT_DIR / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
