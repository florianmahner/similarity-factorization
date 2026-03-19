"""
Test NetMF-style matrix factorization with SRF.

Based on "Network Embedding as Matrix Factorization" (Qiu et al., WSDM 2018):
- DeepWalk implicitly factorizes: log(vol(G)/T * sum_{r=1}^T P^r * D^{-1})
- Where P = D^{-1}A is the random walk transition matrix
- NetMF directly computes and factorizes this matrix, outperforming DeepWalk by 50%

This sandbox tests if SRF can match/beat DeepWalk by factorizing the same matrix.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from experiments.ppi.node_classification_sweep import (
    load_ppi, get_subgraph, filter_go_terms, load_go_ontology_mapping,
    PPI_DATA, GO_BIN_RANGES
)
from experiments.ppi.embeddings import fit_openne, embeddings_to_matrix

SEED = 42
SIZE = 1000
RANK = 32
WINDOW = 10  # context window size (T in NetMF paper)


def build_netmf_matrix(adj: np.ndarray, window: int = 10) -> np.ndarray:
    """Build the NetMF matrix: log(max(vol(G)/T * sum P^r * D^{-1}, 1))."""
    degree = adj.sum(axis=1)
    safe_degree = np.maximum(degree, 1.0)
    vol_g = degree.sum()

    # P = D^{-1}A is the transition matrix
    d_inv = np.diag(1.0 / safe_degree)
    P = d_inv @ adj

    # Sum of P^r for r=1 to T
    P_sum = np.zeros_like(adj)
    P_power = P.copy()
    for _ in range(window):
        P_sum += P_power
        P_power = P_power @ P

    # M = vol(G)/T * P_sum * D^{-1}
    M = (vol_g / window) * P_sum @ d_inv

    # Symmetrize (since P is not symmetric, but SRF needs symmetric input)
    M_sym = (M + M.T) / 2

    # Shifted PPMI: log(max(M, 1))
    return np.log(np.maximum(M_sym, 1.0))


def build_aa_matrix(adj: np.ndarray, use_log: bool = True) -> np.ndarray:
    """Build Adamic-Adar similarity matrix."""
    degree = adj.sum(axis=1)
    safe_degree = np.maximum(degree, 2.0)
    inv_log_degree = 1.0 / np.log(safe_degree)
    aa = adj @ np.diag(inv_log_degree) @ adj
    if use_log:
        return np.log1p(aa)
    return aa


def fit_srf(similarity: np.ndarray, rank: int, rho: float = 0.5, max_outer: int = 300) -> np.ndarray:
    """Fit SRF on similarity matrix."""
    sim = similarity.copy()
    np.fill_diagonal(sim, np.nan)

    model = SRF(
        rank=rank,
        rho=rho,
        max_outer=max_outer,
        max_inner=30,
        tol=1e-4,
        verbose=0,
        init="random_sqrt",
        random_state=SEED,
        missing_values=np.nan,
        loss="frobenius",
    )
    return model.fit_transform(sim)


def evaluate(W: np.ndarray, nodes: list, labels: dict, train_ratio: float = 0.8) -> dict:
    """Evaluate embeddings on node classification."""
    labeled_nodes = [n for n in nodes if n in labels]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    X = np.array([W[node_to_idx[n]] for n in labeled_nodes])
    y = [labels[n] for n in labeled_nodes]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_ratio, random_state=SEED)

    binarizer = MultiLabelBinarizer()
    y_tr_bin = binarizer.fit_transform(y_tr)
    y_te_bin = binarizer.transform(y_te)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=500, random_state=SEED))
    clf.fit(X_tr, y_tr_bin)
    y_pred = clf.predict(X_te)

    return {
        "micro_f1": f1_score(y_te_bin, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_te_bin, y_pred, average="macro", zero_division=0),
    }


def main():
    print("Loading PPI...")
    g_full, all_labels, _ = load_ppi()
    g = get_subgraph(g_full, SIZE)
    nodes = sorted(g.nodes())
    edges = list(g.edges())
    go_ontology = load_go_ontology_mapping(PPI_DATA)

    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")

    # Build adjacency matrix
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for u, v in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = adj[j, i] = 1.0

    # Build different similarity matrices
    print("\nBuilding similarity matrices...")
    netmf_sim = build_netmf_matrix(adj, window=WINDOW)
    aa_log_sim = build_aa_matrix(adj, use_log=True)
    aa_raw_sim = build_aa_matrix(adj, use_log=False)

    print(f"NetMF matrix: range [{netmf_sim.min():.2f}, {netmf_sim.max():.2f}], sparsity {(netmf_sim == 0).mean()*100:.1f}%")
    print(f"AA-log matrix: range [{aa_log_sim.min():.2f}, {aa_log_sim.max():.2f}]")

    # Fit embeddings
    methods = {}

    print("\nFitting SRF-NetMF...")
    methods["srf_netmf"] = fit_srf(netmf_sim, RANK, rho=0.5, max_outer=300)

    print("Fitting SRF-AA-log...")
    methods["srf_aa_log"] = fit_srf(aa_log_sim, RANK, rho=0.5, max_outer=300)

    print("Fitting SRF-AA-raw...")
    methods["srf_aa_raw"] = fit_srf(aa_raw_sim, RANK, rho=0.5, max_outer=300)

    print("Fitting DeepWalk...")
    dw_emb = fit_openne("deepwalk", g, RANK, SEED)
    methods["deepwalk"] = embeddings_to_matrix(dw_emb, nodes)

    print("Fitting Node2Vec...")
    n2v_emb = fit_openne("node2vec", g, RANK, SEED)
    methods["node2vec"] = embeddings_to_matrix(n2v_emb, nodes)

    # Evaluate on different GO bins
    results = []
    for method_name, W in methods.items():
        for go_aspect in ["MF", "BP"]:
            for go_bin in ["rare", "medium", "frequent"]:
                labels = {n: all_labels[n] for n in nodes if n in all_labels}
                min_c, max_c = GO_BIN_RANGES[go_bin]
                labels = filter_go_terms(labels, nodes, min_c, max_c, go_aspect, go_ontology)

                if len(labels) < 20:
                    continue

                metrics = evaluate(W, nodes, labels)
                results.append({
                    "method": method_name,
                    "go_aspect": go_aspect,
                    "go_bin": go_bin,
                    "micro_f1": metrics["micro_f1"],
                })

    # Print results
    import pandas as pd
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS: Average micro_f1 by method")
    print("=" * 70)
    avg = df.groupby("method")["micro_f1"].mean().sort_values(ascending=False)
    print(avg.to_string())

    print("\n" + "=" * 70)
    print("By method and GO bin:")
    print("=" * 70)
    pivot = df.pivot_table(values="micro_f1", index="method", columns="go_bin", aggfunc="mean")
    pivot = pivot[["rare", "medium", "frequent"]]
    print(pivot.to_string())


if __name__ == "__main__":
    main()
