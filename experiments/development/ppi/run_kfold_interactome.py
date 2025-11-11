import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix, diags

try:
    from pysrf import SRF

    srf_available = True
except ImportError:
    srf_available = False


def load_network(csv_path: str) -> tuple[nx.Graph, bool]:
    df = pd.read_csv(csv_path)

    has_weights = "weight" in df.columns

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
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n = len(all_nodes)

    a = lil_matrix((n, n), dtype=np.float32)

    for u, v, data in g.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get("weight", 1.0)
        a[i, j] = weight
        a[j, i] = weight

    return a.tocsr()


def evaluate_baseline_fast(
    a_sparse: csr_matrix,
    pairs: np.ndarray,
    labels: np.ndarray,
    method: str,
    batch_size: int = 1000000,
) -> dict[str, float]:
    n_pairs = len(pairs)
    scores = np.zeros(n_pairs, dtype=np.float32)

    if method == "CN":
        similarity_matrix = a_sparse @ a_sparse
    elif method == "AA":
        degree = np.array(a_sparse.sum(axis=1)).flatten()
        safe_degree = np.maximum(degree, 2.0)
        inv_log_degree = 1.0 / np.log(safe_degree)
        weights_diag = diags(inv_log_degree, format="csr")
        similarity_matrix = a_sparse @ weights_diag @ a_sparse
    elif method == "RA":
        degree = np.array(a_sparse.sum(axis=1)).flatten()
        safe_degree = np.maximum(degree, 1.0)
        inv_degree = 1.0 / safe_degree
        weights_diag = diags(inv_degree, format="csr")
        similarity_matrix = a_sparse @ weights_diag @ a_sparse
    elif method == "JC":
        similarity_matrix = None

    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_pairs = pairs[start:end]

        if method in ["CN", "AA", "RA"]:
            batch_scores = np.array(
                similarity_matrix[batch_pairs[:, 0], batch_pairs[:, 1]]
            ).flatten()

        elif method == "JC":
            batch_size_jc = min(10000, end - start)
            batch_scores = np.zeros(end - start, dtype=np.float32)

            for sub_start in range(0, end - start, batch_size_jc):
                sub_end = min(sub_start + batch_size_jc, end - start)
                sub_pairs = batch_pairs[sub_start:sub_end]

                for idx, (i, j) in enumerate(sub_pairs):
                    neighbors_i = a_sparse[i].indices
                    neighbors_j = a_sparse[j].indices

                    if len(neighbors_i) == 0 or len(neighbors_j) == 0:
                        batch_scores[sub_start + idx] = 0.0
                        continue

                    intersection = np.intersect1d(
                        neighbors_i, neighbors_j, assume_unique=True
                    )
                    union_size = len(neighbors_i) + len(neighbors_j) - len(intersection)

                    batch_scores[sub_start + idx] = (
                        len(intersection) / union_size if union_size > 0 else 0.0
                    )

        scores[start:end] = batch_scores

    return compute_metrics(scores, labels)


def build_adjacency_with_nan(
    g_train: nx.Graph, g_full: nx.Graph, all_nodes: list
) -> np.ndarray:
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n = len(all_nodes)

    adj = np.zeros((n, n), dtype=np.float32)

    has_weights = any("weight" in d for _, _, d in g_full.edges(data=True))
    train_edges = {
        tuple(sorted([node_to_idx[u], node_to_idx[v]])) for u, v in g_train.edges()
    }

    for u, v, data in g_full.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        i_sorted, j_sorted = sorted([i, j])
        edge_key = (i_sorted, j_sorted)
        weight = data.get("weight", 1.0) if has_weights else 1.0

        if edge_key in train_edges:
            adj[i, j] = weight
            adj[j, i] = weight
        else:
            adj[i, j] = np.nan
            adj[j, i] = np.nan

    np.fill_diagonal(adj, np.nan)

    return adj


def evaluate_fold(
    fold_idx: int,
    g_train: nx.Graph,
    g_full: nx.Graph,
    rank: int,
    seed: int,
    compute_baselines: bool,
    compute_srf: bool,
    is_weighted: bool,
) -> dict:
    all_nodes = sorted(g_full.nodes())

    pairs, labels = build_test_set(g_train, g_full)

    results = {"fold": fold_idx, "n_test_pairs": len(pairs)}

    if compute_baselines:
        a_sparse = build_sparse_adjacency(g_train, all_nodes)
        for method in ["CN", "AA", "RA", "JC"]:
            metrics = evaluate_baseline_fast(a_sparse, pairs, labels, method)
            results.update({f"{method.lower()}_{k}": v for k, v in metrics.items()})

    if compute_srf:
        adj = build_adjacency_with_nan(g_train, g_full, all_nodes)
        metrics = evaluate_srf(adj, pairs, labels, rank, seed + fold_idx, is_weighted)
        results.update({f"srf_{k}": v for k, v in metrics.items()})

    return results


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    idx = np.argsort(scores)[::-1]
    sorted_labels = labels[idx]

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    k = 500
    p_at_500 = sorted_labels[:k].sum() / min(k, len(sorted_labels))

    n_positives = labels.sum()
    if n_positives > 0:
        dcg = np.sum(sorted_labels / np.log2(np.arange(2, len(sorted_labels) + 2)))
        idcg = np.sum(1.0 / np.log2(np.arange(2, n_positives + 2)))
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return {"auroc": auroc, "auprc": auprc, "p500": p_at_500, "ndcg": ndcg}


def build_test_set(
    g_train: nx.Graph, g_full: nx.Graph
) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate_srf(
    adj: np.ndarray,
    pairs: np.ndarray,
    labels: np.ndarray,
    rank: int,
    seed: int,
    is_weighted: bool,
) -> dict[str, float]:
    if not srf_available:
        return {"auroc": 0.0, "auprc": 0.0, "p500": 0.0, "ndcg": 0.0}

    verbose = 0 if seed != 0 else 1

    model = SRF(
        rank=rank,
        rho=3.0,
        # bounds=(
        #     1e-10,
        #     1.0,
        # ),  # note important to set bounds to 1e-10 and 1.0 for KL loss for this count data!
        bounds=(0.0, 1.0),
        # bounds=(0.0, np.inf),
        max_outer=300,
        max_inner=50,
        tol=1e-4,
        verbose=verbose,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        # loss="kullback-leibler" if not is_weighted else "frobenius",
        loss="frobenius",
        # loss="binary-cross-entropy",
        # loss="bce",
        # loss="kullback-leibler",
    )

    w = model.fit_transform(adj)

    np.save("w_string.npy", w)

    scores = np.sum(w[pairs[:, 0]] * w[pairs[:, 1]], axis=1)

    return compute_metrics(scores, labels)


def kfold_cv(g: nx.Graph, n_folds: int, seed: int) -> list:
    edges = np.array(g.edges())
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
        g_train.add_edges_from(train_edges)

        if any("weight" in d for _, _, d in g.edges(data=True)):
            for u, v in train_edges:
                if g.has_edge(u, v):
                    g_train[u][v]["weight"] = g[u][v]["weight"]

        folds.append((g_train, g))

    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--compute", type=str, default="both", choices=["baselines", "srf", "both"]
    )
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    compute_baselines = args.compute in ("baselines", "both")
    compute_srf = args.compute in ("srf", "both")

    g, is_weighted = load_network(args.csv)
    print(
        f"{g.number_of_nodes()} nodes | {g.number_of_edges()} edges | "
        f"weighted={is_weighted}"
    )

    folds = kfold_cv(g, args.n_folds, args.seed)

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(evaluate_fold)(
            i,
            g_train,
            g_full,
            args.rank,
            args.seed,
            compute_baselines,
            compute_srf,
            is_weighted,
        )
        for i, (g_train, g_full) in enumerate(folds)
    )

    df = pd.DataFrame(results)

    if args.output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
        output_dir = output_dir / datetime.now().strftime("%y%m%d/%H%M%S")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)

    metric_cols = [c for c in df.columns if c not in ["fold", "n_test_pairs"]]
    summary = pd.DataFrame(
        {
            "metric": metric_cols,
            "mean": [df[c].mean() for c in metric_cols],
            "std": [df[c].std() for c in metric_cols],
        }
    )

    summary.to_csv(output_dir / "summary.csv", index=False)

    print(f"\n{len(df)} folds | {df['n_test_pairs'].mean():.0f} avg test pairs")
    print(summary.to_string(index=False))
    print(f"→ {output_dir}")


if __name__ == "__main__":
    main()
