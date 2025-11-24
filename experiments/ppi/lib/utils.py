import copy
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.graphs import (
    build_undirected_graph,
    largest_component,
    load_edgelist_csv,
)

import numpy as np
import pandas as pd
from pathlib import Path
import json


import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


def map_ensg_to_gene_names(ensg_ids: list[str]) -> dict[str, str]:
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            ensg_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            returnall=True,
        )

        ensg_to_gene = {}
        for result in results["out"]:
            if "symbol" in result and "query" in result:
                ensg_to_gene[result["query"]] = result["symbol"]

        return ensg_to_gene
    except ImportError:
        print("Warning: mygene not available, trying pandas-based mapping")
        return {}


def load_network(csv_path: Path | str) -> tuple[nx.Graph, bool]:
    sources, targets, weights = load_edgelist_csv(csv_path, weight_col=None)
    edges = list(zip(sources, targets))
    g = build_undirected_graph(edges, weighted=False)

    return largest_component(g)


import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_link_prediction_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """
    Computes metrics using full sorting (User provided implementation).
    """
    # 1. Sort scores descending (High score = Index 0)
    # WARNING: This is O(N log N). On 200M edges, this might take RAM/Time.
    idx = np.argsort(scores)[::-1]
    sorted_labels = labels[idx]

    # 2. Sklearn Metrics
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # 3. P@500
    k = 500
    # Guard against case where list is shorter than k
    limit = min(k, len(sorted_labels))
    p_at_500 = sorted_labels[:k].sum() / limit

    # 4. NDCG (Full List)
    n_positives = labels.sum()
    if n_positives > 0:
        # DCG: Sum of (label / log2(rank+1)) for the whole list
        # Your code uses arange(2, ...+2) which corresponds to log2(rank+1) where rank starts at 1
        rank_discounts = np.log2(np.arange(2, len(sorted_labels) + 2))
        dcg = np.sum(sorted_labels / rank_discounts)

        # IDCG: Best possible case is all 1s at the top
        ideal_discounts = np.log2(np.arange(2, n_positives + 2))
        idcg = np.sum(1.0 / ideal_discounts)

        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return {"auroc": auroc, "auprc": auprc, "p500": p_at_500, "ndcg": ndcg}


import numpy as np
from scipy.sparse import csr_matrix


def evaluate_open_world(score_matrix, train_edges, test_pos_edges):
    """
    Prepares Open World data (Upper Triangle) and calls the metric function.
    """
    n_nodes = score_matrix.shape[0]

    # --- 1. Mask Training Edges ---
    # Set to -inf so they effectively disappear from the ranking
    # (or appear at the very bottom after sorting)
    score_matrix[train_edges[:, 0], train_edges[:, 1]] = -np.inf
    score_matrix[train_edges[:, 1], train_edges[:, 0]] = -np.inf
    np.fill_diagonal(score_matrix, -np.inf)

    # --- 2. Extract Upper Triangle (u < v) ---
    # This creates the "Open World" list of candidate pairs
    rows, cols = np.triu_indices(n_nodes, k=1)

    # Extract Scores (Dense Read)
    y_scores = score_matrix[rows, cols]

    # --- 3. Generate Ground Truth Labels (Fast) ---
    # We use a sparse matrix trick to map (u,v) -> 1 quickly
    # This is much faster than list comprehension for millions of edges
    test_pos_edges = np.sort(test_pos_edges, axis=1)  # Ensure u < v

    data = np.ones(len(test_pos_edges), dtype=np.int8)
    # Create sparse matrix of Test Edges
    test_mat = csr_matrix(
        (data, (test_pos_edges[:, 0], test_pos_edges[:, 1])), shape=(n_nodes, n_nodes)
    )

    # Extract Labels corresponding to the rows/cols we selected
    # .A1 converts the matrix result to a flat 1D numpy array
    y_true = test_mat[rows, cols].A1

    # --- 4. Filter Invalid Training Data ---
    # We remove the -inf entries entirely.
    # This ensures your argsort() isn't sorting millions of training edges unnecessarily.
    valid_mask = y_scores > -1e30

    y_scores_clean = y_scores[valid_mask]
    y_true_clean = y_true[valid_mask]

    # --- 5. Compute Metrics ---
    # Pass clean vectors to your function
    return compute_link_prediction_metrics(y_scores_clean, y_true_clean)


def prepare_splits(dataset, data_path, output_dir, n_folds, seed, n_jobs=1):
    """
    Pre-computes K-Fold splits.
    CRITICAL FIX: Enforces unique undirected edges to prevent data leakage/masking issues.
    """
    out_path = Path(output_dir) / dataset
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(data_path / f"{dataset}.csv")
    # omly use source and target columns
    df = df[["source", "target"]]
    df.columns = ["source", "target"]

    # Map nodes to integers 0..N-1
    unique_nodes = sorted(list(set(df.iloc[:, 0]) | set(df.iloc[:, 1])))
    node_map = {n: i for i, n in enumerate(unique_nodes)}
    n_nodes = len(unique_nodes)

    pd.DataFrame({"node_id": unique_nodes}).to_csv(out_path / "nodes.csv", index=False)

    # Convert to numpy
    raw_edges = np.array([[node_map[u], node_map[v]] for u, v in df.values])

    # --- CRITICAL DATA CLEANING ---
    # 1. Sort every row so (u, v) is always u < v
    #    This treats (A, B) and (B, A) as the exact same edge.
    raw_edges.sort(axis=1)

    # 2. Remove Duplicates
    #    If CSV had A->B and B->A, this keeps only one copy.
    edges = np.unique(raw_edges, axis=0)

    # 3. Remove Self-Loops
    #    Link prediction generally fails/behaves weirdly on self-loops
    edges = edges[edges[:, 0] != edges[:, 1]]
    # -------------------------------

    # 2. Shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(edges)

    # 3. Split and Save
    fold_size = len(edges) // n_folds

    for i in range(n_folds):
        test_start, test_end = i * fold_size, (i + 1) * fold_size

        test_edges = edges[test_start:test_end]
        # Train is everything else
        train_edges = np.concatenate([edges[:test_start], edges[test_end:]])

        np.savez_compressed(
            out_path / f"fold{i}.npz",
            train_edges=train_edges,
            test_pos_edges=test_edges,
        )

    with open(out_path / "metadata.json", "w") as f:
        json.dump({"n_nodes": n_nodes, "n_folds": n_folds}, f)


def load_fold_data(splits_dir, dataset, fold_idx):
    """Lightweight loader."""
    path = Path(splits_dir) / dataset
    data = np.load(path / f"fold{fold_idx}.npz")

    # We need n_nodes often, read it from metadata once or len(nodes.csv)
    # Loading the full node list is optional unless you need string IDs
    n_nodes = pd.read_csv(path / "nodes.csv").shape[0]

    return {
        "n_nodes": n_nodes,
        "train_edges": data["train_edges"],
        "test_pos_edges": data["test_pos_edges"],
    }


def get_balanced_samples(train_edges, n_nodes, seed, neg_ratio=1.0):
    """
    Core sampling logic. Returns raw arrays of positive and negative edges.
    Used by BOTH SRF (to build matrix) and SEAL (to build edge lists).
    """
    rng = np.random.RandomState(seed)

    # 1. Positives are just the training edges
    pos_edges = np.array(train_edges)

    # 2. Negative Sampling
    n_pos = len(train_edges)
    n_neg = int(n_pos * neg_ratio)

    # Use a set for fast lookup of existing edges
    # We sort the tuple to ensure (u, v) is treated same as (v, u)
    existing_set = set(tuple(sorted(e)) for e in train_edges)
    negatives = set()

    # Sample in bulk
    while len(negatives) < n_neg:
        # Sample slightly more than needed to account for collisions
        batch_size = int((n_neg - len(negatives)) * 1.5) + 100
        u = rng.randint(0, n_nodes, batch_size)
        v = rng.randint(0, n_nodes, batch_size)

        mask = u != v
        u, v = u[mask], v[mask]

        for i in range(len(u)):
            pair = tuple(sorted((u[i], v[i])))
            if pair not in existing_set:
                negatives.add(pair)
            if len(negatives) >= n_neg:
                break

    neg_edges = np.array(list(negatives))
    return pos_edges, neg_edges


def get_balanced_matrix(train_edges, n_nodes, seed, neg_ratio=1.0):
    """
    Constructs the dense matrix specifically for SRF/NMF using the shared sampler.
    1.0 = Positive, 0.0 = Negative, NaN = Missing
    """
    pos_edges, neg_edges = get_balanced_samples(train_edges, n_nodes, seed, neg_ratio)

    # Init with NaNs
    adj = np.full((n_nodes, n_nodes), np.nan, dtype=np.float32)

    # Fill Positives (Symmetric)
    adj[pos_edges[:, 0], pos_edges[:, 1]] = 1.0
    adj[pos_edges[:, 1], pos_edges[:, 0]] = 1.0

    # Fill Negatives (Symmetric)
    if len(neg_edges) > 0:
        adj[neg_edges[:, 0], neg_edges[:, 1]] = 0.0
        adj[neg_edges[:, 1], neg_edges[:, 0]] = 0.0

    np.fill_diagonal(adj, np.nan)
    return adj
