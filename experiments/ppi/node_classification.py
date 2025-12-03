from __future__ import annotations

import pickle
from pathlib import Path

import mygene
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .utils import load_network


def fetch_go_annotations(
    proteins: np.ndarray, cache_path: Path
) -> dict[str, list[str]]:
    """Fetch GO annotations using MyGene API with local caching."""
    if cache_path.exists():
        print(f"Loading GO annotations from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Fetching GO annotations from MyGene...")
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        proteins.tolist(),
        scopes="symbol,ensembl.protein",
        fields="go",
        species="human",
        verbose=False,
    )

    annotations = {}
    for entry in res:
        if "go" not in entry:
            continue
        go_terms = []
        for ont in ["MF", "BP"]:
            if ont in entry["go"]:
                terms = entry["go"][ont]
                if isinstance(terms, list):
                    go_terms.extend(
                        [t["id"] for t in terms if isinstance(t, dict) and "id" in t]
                    )
                elif isinstance(terms, dict) and "id" in terms:
                    go_terms.append(terms["id"])

        if go_terms:
            annotations[entry["query"]] = list(set(go_terms))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(annotations, f)

    return annotations


def build_label_matrix(
    proteins: np.ndarray, annotations: dict[str, list[str]]
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build binary label matrix from GO annotations."""
    all_terms = sorted(set(term for terms in annotations.values() for term in terms))
    term_to_idx = {term: idx for idx, term in enumerate(all_terms)}

    Y = np.zeros((len(proteins), len(all_terms)), dtype=np.int8)
    valid_indices = []

    for prot_idx, protein in enumerate(proteins):
        if protein in annotations:
            valid_indices.append(prot_idx)
            for term in annotations[protein]:
                Y[prot_idx, term_to_idx[term]] = 1

    return Y, all_terms, valid_indices


def filter_terms(
    Y: np.ndarray, terms: list[str], min_size: int, max_size: int
) -> tuple[np.ndarray, list[str]]:
    """Filter GO terms by annotation count."""
    term_counts = Y.sum(axis=0)
    valid = (term_counts >= min_size) & (term_counts <= max_size)
    return Y[:, valid], [t for t, v in zip(terms, valid) if v]


def fit_pysrf(
    nodes: list, edges: list, rank: int, seed: int, max_outer: int = 150
) -> dict[str, np.ndarray]:
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Build adjacency matrix (symmetric, undirected)
    rows = []
    cols = []
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            rows.extend([u_idx, v_idx])
            cols.extend([v_idx, u_idx])

    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Fit SRF to get node embeddings
    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=max_outer,
        max_inner=50,
        tol=1e-5,
        verbose=1,
        init="random_sqrt",
        random_state=seed,
        missing_values=np.nan,
        loss="frobenius",
    )

    # Convert to dense for SRF (handles 0s as observed values)
    adj_dense = adj.toarray()

    # Explicitly set diagonal to NaN (Self-loops should be ignored/missing)
    # This prevents SRF from fitting the trivial diagonal=0 structure
    np.fill_diagonal(adj_dense, np.nan)

    W = model.fit_transform(adj_dense)
    embeddings_dict = {node: W[idx] for node, idx in node_to_idx.items()}

    return embeddings_dict


def _evaluate_method(
    nodes: list,
    edges: list,
    label_dict: dict[str, list[str]],
    labeled_nodes: list,
    rank: int,
    test_ratio: float,
    seed: int,
    run_idx: int,
) -> dict:
    # Generate embeddings using SRF
    embeddings_dict = fit_pysrf(
        nodes=nodes, edges=edges, rank=rank, seed=seed + run_idx
    )

    # Extract embeddings for labeled nodes
    X = np.array([embeddings_dict.get(node, np.zeros(rank)) for node in labeled_nodes])
    y = [label_dict[node] for node in labeled_nodes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed + run_idx
    )

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    model = OneVsRestClassifier(
        LogisticRegression(random_state=seed + run_idx, max_iter=1000, solver="lbfgs")
    )
    model.fit(X_train, y_train_bin)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_bin, y_pred)
    micro_f1 = f1_score(y_test_bin, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test_bin, y_pred, average="macro", zero_division=0)

    return {
        "run": run_idx,
        "method": "srf",
        "rank": rank,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_labels": len(mlb.classes_),
    }


def run(cfg: DictConfig) -> None:
    # Load network
    string_path = Path(cfg.data_dir) / "STRING_human_min900_v12.csv"
    g = load_network(string_path)
    nodes = sorted(g.nodes())
    edges = list(g.edges())

    # Clean protein IDs: strip species prefix (e.g., "9606.ENSP..." -> "ENSP...")
    proteins_raw = np.array(nodes)
    proteins_clean = np.array(
        [p.split(".")[-1] if "." in p else p for p in proteins_raw]
    )

    print(f"Loaded network: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Sample protein IDs: {proteins_clean[:5].tolist()}")

    # Fetch GO annotations (using clean IDs)
    cache = Path(cfg.data_dir) / "go_annotations_human.pkl"
    annotations = fetch_go_annotations(proteins_clean, cache)
    print(f"GO annotations available for: {len(annotations)} proteins")

    # Build label matrix (using clean protein IDs)
    Y, terms, valid_indices = build_label_matrix(proteins_clean, annotations)
    print(f"Label matrix: {len(valid_indices)} proteins × {len(terms)} GO terms")

    # Only use proteins with annotations for filtering
    Y_valid = Y[valid_indices]
    # Map back to original node IDs (with species prefix) for network operations
    labeled_nodes_all = [nodes[i] for i in valid_indices]

    # Define 3 bins
    bins = [(11, 30), (31, 100), (101, 300)]
    all_results = []

    print(f"\nRunning {cfg.n_runs} trials for 3 term size bins...\n")

    for min_size, max_size in bins:
        # Filter terms by size (using only annotated proteins)
        Y_filtered, terms_filtered = filter_terms(Y_valid, terms, min_size, max_size)

        if Y_filtered.shape[1] == 0:
            print(f"Bin [{min_size}-{max_size}]: No terms found, skipping")
            continue

        print(
            f"Bin [{min_size}-{max_size}]: {Y_filtered.shape[0]} proteins × {Y_filtered.shape[1]} GO terms"
        )

        # Build label dict for this bin
        label_dict = {
            protein: [
                terms_filtered[j]
                for j in range(Y_filtered.shape[1])
                if Y_filtered[i, j]
            ]
            for i, protein in enumerate(labeled_nodes_all)
        }

        # Filter out proteins with no terms in this bin
        labeled_nodes_with_terms = [p for p in labeled_nodes_all if label_dict[p]]
        label_dict = {p: label_dict[p] for p in labeled_nodes_with_terms}

        if len(labeled_nodes_with_terms) < 100:
            print(
                f"Bin [{min_size}-{max_size}]: Too few labeled nodes ({len(labeled_nodes_with_terms)}), skipping"
            )
            continue

        # Create tasks for parallel execution
        tasks = [
            (
                nodes,
                edges,
                label_dict,
                labeled_nodes_with_terms,
                cfg.rank,
                cfg.test_ratio,
                cfg.seed,
                run_idx,
            )
            for run_idx in range(cfg.n_runs)
        ]

        # Run in parallel
        results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
            delayed(_evaluate_method)(*task) for task in tasks
        )

        # Add bin label to results
        for r in results:
            r["bin"] = f"[{min_size}-{max_size}]"

        all_results.extend(results)

        # Print bin summary
        df_bin = pd.DataFrame(results)
        print(
            f"Bin [{min_size}-{max_size}] mean: {df_bin[['accuracy', 'micro_f1', 'macro_f1']].mean().to_dict()}\n"
        )

    if not all_results:
        raise ValueError("No results generated for any bin")

    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(Path.cwd() / "results.csv", index=False)

    # Print summary by bin
    print("\nResults summary by bin:")
    summary = df.groupby("bin")[["accuracy", "micro_f1", "macro_f1"]].agg(
        ["mean", "std"]
    )
    print(summary)
    summary.to_csv(Path.cwd() / "results_summary.csv")
