#!/usr/bin/env python3
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analyses.ppi import (
    load_splits_from_csv,
    evaluate_srf,
    evaluate_baselines,
    evaluate_skipgnn,
)


def load_full_huri(splits_dir: Path, dataset: str):
    split_data = load_splits_from_csv(splits_dir / dataset, fold_idx=0)
    nodes = split_data["nodes"]

    all_edges = []
    for fold_idx in range(10):
        fold_data = load_splits_from_csv(splits_dir / dataset, fold_idx)
        all_edges.extend(fold_data["train_edges"])
        all_edges.extend(fold_data["test_edges"])

    unique_edges = list(set(tuple(sorted(e)) for e in all_edges))

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(unique_edges)

    return g, nodes, unique_edges


def generate_non_edges(g, nodes, max_pairs=None):
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges_set = {tuple(sorted([node_to_idx[u], node_to_idx[v]])) for u, v in g.edges()}

    i_idx, j_idx = np.triu_indices(n, k=1)
    all_pairs = list(zip(i_idx, j_idx))

    non_edges = [p for p in all_pairs if p not in edges_set]

    if max_pairs and len(non_edges) > max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(non_edges), size=max_pairs, replace=False)
        non_edges = [non_edges[i] for i in indices]

    return np.array(non_edges)


def evaluate_method(
    method: str,
    nodes: list,
    train_edges: list,
    non_edges: np.ndarray,
    test_labels: np.ndarray,
    rank: int,
    skipgnn_epochs: int,
    seed: int,
) -> tuple[str, np.ndarray]:
    print(f"[Progress] Evaluating {method.upper()}")

    if method == "srf":
        _, scores = evaluate_srf(
            nodes,
            train_edges,
            test_edges=[],
            test_pairs=non_edges,
            test_labels=test_labels,
            rank=rank,
            seed=seed,
            verbose=False,
        )
    elif method in ["cn", "aa", "ra", "jc"]:
        baseline_methods = [method.upper()]
        _, predictions = evaluate_baselines(
            nodes, train_edges, non_edges, test_labels, baseline_methods
        )
        scores = predictions[method]
    elif method == "skipgnn":
        _, scores = evaluate_skipgnn(
            nodes,
            train_edges,
            non_edges,
            test_labels,
            rank=rank,
            epochs=skipgnn_epochs,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[Progress] Completed {method.upper()}")
    return method, scores


def save_top_k(nodes, non_edges, scores, output_file, k=500):
    top_idx = np.argsort(scores)[::-1][:k]
    top_pairs = non_edges[top_idx]
    top_scores = scores[top_idx]

    df = pd.DataFrame(
        {
            "source": [nodes[i] for i in top_pairs[:, 0]],
            "target": [nodes[j] for j in top_pairs[:, 1]],
            "score": top_scores,
        }
    )

    df.to_csv(output_file, index=False)
    print(f"Saved top-{k} to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuRI")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/ppi/splits"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/ppi/outputs/link_prediction"),
    )
    parser.add_argument(
        "--methods", nargs="+", default=["srf", "cn", "aa", "ra", "jc", "skipgnn"]
    )
    parser.add_argument("--top_k", nargs="+", type=int, default=[500, 1000, 5000])
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--skipgnn_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading full {args.dataset} network...")
    g, nodes, train_edges = load_full_huri(args.splits_dir, args.dataset)
    print(f"Network: {len(nodes)} nodes, {g.number_of_edges()} edges")

    print("Generating non-edges...")
    non_edges = generate_non_edges(g, nodes)
    print(f"Generated {len(non_edges)} non-edge pairs")

    test_labels = np.zeros(len(non_edges))

    output_dir = args.output_dir / args.dataset / "topk"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nEvaluating {len(args.methods)} methods in parallel (n_jobs={args.n_jobs})..."
    )
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_method)(
            method,
            nodes,
            train_edges,
            non_edges,
            test_labels,
            args.rank,
            args.skipgnn_epochs,
            args.seed,
        )
        for method in args.methods
    )

    for method, scores in results:
        print(f"\nSaving top-k predictions for {method.upper()}...")
        for k in args.top_k:
            output_file = output_dir / f"{method}_top{k}.csv"
            save_top_k(nodes, non_edges, scores, output_file, k=k)


if __name__ == "__main__":
    main()
