#!/usr/bin/env python3
"""Prepare k-fold splits for PPI link prediction experiments.

Creates CSV-based splits for C. elegans, HURI, and STRING datasets.
Same splits are used for both GNNs and other methods.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analyses.ppi import kfold_cv, build_test_set, load_network


def process_fold(
    fold_idx: int,
    g_train: object,
    g_full: object,
    all_nodes: list,
    splits_dir: Path,
) -> int:
    """Process a single fold and save to CSV files.

    Returns:
        Number of training edges
    """
    train_edges = list(g_train.edges())

    train_edges_df = pd.DataFrame(train_edges, columns=["source", "target"])
    train_edges_df.to_csv(splits_dir / f"fold{fold_idx}_train_edges.csv", index=False)

    test_pairs, test_labels = build_test_set(g_train, g_full)

    test_edges = [(all_nodes[i], all_nodes[j]) for i, j in test_pairs[test_labels == 1]]
    test_edges_df = pd.DataFrame(test_edges, columns=["source", "target"])
    test_edges_df.to_csv(splits_dir / f"fold{fold_idx}_test_edges.csv", index=False)

    test_pairs_df = pd.DataFrame(
        {
            "node_i": test_pairs[:, 0],
            "node_j": test_pairs[:, 1],
            "label": test_labels,
        }
    )
    test_pairs_df.to_csv(splits_dir / f"fold{fold_idx}_test_pairs.csv", index=False)

    print(
        f"  Fold {fold_idx}: {len(train_edges)} train edges, "
        f"{len(test_edges)} test positives, {len(test_pairs)} total test pairs"
    )

    return len(train_edges)


def prepare_dataset_splits(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    n_folds: int,
    seed: int,
    n_jobs: int = -1,
):
    """Prepare k-fold splits for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (C.elegans, HURI, STRING)
    data_dir : Path
        Directory containing input CSV files
    output_dir : Path
        Output directory for splits
    n_folds : int
        Number of folds
    seed : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    """
    input_file = data_dir / f"{dataset}.csv"
    if not input_file.exists():
        print(f"Warning: {input_file} not found, skipping {dataset}")
        return

    print(f"\nPreparing splits for {dataset}...")
    g, has_weights = load_network(input_file)

    breakpoint()
    print(f"  Loaded: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    splits_dir = output_dir / dataset
    splits_dir.mkdir(parents=True, exist_ok=True)

    folds = kfold_cv(g, n_folds, seed)

    all_nodes = sorted(g.nodes())
    nodes_df = pd.DataFrame({"node_id": all_nodes})
    nodes_df.to_csv(splits_dir / "nodes.csv", index=False)

    print(f"  Processing {n_folds} folds in parallel...")
    n_edges_per_fold = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_fold)(fold_idx, g_train, g_full, all_nodes, splits_dir)
        for fold_idx, (g_train, g_full) in enumerate(folds)
    )

    metadata = {
        "dataset": dataset,
        "n_folds": n_folds,
        "seed": seed,
        "n_nodes": len(all_nodes),
        "n_edges_total": g.number_of_edges(),
        "n_edges_per_fold": n_edges_per_fold,
        "has_weights": has_weights,
    }

    with open(splits_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved splits to {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PPI data splits")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["C.elegans", "HuRI", "STRING_human_min900_v12"],
        help="Datasets to prepare",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ppi"),
        help="Directory with input CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ppi/splits"),
        help="Output directory for splits",
    )
    parser.add_argument("--n-folds", type=int, default=10, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel jobs (-1 for all cores)"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        prepare_dataset_splits(
            dataset,
            args.data_dir,
            args.output_dir,
            args.n_folds,
            args.seed,
            args.n_jobs,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
