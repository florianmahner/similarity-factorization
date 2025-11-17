from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from analyses.ppi.evaluators import (
    evaluate_baselines,
    evaluate_skipgnn,
    evaluate_srf,
)
from analyses.ppi.splits import load_splits_from_csv


def _evaluate_fold_method(
    fold_idx: int,
    method: str,
    dataset: str,
    splits_dir: Path,
    rank: int,
    seed: int,
    skipgnn_epochs: int,
    seal_hop: int,
    seal_epochs: int,
    baseline_methods: list[str] | None,
) -> list[dict]:
    split = load_splits_from_csv(splits_dir / dataset, fold_idx)
    nodes = split["nodes"]
    train_edges = split["train_edges"]
    test_edges = split["test_edges"]
    test_pairs = split["test_pairs"]
    test_labels = split["test_labels"]

    results: list[dict] = []
    if method == "srf":
        metrics, _ = evaluate_srf(
            nodes,
            train_edges,
            test_edges,
            test_pairs,
            test_labels,
            rank,
            seed + fold_idx,
            verbose=False,
        )
        results.extend(
            {
                "fold": fold_idx,
                "method": "srf",
                "metric": k,
                "value": v,
            }
            for k, v in metrics.items()
        )
    elif method in {"cn", "aa", "ra", "jc"}:
        metrics, _ = evaluate_baselines(
            nodes,
            train_edges,
            test_pairs,
            test_labels,
            baseline_methods or [method.upper()],
        )
        for key, value in metrics.items():
            method_name, metric = key.split("_", 1)
            if method_name.lower() == method:
                results.append(
                    {
                        "fold": fold_idx,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
    elif method == "skipgnn":
        metrics, _ = evaluate_skipgnn(
            nodes,
            train_edges,
            test_pairs,
            test_labels,
            rank,
            skipgnn_epochs,
            seed,
        )
        results.extend(
            {
                "fold": fold_idx,
                "method": "skipgnn",
                "metric": k,
                "value": v,
            }
            for k, v in metrics.items()
        )
    return results


def run_link_prediction(
    dataset: str,
    splits_dir: Path,
    rank: int,
    seed: int,
    n_folds: int,
    methods: list[str],
    skipgnn_epochs: int,
    seal_hop: int,
    seal_epochs: int,
    n_jobs: int,
) -> pd.DataFrame:
    baseline_methods = [m.upper() for m in methods if m in {"cn", "aa", "ra", "jc"}]
    tasks = [
        (
            fold_idx,
            method,
            dataset,
            splits_dir,
            rank,
            seed,
            skipgnn_epochs,
            seal_hop,
            seal_epochs,
            baseline_methods if method in {"cn", "aa", "ra", "jc"} else None,
        )
        for fold_idx in range(n_folds)
        for method in methods
    ]

    outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_evaluate_fold_method)(*task) for task in tasks
    )
    flat = [row for rows in outputs for row in rows]
    df = pd.DataFrame(flat)
    return df.pivot_table(index=["fold", "method"], columns="metric", values="value").reset_index()

