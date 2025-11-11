#!/usr/bin/env python3
"""PPI Link Prediction and CORUM Validation Experiment."""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.ppi import (
    load_network,
    build_adjacency_with_nan,
    load_splits_from_csv,
    evaluate_srf,
    evaluate_baselines,
    evaluate_skipgnn,
    map_string_ids_to_genes,
)


def evaluate_fold_method(
    fold_idx: int,
    method: str,
    dataset: str,
    splits_dir: Path,
    rank: int,
    seed: int,
    skipgnn_epochs: int,
    seal_hop: int = 1,
    seal_epochs: int = 50,
    baseline_methods: list[str] | None = None,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate a method on a fold with progress reporting.

    Returns:
        (results, timing_info)
    """
    print(f"[Progress] Starting fold {fold_idx}, method {method}")
    t_start = time.time()

    t_load = time.time()
    split_data = load_splits_from_csv(splits_dir / dataset, fold_idx)
    t_load = time.time() - t_load

    nodes = split_data["nodes"]
    train_edges = split_data["train_edges"]
    test_edges = split_data["test_edges"]
    test_pairs = split_data["test_pairs"]
    test_labels = split_data["test_labels"]

    results = []
    predictions = None
    t_eval_start = time.time()

    if method == "srf":
        metrics, predictions = evaluate_srf(
            nodes,
            train_edges,
            test_edges,
            test_pairs,
            test_labels,
            rank,
            seed + fold_idx,
            verbose=verbose and fold_idx == 0,
        )
        for k, v in metrics.items():
            results.append({"fold": fold_idx, "method": "srf", "metric": k, "value": v})

    elif method in ["cn", "aa", "ra", "jc"]:
        if baseline_methods is None:
            baseline_methods = [method.upper()]
        baseline_results, baseline_predictions = evaluate_baselines(
            nodes, train_edges, test_pairs, test_labels, baseline_methods
        )
        for key, value in baseline_results.items():
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
        predictions = baseline_predictions.get(method.lower())

    elif method == "skipgnn":
        metrics, predictions = evaluate_skipgnn(
            nodes, train_edges, test_pairs, test_labels, rank, skipgnn_epochs, seed
        )
        for k, v in metrics.items():
            results.append(
                {"fold": fold_idx, "method": "skipgnn", "metric": k, "value": v}
            )
    t_eval = time.time() - t_eval_start
    t_total = time.time() - t_start

    # Save predictions if output_dir is provided
    if output_dir is not None and predictions is not None:
        # Ensure predictions match test_pairs length
        if len(predictions) == len(test_pairs):
            pred_dir = output_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)

            pred_file = pred_dir / f"fold{fold_idx}_{method}_predictions.csv"
            pred_df = pd.DataFrame(
                {
                    "node_i": test_pairs[:, 0],
                    "node_j": test_pairs[:, 1],
                    "label": test_labels,
                    "score": predictions,
                }
            )
            pred_df.to_csv(pred_file, index=False)
        else:
            print(
                f"Warning: Predictions length ({len(predictions)}) != test_pairs length ({len(test_pairs)}) for fold {fold_idx}, method {method}"
            )

    print(
        f"[Progress] Completed fold {fold_idx}, method {method} (took {t_total:.1f}s)"
    )

    timing = {
        "fold": fold_idx,
        "method": method,
        "load_time": t_load,
        "eval_time": t_eval,
        "total_time": t_total,
    }

    return results, timing


def run_link_prediction(
    dataset: str,
    splits_dir: Path,
    output_dir: Path,
    n_folds: int,
    rank: int,
    seed: int,
    n_jobs: int,
    methods: list[str],
    skipgnn_epochs: int,
    seal_hop: int = 1,
    seal_epochs: int = 50,
):
    """Run link prediction experiment with parallel evaluation."""
    print(f"\n=== Link Prediction: {dataset} ===")
    print(f"Methods: {methods}")
    print(f"Folds: {n_folds}")
    print(f"Parallel jobs: {n_jobs}")

    baseline_methods = [m.upper() for m in methods if m in ["cn", "aa", "ra", "jc"]]

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
            baseline_methods if method in ["cn", "aa", "ra", "jc"] else None,
            True,  # verbose
            output_dir,
        )
        for fold_idx in range(n_folds)
        for method in methods
    ]

    print(f"\nRunning {len(tasks)} tasks in parallel...")
    t_parallel_start = time.time()

    all_outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_fold_method)(*task) for task in tasks
    )

    t_parallel = time.time() - t_parallel_start

    results_flat = []
    timings = []
    for results, timing in all_outputs:
        results_flat.extend(results)
        timings.append(timing)
    results_df = pd.DataFrame(results_flat)
    results_df = results_df.pivot_table(
        index=["fold", "method"], columns="metric", values="value"
    ).reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "results.csv", index=False)

    timings_df = pd.DataFrame(timings)
    timings_df.to_csv(output_dir / "timings.csv", index=False)

    print(f"\nResults saved to {output_dir / 'results.csv'}")
    print(f"Timings saved to {output_dir / 'timings.csv'}")
    print(f"Predictions saved to {output_dir / 'predictions'}/")

    print(f"\n=== Performance Summary ===")
    print(f"Total parallel time: {t_parallel:.2f}s")
    print(f"Average time per task: {t_parallel / len(tasks):.2f}s")
    if len(timings_df) > 0:
        print(f"\nTime breakdown by method:")
        method_times = timings_df.groupby("method").agg(
            {
                "load_time": "mean",
                "eval_time": "mean",
                "total_time": "mean",
            }
        )
        print(method_times.round(2))

    if len(results_df) > 0:
        print(f"\n=== Results Summary ===")
        summary_cols = ["auroc", "auprc", "p500", "ndcg"]
        available_cols = [c for c in summary_cols if c in results_df.columns]
        if available_cols:
            print(results_df.groupby("method")[available_cols].mean())


def run_corum_validation(
    string_data_path: Path,
    output_dir: Path,
    rank: int,
    seed: int,
    load_model: Path | None,
):
    """Train SRF on STRING network and save embedding for CORUM validation."""
    if string_data_path.suffix == ".csv":
        g, _ = load_network(string_data_path)
        nodes = sorted(g.nodes())
        adj = build_adjacency_with_nan(g, g, nodes, fill_missing_with_nan=False)
    elif string_data_path.suffix == ".npy":
        adj = np.load(string_data_path)
        proteins_file = string_data_path.parent / "proteins.txt"
        nodes = (
            np.loadtxt(proteins_file, dtype=str).tolist()
            if proteins_file.exists()
            else [f"protein_{i}" for i in range(adj.shape[0])]
        )
    else:
        raise ValueError(f"Unknown file format: {string_data_path.suffix}")

    if load_model is not None:
        from joblib import load

        model = load(load_model)
        embedding = model.transform(adj)
    else:
        model = SRF(
            rank=rank,
            rho=3.0,
            max_outer=2000,
            max_inner=50,
            tol=1e-4,
            verbose=1,
            init="random_sqrt",
            random_state=seed,
            missing_values=np.nan,
            loss="frobenius",
        )
        embedding = model.fit_transform(adj)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embedding.npy", embedding)
    np.savetxt(output_dir / "proteins.txt", nodes, fmt="%s")

    if load_model is None:
        from joblib import dump

        dump(model, output_dir / "model.joblib")


def main():
    parser = argparse.ArgumentParser(description="PPI experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["link_prediction", "corum_validation"],
        required=True,
        help="Experiment mode",
    )

    parser.add_argument("--dataset", type=str, help="Dataset name (link_prediction)")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/ppi/splits"),
        help="Directory with CSV splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument("--n-folds", type=int, default=10, help="Number of folds")
    parser.add_argument("--rank", type=int, default=10, help="SRF rank")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["srf", "cn", "aa", "ra", "jc", "skipgnn"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--skipgnn-epochs", type=int, default=200, help="SkipGNN training epochs"
    )
    parser.add_argument(
        "--seal-hop", type=int, default=1, help="SEAL subgraph hop number"
    )
    parser.add_argument(
        "--seal-epochs", type=int, default=50, help="SEAL training epochs"
    )

    parser.add_argument(
        "--string-data",
        type=Path,
        help="STRING network path (corum_validation)",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Path to saved SRF model (optional)",
    )

    args = parser.parse_args()

    if args.mode == "link_prediction":

        args.output_dir = (
            Path(__file__).parent / "outputs" / "link_prediction" / args.dataset
        )
        if args.dataset is None:
            parser.error("--dataset required for link_prediction mode")

        run_link_prediction(
            args.dataset,
            args.splits_dir,
            args.output_dir,
            args.n_folds,
            args.rank,
            args.seed,
            args.n_jobs,
            args.methods,
            args.skipgnn_epochs,
            args.seal_hop,
            args.seal_epochs,
        )
    else:
        if args.string_data is None:
            parser.error("--string-data required for corum_validation mode")

        args.output_dir = Path(__file__).parent / "outputs" / "corum_validation"

        run_corum_validation(
            args.string_data,
            args.output_dir,
            args.rank,
            args.seed,
            args.load_model,
        )


if __name__ == "__main__":
    main()
