"""
Aggregate and plot node classification benchmark results.

Usage:
    poetry run python plot_node_classification.py \
        --results-dir experiments/ppi/outputs/node_classification \
        --output-dir experiments/ppi/outputs/node_classification/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate all result.json files from sweep directories."""
    records = []
    for json_file in results_dir.rglob("result.json"):
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
    return pd.DataFrame(records)


def plot_by_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison plots for each dataset."""
    datasets = df["dataset"].unique()

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]

        if dataset == "ppi":
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, eval_key in zip(axes, ["rare", "medium", "frequent"]):
                data = subset[subset["eval_key"] == eval_key]
                if data.empty:
                    continue
                sns.barplot(data=data, x="method", y="micro_f1", hue="rank", ax=ax)
                ax.set_title(f"{dataset.upper()} - {eval_key} GO terms")
                ax.set_xlabel("Method")
                ax.set_ylabel("Micro F1")
                ax.set_ylim(0, 1)
                ax.legend(title="Rank", loc="lower right")
            plt.tight_layout()
            fig.savefig(output_dir / f"{dataset}_f1_by_gobin.png", dpi=150)
            plt.close(fig)
        else:
            fig, axes = plt.subplots(1, len(subset["train_ratio"].unique()), figsize=(14, 4))
            if not hasattr(axes, "__iter__"):
                axes = [axes]
            for ax, tr in zip(axes, sorted(subset["train_ratio"].unique())):
                data = subset[subset["train_ratio"] == tr]
                sns.barplot(data=data, x="method", y="micro_f1", hue="rank", ax=ax)
                ax.set_title(f"{dataset.upper()} - Train ratio {tr}")
                ax.set_xlabel("Method")
                ax.set_ylabel("Micro F1")
                ax.set_ylim(0, 1)
                ax.legend(title="Rank", loc="lower right")
            plt.tight_layout()
            fig.savefig(output_dir / f"{dataset}_f1_by_trainratio.png", dpi=150)
            plt.close(fig)


def plot_size_scaling(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance vs graph size for each method."""
    datasets = df["dataset"].unique()

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        if subset["size"].nunique() <= 1:
            continue

        if dataset == "ppi":
            metric_col = subset[subset["eval_key"] == "medium"]
        else:
            metric_col = subset[subset["train_ratio"] == 0.5]

        if metric_col.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=metric_col,
            x="size",
            y="micro_f1",
            hue="method",
            style="rank",
            markers=True,
            ax=ax,
        )
        ax.set_title(f"{dataset.upper()} - F1 vs Graph Size")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Micro F1")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(output_dir / f"{dataset}_f1_vs_size.png", dpi=150)
        plt.close(fig)


def plot_runtime(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot runtime comparison."""
    runtime_df = df.drop_duplicates(subset=["dataset", "method", "size", "rank", "seed"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=runtime_df, x="method", y="runtime", hue="dataset", ax=ax)
    ax.set_title("Runtime Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Runtime (seconds)")
    ax.legend(title="Dataset")
    plt.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot node classification benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/ppi/outputs/node_classification"),
        help="Directory containing result.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: results-dir/plots)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate_results(results_dir)
    if df.empty:
        print("No results found")
        return

    csv_path = output_dir / "aggregated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated results to {csv_path}")
    print(f"Found {len(df)} records from {df['dataset'].nunique()} datasets")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Sizes: {sorted(df['size'].unique())}")
    print(f"Ranks: {sorted(df['rank'].unique())}")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    plot_by_dataset(df, output_dir)
    plot_size_scaling(df, output_dir)
    plot_runtime(df, output_dir)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
