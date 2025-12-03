#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot node classification benchmarks from CSV.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to write plots (default: results parent)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.results
    outdir = args.outdir if args.outdir else results_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="talk")
    df = pd.read_csv(results_path)
    if df.empty:
        return

    min_score = max(0, df["Micro-F1"].min() * 0.9)
    max_score = min(1.0, df["Micro-F1"].max() * 1.05)

    default_nodes = df["Nodes"].max()
    default_rank = df["Rank"].max()

    subset = df[(df["Nodes"] == default_nodes) & (df["Rank"] == default_rank)]
    if not subset.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=subset, x="Bin", y="Micro-F1", hue="Method", palette="viridis")
        plt.xlabel("Label Sparsity Bin")
        plt.ylabel("Micro-F1 Score")
        plt.ylim(min_score, max_score)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig(outdir / "method_comparison.png", dpi=300)
        plt.close()

    if df["Nodes"].nunique() > 1:
        subset = df[df["Rank"] == default_rank]
        if not subset.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=subset,
                x="Nodes",
                y="Micro-F1",
                hue="Method",
                style="Bin",
                markers=True,
                palette="magma",
            )
            plt.xlabel("Number of Nodes")
            plt.ylabel("Micro-F1 Score")
            plt.ylim(min_score, max_score)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
            sns.despine()
            plt.tight_layout()
            plt.savefig(outdir / "scaling_nodes.png", dpi=300)
            plt.close()

    if df["Rank"].nunique() > 1:
        subset = df[df["Nodes"] == default_nodes]
        if not subset.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=subset,
                x="Rank",
                y="Micro-F1",
                hue="Method",
                style="Bin",
                markers=True,
                palette="magma",
            )
            plt.xlabel("Embedding Rank")
            plt.ylabel("Micro-F1 Score")
            plt.ylim(min_score, max_score)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
            sns.despine()
            plt.tight_layout()
            plt.savefig(outdir / "scaling_rank.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    main()
