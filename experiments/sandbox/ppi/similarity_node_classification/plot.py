"""
Plotting script for node classification benchmark results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def main():
    results_path = Path.cwd() / "results.csv"
    if not results_path.exists():
        print(f"No results found at {results_path}")
        return

    df = pd.read_csv(results_path)

    method_order = ["SRF+binary", "SRF+CN", "SRF+AA", "deepwalk", "node2vec", "line"]
    method_order = [m for m in method_order if m in df["method"].unique()]

    palette = {
        "SRF+binary": "#3498db",
        "SRF+CN": "#2ecc71",
        "SRF+AA": "#e74c3c",
        "SRF+RA": "#9b59b6",
        "deepwalk": "#f39c12",
        "node2vec": "#1abc9c",
        "line": "#95a5a6",
    }

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    n_nodes = df["n_nodes"].iloc[0] if "n_nodes" in df.columns else "?"
    rank = df["rank"].iloc[0] if "rank" in df.columns else "?"

    # Plot 1: Micro-F1 by method and train ratio (grouped bar)
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    train_ratios = sorted(df["train_ratio"].unique())
    ratio_labels = {0.1: "10%", 0.5: "50%", 0.9: "90%"}
    df["train_label"] = df["train_ratio"].map(ratio_labels)

    pivot = df.pivot_table(index="method", columns="train_ratio", values="micro_f1")
    pivot = pivot.reindex(method_order)

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    pivot.plot(kind="bar", ax=ax1, color=colors, width=0.8, edgecolor="white", linewidth=1)

    ax1.set_xlabel("Method", fontsize=14)
    ax1.set_ylabel("Micro-F1", fontsize=14)
    ax1.set_title(
        f"Node Classification: {dataset.title()} (n={n_nodes}, rank={rank})",
        fontsize=16,
        fontweight="bold",
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax1.legend(
        title="Train %",
        labels=["10%", "50%", "90%"],
        loc="upper right",
        fontsize=11,
        title_fontsize=12,
    )
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.2f", fontsize=9, padding=2)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path.cwd() / "micro_f1_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {Path.cwd() / 'micro_f1_comparison.png'}")

    # Plot 2: Line plot showing performance across train ratios
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    for method in method_order:
        method_df = df[df["method"] == method].sort_values("train_ratio")
        ax2.plot(
            method_df["train_ratio"],
            method_df["micro_f1"],
            marker="o",
            markersize=10,
            linewidth=2.5,
            label=method,
            color=palette.get(method, "#333333"),
        )

    ax2.set_xlabel("Training Ratio", fontsize=14)
    ax2.set_ylabel("Micro-F1", fontsize=14)
    ax2.set_title(
        f"Performance vs Training Data: {dataset.title()}",
        fontsize=16,
        fontweight="bold",
    )
    ax2.set_xticks(train_ratios)
    ax2.set_xticklabels(["10%", "50%", "90%"], fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path.cwd() / "performance_vs_train_ratio.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {Path.cwd() / 'performance_vs_train_ratio.png'}")

    # Plot 3: Heatmap of results
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes3, ["micro_f1", "macro_f1"]):
        pivot_heat = df.pivot_table(index="method", columns="train_ratio", values=metric)
        pivot_heat = pivot_heat.reindex(method_order)
        pivot_heat.columns = ["10%", "50%", "90%"]

        sns.heatmap(
            pivot_heat,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            annot_kws={"fontsize": 11},
        )
        ax.set_title(metric.replace("_", "-").title(), fontsize=14, fontweight="bold")
        ax.set_xlabel("Training Ratio", fontsize=12)
        ax.set_ylabel("Method", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig3.suptitle(
        f"Node Classification Results: {dataset.title()} (n={n_nodes})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(Path.cwd() / "results_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {Path.cwd() / 'results_heatmap.png'}")

    print("\nResults Summary (Micro-F1):")
    pivot.columns = ["10%", "50%", "90%"]
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
