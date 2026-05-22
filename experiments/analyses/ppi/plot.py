"""
Create all PPI experiment plots.

Usage:
    poetry run python experiments/ppi/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import (
    create_figure,
    despine,
    save_figure,
)

PROJECT_ROOT = Path(__file__).parents[2]
PPI_DIR = PROJECT_ROOT / "outputs/experiments/ppi"
DATA_DIR = PPI_DIR / "data"
PLOT_DIR = PPI_DIR / "plots"

METHOD_COLORS = {
    "srf": ROSE,
    "deepwalk": TEAL,
    "node2vec": CYAN,
    "line": SAND,
    "spectral": GRAY,
    "skipgnn": PURPLE,
    "aa": GRAY_DARK,
    "cn": GRAY,
    "ra": GRAY_LIGHT,
    "jc": GRAY_PALE,
}

METHOD_ORDER = ["srf", "deepwalk", "node2vec", "line", "spectral", "skipgnn"]
LINK_METHOD_ORDER = ["srf", "skipgnn", "aa", "cn", "ra", "jc"]
METHOD_LABELS = {
    "srf": "SRF",
    "deepwalk": "DeepWalk",
    "node2vec": "Node2Vec",
    "line": "LINE",
    "spectral": "Spectral",
    "skipgnn": "SkipGNN",
    "aa": "AA",
    "cn": "CN",
    "ra": "RA",
    "jc": "JC",
}

DATASET_LABELS = {
    "c_elegans": "C. elegans",
    "huri": "HuRI",
    "string": "STRING",
}

BIN_LABELS = {
    "11-30": "Rare",
    "31-100": "Medium",
    "101-300": "Frequent",
}


def plot_node_classification_by_bin() -> None:
    """Bar chart: Micro-F1 by method and GO bin."""
    df = pd.read_csv(DATA_DIR / "node_classification_benchmark.csv")

    avg = df.groupby(["Method", "Bin"])["Micro-F1"].mean().reset_index()
    pivot = avg.pivot(index="Method", columns="Bin", values="Micro-F1")

    methods = [m for m in METHOD_ORDER if m in pivot.index]
    bins = ["11-30", "31-100", "101-300"]

    fig, ax = create_figure("wide", pad_bottom=0.6)

    x = np.arange(len(bins))
    width = 0.15
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    for i, method in enumerate(methods):
        values = [pivot.loc[method, b] for b in bins]
        color = METHOD_COLORS.get(method, GRAY)
        ax.bar(
            x + offsets[i], values, width * 0.9,
            label=METHOD_LABELS.get(method, method),
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[b] for b in bins])
    ax.set_xlabel("GO term frequency")
    ax.set_ylabel("Micro F1")
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    despine(ax)

    save_figure(fig, PLOT_DIR / "node_classification_by_bin.pdf")


def plot_node_classification_by_size() -> None:
    """Line plot: Micro-F1 by network size."""
    df = pd.read_csv(DATA_DIR / "node_classification_benchmark.csv")

    avg = df.groupby(["Method", "Nodes"])["Micro-F1"].mean().reset_index()

    fig, ax = create_figure("single")

    methods = [m for m in METHOD_ORDER if m in avg["Method"].unique()]
    for method in methods:
        subset = avg[avg["Method"] == method].sort_values("Nodes")
        color = METHOD_COLORS.get(method, GRAY)
        ax.plot(
            subset["Nodes"], subset["Micro-F1"],
            marker="o", label=METHOD_LABELS.get(method, method),
            color=color, linewidth=1.5, markersize=5
        )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Micro F1")
    ax.legend(fontsize=7)
    despine(ax)

    save_figure(fig, PLOT_DIR / "node_classification_by_size.pdf")


def plot_corum_f1_distribution() -> None:
    """Histogram of F1 scores across dimensions."""
    df = pd.read_csv(DATA_DIR / "corum_validation.csv")

    fig, ax = create_figure("single")

    ax.hist(df["f1"], bins=20, color=ROSE, edgecolor="white", linewidth=0.5)
    ax.axvline(df["f1"].median(), color=GRAY_DARK, linestyle="--", linewidth=1)

    ax.set_xlabel("F1 score")
    ax.set_ylabel("Count")
    despine(ax)

    median_f1 = df["f1"].median()
    ax.text(
        0.95, 0.95, f"Median: {median_f1:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8
    )

    save_figure(fig, PLOT_DIR / "corum_f1_distribution.pdf")


def plot_corum_top_dimensions() -> None:
    """Bar chart of top dimensions by F1."""
    df = pd.read_csv(DATA_DIR / "corum_validation.csv")
    top = df.nlargest(10, "f1")

    fig, ax = create_figure("wide", pad_left=1.8)

    y = np.arange(len(top))
    ax.barh(y, top["f1"].values, color=ROSE, height=0.7)

    labels = [f"Dim {d}: {c[:25]}..." if len(c) > 25 else f"Dim {d}: {c}"
              for d, c in zip(top["dimension"], top["best_complex"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("F1 score")
    ax.invert_yaxis()
    despine(ax)

    save_figure(fig, PLOT_DIR / "corum_top_dimensions.pdf")


def load_link_prediction_results() -> pd.DataFrame:
    """Load link prediction results from JSON files and save consolidated CSV."""
    import json

    link_dir = DATA_DIR / "link_prediction"
    records = []
    for f in link_dir.glob("**/*.json"):
        with open(f) as fp:
            records.append(json.load(fp))

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "link_prediction.csv", index=False)
    return df


def plot_link_prediction_by_dataset() -> None:
    """Grouped bar chart: AUROC by method and dataset."""
    df = load_link_prediction_results()

    avg = df.groupby(["dataset", "method"])["auroc"].agg(["mean", "std"]).reset_index()

    datasets = ["c_elegans", "huri", "string"]
    methods = [m for m in LINK_METHOD_ORDER if m in avg["method"].unique()]

    fig, ax = create_figure("wide", pad_bottom=0.7)

    x = np.arange(len(datasets))
    width = 0.12
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    for i, method in enumerate(methods):
        subset = avg[avg["method"] == method]
        values = []
        errors = []
        for ds in datasets:
            row = subset[subset["dataset"] == ds]
            if len(row) > 0:
                values.append(row["mean"].values[0])
                errors.append(row["std"].values[0])
            else:
                values.append(0)
                errors.append(0)
        color = METHOD_COLORS.get(method, GRAY)
        ax.bar(
            x + offsets[i], values, width * 0.9,
            yerr=errors, capsize=2,
            label=METHOD_LABELS.get(method, method),
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    despine(ax)

    save_figure(fig, PLOT_DIR / "link_prediction_by_dataset.pdf")


def plot_link_prediction_summary() -> None:
    """Summary bar chart: average AUROC per dataset (only methods in LINK_METHOD_ORDER)."""
    df = load_link_prediction_results()
    df = df[df["method"].isin(LINK_METHOD_ORDER)]

    stats = df.groupby(["dataset", "method"])["auroc"].agg(["mean", "std"]).reset_index()

    datasets = ["c_elegans", "huri"]
    methods = [m for m in LINK_METHOD_ORDER if m in stats["method"].unique()]

    fig, ax = create_figure("wide", pad_bottom=0.6)

    x = np.arange(len(datasets))
    width = 0.12
    n_methods = len(methods)
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    for i, method in enumerate(methods):
        subset = stats[stats["method"] == method]
        values = []
        errors = []
        for ds in datasets:
            row = subset[subset["dataset"] == ds]
            if len(row) > 0:
                values.append(row["mean"].values[0])
                errors.append(row["std"].values[0])
            else:
                values.append(0)
                errors.append(0)
        color = METHOD_COLORS.get(method, GRAY)
        ax.bar(
            x + offsets[i], values, width * 0.9,
            yerr=errors, capsize=2,
            label=METHOD_LABELS.get(method, method),
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    despine(ax)

    save_figure(fig, PLOT_DIR / "link_prediction_summary.pdf")


def plot_method_summary() -> None:
    """Summary bar chart: average Micro-F1 across all conditions."""
    df = pd.read_csv(DATA_DIR / "node_classification_benchmark.csv")

    avg = df.groupby("Method")["Micro-F1"].mean().sort_values(ascending=True)

    fig, ax = create_figure("single", pad_left=0.8)

    methods = avg.index.tolist()
    colors = [METHOD_COLORS.get(m, GRAY) for m in methods]

    y = np.arange(len(methods))
    ax.barh(y, avg.values, color=colors, height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_xlabel("Average Micro F1")
    despine(ax)

    for i, v in enumerate(avg.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    save_figure(fig, PLOT_DIR / "method_summary.pdf")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting node classification by GO bin...")
    plot_node_classification_by_bin()

    print("Plotting node classification by network size...")
    plot_node_classification_by_size()

    print("Plotting CORUM F1 distribution...")
    plot_corum_f1_distribution()

    print("Plotting CORUM top dimensions...")
    plot_corum_top_dimensions()

    print("Plotting link prediction by dataset...")
    plot_link_prediction_by_dataset()

    print("Plotting link prediction summary...")
    plot_link_prediction_summary()

    print("Plotting method summary...")
    plot_method_summary()

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
