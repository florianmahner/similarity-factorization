"""Plot PPI debug results."""

from pathlib import Path

import pandas as pd
import numpy as np

from src.colors import TEAL, SAND, ROSE
from src.utils.figure_theme import create_figure, save_figure, despine
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def main():
    df = pd.read_csv(OUTPUT_DIR / "results.csv")

    # Simplify method names
    df["method"] = df["method"].replace({
        "srf_rho0.5": "SRF (ρ=0.5)",
        "srf_rho3.0": "SRF (ρ=3.0)",
        "deepwalk": "DeepWalk",
    })

    # Get number of nodes from the data
    n_nodes = 500  # From the debug script default

    # Plot 1: Micro F1 by rank for each GO bin
    go_bins = ["rare", "medium", "frequent"]
    methods = ["SRF (ρ=0.5)", "SRF (ρ=3.0)", "DeepWalk"]
    colors = {
        "SRF (ρ=0.5)": TEAL,
        "SRF (ρ=3.0)": SAND,
        "DeepWalk": ROSE,
    }

    fig, axes = create_figure("full_width", nrows=1, ncols=3, pad_top=0.3)

    for i, go_bin in enumerate(go_bins):
        ax = axes[i]
        bin_df = df[df["go_bin"] == go_bin]

        for method in methods:
            method_df = bin_df[bin_df["method"] == method].sort_values("rank")
            ax.plot(
                method_df["rank"],
                method_df["micro_f1"],
                "o-",
                color=colors[method],
                label=method,
                markersize=5,
                linewidth=1.5,
            )

        ax.set_xlabel("Embedding rank")
        ax.set_ylabel("Micro F1")
        ax.set_title(f"{go_bin.capitalize()} GO terms")
        ax.set_xticks([16, 32, 64, 128])
        if go_bin == "rare":
            ax.set_ylim(0.2, 0.4)
        elif go_bin == "medium":
            ax.set_ylim(0.45, 0.65)
        else:
            ax.set_ylim(0.85, 0.95)

        if i == 2:
            ax.legend(fontsize=7, loc="lower right")
        despine(ax)

    save_figure(fig, OUTPUT_DIR / "micro_f1_by_rank.pdf")
    print("Created micro_f1_by_rank.pdf")

    # Plot 2: Summary bar chart at rank 64
    fig, ax = create_figure("wide", pad_top=0.2)

    rank_df = df[df["rank"] == 64].copy()

    x = np.arange(len(go_bins))
    width = 0.25

    for i, method in enumerate(methods):
        method_df = rank_df[rank_df["method"] == method]
        values = [method_df[method_df["go_bin"] == b]["micro_f1"].values[0] for b in go_bins]
        ax.bar(x + i * width, values, width, label=method, color=colors[method])

    ax.set_xlabel("GO term frequency bin")
    ax.set_ylabel("Micro F1")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Rare\n(11-30)", "Medium\n(31-100)", "Frequent\n(101-300)"])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "comparison_rank64.pdf")
    print("Created comparison_rank64.pdf")

    # Plot 3: Runtime comparison
    fig, ax = create_figure("single", pad_top=0.2)

    # Average runtime per method/rank
    runtime_df = df.groupby(["method", "rank"])["runtime"].first().reset_index()

    for method in methods:
        method_df = runtime_df[runtime_df["method"] == method].sort_values("rank")
        ax.plot(
            method_df["rank"],
            method_df["runtime"],
            "o-",
            color=colors[method],
            label=method,
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Embedding rank")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks([16, 32, 64, 128])
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "runtime_comparison.pdf")
    print("Created runtime_comparison.pdf")


if __name__ == "__main__":
    main()
