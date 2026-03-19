"""Mockups showing distribution better for alpha = 0.1, 1.0, 5.0.

Exploring different ways to visualize the error distribution clearly.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_v4.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_DARK, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

ALPHAS = [0.1, 1.0, 5.0]
ALPHA_COLORS = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"].isin(ALPHAS)].copy()
    return df


# ==============================================================================
# Option A: Three-panel boxplots (one per alpha)
# ==============================================================================
def plot_boxplot_panels(df: pd.DataFrame, output_dir: Path) -> None:
    """Three panels, one boxplot per alpha value."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha]
        ranks = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        bp = ax.boxplot(data, positions=ranks, widths=1.5, patch_artist=True,
                        showfliers=False, whis=[10, 90])

        for patch in bp["boxes"]:
            patch.set_facecolor(ALPHA_COLORS[alpha])
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=9)
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)
        despine(ax)

    axes[0].set_ylabel("Selected rank")
    plt.tight_layout()
    fig.savefig(output_dir / "v4_boxplot_panels.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option B: Overlapping boxplots at each rank (grouped)
# ==============================================================================
def plot_boxplot_grouped(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped boxplots - 3 boxes per true rank, one per alpha."""
    fig, ax = create_figure("full_width")

    ranks = sorted(df["true_rank"].unique())
    width = 0.6
    offsets = [-width, 0, width]

    for i, alpha in enumerate(ALPHAS):
        subset = df[df["alpha"] == alpha]
        positions = [r + offsets[i] for r in ranks]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        bp = ax.boxplot(data, positions=positions, widths=width * 0.8,
                        patch_artist=True, showfliers=False, whis=[10, 90])

        for patch in bp["boxes"]:
            patch.set_facecolor(ALPHA_COLORS[alpha])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(1)

    ax.plot([0, 32], [0, 32], "--", color=GRAY_LIGHT, lw=1.5, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(-1, 32)
    ax.set_ylim(-2, 45)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ALPHA_COLORS[a], alpha=0.7) for a in ALPHAS]
    ax.legend(handles, [f"{a}" for a in ALPHAS], title=r"$\alpha$",
              frameon=False, loc="upper left", fontsize=7)

    despine(ax)
    save_figure(fig, output_dir / "v4_boxplot_grouped.pdf")


# ==============================================================================
# Option C: Violin plots (3 panels)
# ==============================================================================
def plot_violin_panels(df: pd.DataFrame, output_dir: Path) -> None:
    """Three panels with violin plots."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha]
        ranks = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        parts = ax.violinplot(data, positions=ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(ALPHA_COLORS[alpha])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=9)
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)
        despine(ax)

    axes[0].set_ylabel("Selected rank")
    plt.tight_layout()
    fig.savefig(output_dir / "v4_violin_panels.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option D: Strip plot with jitter (shows all points)
# ==============================================================================
def plot_strip_panels(df: pd.DataFrame, output_dir: Path) -> None:
    """Strip plots showing individual data points."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)
    rng = np.random.default_rng(42)

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha].copy()
        jitter = rng.uniform(-0.4, 0.4, len(subset))
        subset["x_jit"] = subset["true_rank"] + jitter

        ax.scatter(subset["x_jit"], subset["selected_rank"],
                   c=ALPHA_COLORS[alpha], alpha=0.3, s=8, edgecolors="none")

        # Add median line
        medians = subset.groupby("true_rank")["selected_rank"].median()
        ax.plot(medians.index, medians.values, "-", color="white", lw=2, zorder=5)
        ax.plot(medians.index, medians.values, "-", color=ALPHA_COLORS[alpha], lw=1.5, zorder=6)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=9)
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)
        despine(ax)

    axes[0].set_ylabel("Selected rank")
    plt.tight_layout()
    fig.savefig(output_dir / "v4_strip_panels.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option E: Median + percentile bands (10-90 and 25-75)
# ==============================================================================
def plot_percentile_bands(df: pd.DataFrame, output_dir: Path) -> None:
    """Single plot with median line and two percentile bands per alpha."""
    fig, ax = create_figure("wide")

    ax.plot([0, 32], [0, 32], "--", color=GRAY_LIGHT, lw=1.5, zorder=0)
    ax.text(28, 26, "identity", fontsize=7, color=GRAY, ha="right")

    for alpha in ALPHAS:
        subset = df[df["alpha"] == alpha]
        summary = subset.groupby("true_rank")["selected_rank"].agg(
            ["median",
             lambda x: x.quantile(0.10),
             lambda x: x.quantile(0.25),
             lambda x: x.quantile(0.75),
             lambda x: x.quantile(0.90)]
        ).reset_index()
        summary.columns = ["true_rank", "median", "p10", "p25", "p75", "p90"]

        color = ALPHA_COLORS[alpha]

        # 10-90 percentile band (lighter)
        ax.fill_between(summary["true_rank"], summary["p10"], summary["p90"],
                        color=color, alpha=0.1, linewidth=0)

        # 25-75 percentile band (darker)
        ax.fill_between(summary["true_rank"], summary["p25"], summary["p75"],
                        color=color, alpha=0.25, linewidth=0)

        # Median line
        ax.plot(summary["true_rank"], summary["median"], "o-",
                color=color, markersize=4, linewidth=1.5, label=f"{alpha}")

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title=r"$\alpha$",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "v4_percentile_bands.pdf")


# ==============================================================================
# Option F: Error distribution histogram (signed error)
# ==============================================================================
def plot_error_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of signed errors, faceted by alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2), sharey=True)

    bins = np.arange(-15, 20, 1)

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha]
        ax.hist(subset["signed_error"], bins=bins, color=ALPHA_COLORS[alpha],
                alpha=0.7, edgecolor="white", linewidth=0.5)

        ax.axvline(0, color=GRAY_DARK, linestyle="-", linewidth=1, zorder=5)

        mae = subset["abs_error"].mean()
        median_err = subset["signed_error"].median()
        ax.text(0.95, 0.95, f"MAE={mae:.1f}\nmedian={median_err:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=GRAY_DARK)

        ax.set_xlabel("Error (selected − true)")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=9)
        ax.set_xlim(-15, 20)
        despine(ax)

    axes[0].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_dir / "v4_error_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option G: Heatmap confusion matrix style (per alpha)
# ==============================================================================
def plot_heatmap_panels(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmaps showing P(selected | true) for each alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    true_ranks = sorted(df["true_rank"].unique())

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha].copy()

        # Bin selected ranks
        subset["sel_bin"] = np.clip(
            np.round(subset["selected_rank"] / 2) * 2, 0, 40
        ).astype(int)

        confusion = pd.crosstab(subset["true_rank"], subset["sel_bin"], normalize="index")

        # Reindex to ensure consistent shape
        sel_bins = list(range(0, 42, 2))
        confusion = confusion.reindex(index=true_ranks, columns=sel_bins, fill_value=0)

        im = ax.imshow(confusion.values, aspect="auto", cmap="Blues",
                       vmin=0, vmax=0.6, origin="lower",
                       extent=[-1, 41, true_ranks[0]-1, true_ranks[-1]+1])

        ax.plot([-1, 41], [true_ranks[0]-1, true_ranks[-1]+1], "--",
                color=ROSE, lw=1.5, zorder=5)

        ax.set_xlabel("Selected rank")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=9)
        ax.set_xlim(-1, 41)
        ax.set_ylim(true_ranks[0]-1, true_ranks[-1]+1)

        if alpha == ALPHAS[0]:
            ax.set_ylabel("True rank")

    plt.tight_layout()
    fig.savefig(output_dir / "v4_heatmap_panels.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data for alpha = 0.1, 1.0, 5.0...")
    df = load_data()
    print(f"  {len(df)} observations")

    print("\nGenerating v4 mockups...")

    plot_boxplot_panels(df, OUTPUT_DIR)
    print("  v4_boxplot_panels.pdf - Three-panel boxplots")

    plot_boxplot_grouped(df, OUTPUT_DIR)
    print("  v4_boxplot_grouped.pdf - Grouped boxplots")

    plot_violin_panels(df, OUTPUT_DIR)
    print("  v4_violin_panels.pdf - Three-panel violins")

    plot_strip_panels(df, OUTPUT_DIR)
    print("  v4_strip_panels.pdf - Strip plots with all points")

    plot_percentile_bands(df, OUTPUT_DIR)
    print("  v4_percentile_bands.pdf - Median + 10-90/25-75 bands")

    plot_error_histogram(df, OUTPUT_DIR)
    print("  v4_error_histogram.pdf - Error distribution")

    plot_heatmap_panels(df, OUTPUT_DIR)
    print("  v4_heatmap_panels.pdf - Confusion heatmaps")

    print(f"\nSaved 7 mockups to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
