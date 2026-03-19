"""Mockup visualizations for rank detection figure.

Explores multiple visualization approaches for publication in a high-impact paper.
Filters out alpha=10.0 as requested.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()


def load_data() -> pd.DataFrame:
    """Load and filter rank detection data (exclude alpha=10.0)."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"] != 10.0].copy()
    return df


# ==============================================================================
# Option 1: Accuracy heatmap (SNR x Rank)
# ==============================================================================
def plot_option1_heatmap_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap showing detection accuracy by true rank and SNR."""
    summary = df.groupby(["true_rank", "snr"])["is_correct"].mean().reset_index()
    pivot = summary.pivot(index="true_rank", columns="snr", values="is_correct")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = create_figure("single")

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(
        pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        extent=[0.1, 1.1, 0.5, pivot.shape[0] + 0.5]
    )

    ax.set_yticks(range(1, len(pivot) + 1))
    ax.set_yticklabels(pivot.index[::-1])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlabel("SNR")
    ax.set_ylabel("True rank")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy")

    despine(ax, left=False, bottom=False)
    save_figure(fig, output_dir / "option1_heatmap_accuracy.pdf")


# ==============================================================================
# Option 2: Line plot - accuracy vs rank faceted by SNR
# ==============================================================================
def plot_option2_line_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: accuracy vs rank, colored by SNR."""
    summary = df.groupby(["true_rank", "snr"]).agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    snr_colors = {0.2: GRAY_LIGHT, 0.4: GRAY, 0.6: CYAN,
                  0.8: TEAL, 1.0: ROSE}

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["accuracy"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"SNR={snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["accuracy"] - subset["sem"],
            subset["accuracy"] + subset["sem"],
            color=snr_colors[snr], alpha=0.15, linewidth=0,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Detection accuracy")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "option2_line_by_snr.pdf")


# ==============================================================================
# Option 3: Line plot - accuracy vs rank faceted by alpha
# ==============================================================================
def plot_option3_line_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: accuracy vs rank, colored by alpha."""
    summary = df.groupby(["true_rank", "alpha"]).agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$", 1.0: r"$\alpha=1$", 5.0: r"$\alpha=5$"}

    for alpha in sorted(df["alpha"].unique()):
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["accuracy"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=alpha_labels[alpha])
        ax.fill_between(
            subset["true_rank"],
            subset["accuracy"] - subset["sem"],
            subset["accuracy"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Detection accuracy")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "option3_line_by_alpha.pdf")


# ==============================================================================
# Option 4: Mean absolute error vs rank (cleaner than accuracy?)
# ==============================================================================
def plot_option4_mae_by_rank(df: pd.DataFrame, output_dir: Path) -> None:
    """Mean absolute error vs true rank, grouped by SNR."""
    summary = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    snr_colors = {0.2: GRAY_LIGHT, 0.4: GRAY, 0.6: CYAN,
                  0.8: TEAL, 1.0: ROSE}

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"SNR={snr}")

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "option4_mae_by_rank.pdf")


# ==============================================================================
# Option 5: Violin plots - show distribution of selected ranks
# ==============================================================================
def plot_option5_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plots showing distribution of selected rank for each true rank."""
    df_subset = df[df["snr"] >= 0.6].copy()

    fig, ax = create_figure("full_width")

    positions = sorted(df_subset["true_rank"].unique())
    data_for_violin = [
        df_subset[df_subset["true_rank"] == tr]["selected_rank"].values
        for tr in positions
    ]

    parts = ax.violinplot(data_for_violin, positions=positions, widths=1.5,
                          showmeans=False, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor(TEAL)
        pc.set_alpha(0.6)

    parts["cmedians"].set_color(ROSE)
    parts["cmedians"].set_linewidth(1.5)

    ax.plot(positions, positions, "--", color=GRAY, lw=1.5,
            label="Ideal", zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(-2, 50)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "option5_violin.pdf")


# ==============================================================================
# Option 6: 2D density / hexbin - true vs selected rank
# ==============================================================================
def plot_option6_hexbin(df: pd.DataFrame, output_dir: Path) -> None:
    """Hexbin density plot of true vs selected rank."""
    df_high_snr = df[df["snr"] >= 0.6].copy()

    fig, ax = create_figure("square")

    hb = ax.hexbin(
        df_high_snr["true_rank"], df_high_snr["selected_rank"],
        gridsize=15, cmap="Blues", mincnt=1,
    )

    min_r, max_r = 0, 35
    ax.plot([min_r, max_r], [min_r, max_r], "--", color=ROSE, lw=1.5, zorder=5)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(min_r, max_r)
    ax.set_ylim(min_r, max_r)
    ax.set_aspect("equal")

    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Count")

    despine(ax)
    save_figure(fig, output_dir / "option6_hexbin.pdf")


# ==============================================================================
# Option 7: Faceted small multiples - scatter by alpha and SNR
# ==============================================================================
def plot_option7_faceted(df: pd.DataFrame, output_dir: Path) -> None:
    """Small multiples: scatter plot faceted by alpha (rows) and SNR (columns)."""
    alphas = sorted(df["alpha"].unique())
    snrs = sorted(df["snr"].unique())

    fig, axes = plt.subplots(len(alphas), len(snrs),
                             figsize=(10, 6), sharex=True, sharey=True)

    rng = np.random.default_rng(42)

    for i, alpha in enumerate(alphas):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            subset = df[(df["alpha"] == alpha) & (df["snr"] == snr)].copy()

            subset["x_jit"] = subset["true_rank"] + rng.uniform(-0.3, 0.3, len(subset))
            subset["y_jit"] = subset["selected_rank"] + rng.uniform(-0.3, 0.3, len(subset))

            correct = subset[subset["is_correct"]]
            errors = subset[~subset["is_correct"]]

            ax.scatter(correct["x_jit"], correct["y_jit"], c=TEAL, alpha=0.4,
                       s=8, edgecolors="none")
            ax.scatter(errors["x_jit"], errors["y_jit"], c=ROSE, alpha=0.6,
                       s=12, edgecolors="none")

            ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=0.8, zorder=0)

            if i == 0:
                ax.set_title(f"SNR={snr}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"α={alpha}", fontsize=8)
            if i == len(alphas) - 1:
                ax.set_xlabel("True rank", fontsize=8)

            ax.set_xlim(0, 32)
            ax.set_ylim(0, 45)

            acc = subset["is_correct"].mean()
            ax.text(2, 42, f"{acc:.0%}", fontsize=7, color=GRAY_DARK)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "option7_faceted.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option 8: Compact summary - accuracy by alpha (barplot)
# ==============================================================================
def plot_option8_bar_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot: overall accuracy by alpha, with error bars."""
    summary = df.groupby("alpha").agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    fig, ax = create_figure("single")

    colors = [ROSE, TEAL, CYAN]
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["accuracy"], yerr=summary["sem"],
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 1})

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a}" for a in summary["alpha"]])
    ax.set_ylabel("Detection accuracy")
    ax.set_ylim(0, 1.0)

    despine(ax)
    save_figure(fig, output_dir / "option8_bar_by_alpha.pdf")


# ==============================================================================
# Option 9: Signed error distribution (bias analysis)
# ==============================================================================
def plot_option9_signed_error(df: pd.DataFrame, output_dir: Path) -> None:
    """Show systematic over/under-estimation by true rank."""
    summary = df.groupby("true_rank").agg(
        mean_error=("signed_error", "mean"),
        sem=("signed_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    ax.bar(summary["true_rank"], summary["mean_error"], yerr=summary["sem"],
           color=TEAL, edgecolor="white", linewidth=0.5,
           capsize=2, error_kw={"linewidth": 0.8}, width=1.5)

    ax.axhline(0, color=GRAY_DARK, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Signed error (selected - true)")

    despine(ax)
    save_figure(fig, output_dir / "option9_signed_error.pdf")


# ==============================================================================
# Option 10: Two-panel publication figure
# ==============================================================================
def plot_option10_two_panel(df: pd.DataFrame, output_dir: Path) -> None:
    """Publication-quality two-panel figure."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Panel A: Accuracy by rank, averaged over alpha, colored by SNR
    ax = axes[0]
    summary_snr = df.groupby(["true_rank", "snr"]).agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    for snr in [0.6, 0.8, 1.0]:
        subset = summary_snr[summary_snr["snr"] == snr].sort_values("true_rank")
        color = {0.6: CYAN, 0.8: TEAL, 1.0: ROSE}[snr]
        ax.plot(subset["true_rank"], subset["accuracy"], "o-",
                color=color, markersize=3, linewidth=1.2,
                label=f"SNR={snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["accuracy"] - subset["sem"],
            subset["accuracy"] + subset["sem"],
            color=color, alpha=0.15, linewidth=0,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Detection accuracy")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    despine(ax)

    # Panel B: Accuracy by rank, averaged over SNR, colored by alpha
    ax = axes[1]
    summary_alpha = df.groupby(["true_rank", "alpha"]).agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$ (sparse)", 1.0: r"$\alpha=1$ (mixed)",
                    5.0: r"$\alpha=5$ (distributed)"}

    for alpha in [0.1, 1.0, 5.0]:
        subset = summary_alpha[summary_alpha["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["accuracy"], "o-",
                color=alpha_colors[alpha], markersize=3, linewidth=1.2,
                label=alpha_labels[alpha])
        ax.fill_between(
            subset["true_rank"],
            subset["accuracy"] - subset["sem"],
            subset["accuracy"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Detection accuracy")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "option10_two_panel.pdf")


# ==============================================================================
# Option 11: Heatmap with accuracy values annotated
# ==============================================================================
def plot_option11_annotated_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Annotated heatmap: alpha x SNR, with accuracy values."""
    summary = df.groupby(["alpha", "snr"])["is_correct"].mean().reset_index()
    pivot = summary.pivot(index="alpha", columns="snr", values="is_correct")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = create_figure("single")

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, vmin=0.3, vmax=0.9,
        annot=True, fmt=".0%", annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "Accuracy", "shrink": 0.7},
        linewidths=1, linecolor="white"
    )

    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$\alpha$")

    save_figure(fig, output_dir / "option11_annotated_heatmap.pdf")


# ==============================================================================
# Option 12: Focused scatter - only best conditions (SNR >= 0.8, alpha = 1)
# ==============================================================================
def plot_option12_focused_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Focused scatter for best-case scenario."""
    df_focus = df[(df["snr"] >= 0.8) & (df["alpha"] == 1.0)].copy()

    fig, ax = create_figure("square")

    rng = np.random.default_rng(42)
    jitter = 0.4
    df_focus["x_jit"] = df_focus["true_rank"] + rng.uniform(-jitter, jitter, len(df_focus))
    df_focus["y_jit"] = df_focus["selected_rank"] + rng.uniform(-jitter, jitter, len(df_focus))

    correct = df_focus[df_focus["is_correct"]]
    errors = df_focus[~df_focus["is_correct"]]

    ax.scatter(correct["x_jit"], correct["y_jit"], c=TEAL, alpha=0.5,
               s=15, edgecolors="none", label="Correct")
    ax.scatter(errors["x_jit"], errors["y_jit"], c=ROSE, alpha=0.7,
               s=20, edgecolors="white", linewidths=0.3, label="Error")

    ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(-2, 40)
    ax.legend(frameon=False, loc="upper left", fontsize=7)

    acc = df_focus["is_correct"].mean()
    ax.text(25, 5, f"Accuracy: {acc:.0%}", fontsize=8, color=GRAY_DARK)

    despine(ax)
    save_figure(fig, output_dir / "option12_focused_scatter.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (excluding alpha=10.0)...")
    df = load_data()
    print(f"  {len(df)} observations")
    print(f"  Alphas: {sorted(df['alpha'].unique())}")
    print(f"  SNRs: {sorted(df['snr'].unique())}")
    print(f"  Overall accuracy: {df['is_correct'].mean():.1%}")

    print("\nGenerating mockups...")

    plot_option1_heatmap_accuracy(df, OUTPUT_DIR)
    print("  option1_heatmap_accuracy.pdf")

    plot_option2_line_by_snr(df, OUTPUT_DIR)
    print("  option2_line_by_snr.pdf")

    plot_option3_line_by_alpha(df, OUTPUT_DIR)
    print("  option3_line_by_alpha.pdf")

    plot_option4_mae_by_rank(df, OUTPUT_DIR)
    print("  option4_mae_by_rank.pdf")

    plot_option5_violin(df, OUTPUT_DIR)
    print("  option5_violin.pdf")

    plot_option6_hexbin(df, OUTPUT_DIR)
    print("  option6_hexbin.pdf")

    plot_option7_faceted(df, OUTPUT_DIR)
    print("  option7_faceted.pdf")

    plot_option8_bar_by_alpha(df, OUTPUT_DIR)
    print("  option8_bar_by_alpha.pdf")

    plot_option9_signed_error(df, OUTPUT_DIR)
    print("  option9_signed_error.pdf")

    plot_option10_two_panel(df, OUTPUT_DIR)
    print("  option10_two_panel.pdf")

    plot_option11_annotated_heatmap(df, OUTPUT_DIR)
    print("  option11_annotated_heatmap.pdf")

    plot_option12_focused_scatter(df, OUTPUT_DIR)
    print("  option12_focused_scatter.pdf")

    print(f"\nSaved {12} mockups to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
