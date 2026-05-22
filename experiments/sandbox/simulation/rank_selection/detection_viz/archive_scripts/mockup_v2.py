"""Improved rank detection visualizations focusing on error distribution.

Key insight: Binary accuracy is too harsh - being off by 1 is still good.
Better metrics: MAE, error distribution, tolerance-based accuracy.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_v2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"] != 10.0].copy()
    return df


# ==============================================================================
# Option A: Boxplots by true rank, faceted by alpha (3 panels)
# ==============================================================================
def plot_boxplot_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots showing selected rank distribution for each true rank, by alpha."""
    alphas = sorted(df["alpha"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)

    alpha_labels = {0.1: r"$\alpha=0.1$ (sparse)",
                    1.0: r"$\alpha=1$ (mixed)",
                    5.0: r"$\alpha=5$ (distributed)"}

    for ax, alpha in zip(axes, alphas):
        subset = df[df["alpha"] == alpha]
        ranks = sorted(subset["true_rank"].unique())

        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        bp = ax.boxplot(data, positions=ranks, widths=1.5, patch_artist=True,
                        showfliers=False, whis=[5, 95])

        for patch in bp["boxes"]:
            patch.set_facecolor(TEAL)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color(ROSE)
            median.set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(alpha_labels[alpha], fontsize=9)
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)

        mae = subset["abs_error"].mean()
        ax.text(2, 40, f"MAE={mae:.1f}", fontsize=8, color=GRAY_DARK)

        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "boxplot_by_alpha.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option B: Boxplots by true rank, faceted by SNR (select 3 levels)
# ==============================================================================
def plot_boxplot_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots showing selected rank distribution for each true rank, by SNR."""
    snrs = [0.4, 0.6, 1.0]  # Low, medium, high

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)

    snr_labels = {0.4: "SNR=0.4 (noisy)",
                  0.6: "SNR=0.6 (moderate)",
                  1.0: "SNR=1.0 (clean)"}

    for ax, snr in zip(axes, snrs):
        subset = df[df["snr"] == snr]
        ranks = sorted(subset["true_rank"].unique())

        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        bp = ax.boxplot(data, positions=ranks, widths=1.5, patch_artist=True,
                        showfliers=False, whis=[5, 95])

        for patch in bp["boxes"]:
            patch.set_facecolor(TEAL)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color(ROSE)
            median.set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(snr_labels[snr], fontsize=9)
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)

        mae = subset["abs_error"].mean()
        ax.text(2, 40, f"MAE={mae:.1f}", fontsize=8, color=GRAY_DARK)

        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "boxplot_by_snr.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Option C: MAE vs true rank, colored by alpha
# ==============================================================================
def plot_mae_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank, with lines for each alpha."""
    summary = df.groupby(["true_rank", "alpha"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$", 1.0: r"$\alpha=1$", 5.0: r"$\alpha=5$"}

    for alpha in [0.1, 1.0, 5.0]:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=alpha_labels[alpha])
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "mae_by_alpha.pdf")


# ==============================================================================
# Option D: MAE vs true rank, colored by SNR
# ==============================================================================
def plot_mae_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank, with lines for each SNR (subset)."""
    summary = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("wide")

    snrs = [0.4, 0.6, 0.8, 1.0]
    snr_colors = {0.4: GRAY, 0.6: CYAN, 0.8: TEAL, 1.0: ROSE}

    for snr in snrs:
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"SNR={snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=snr_colors[snr], alpha=0.15, linewidth=0,
        )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "mae_by_snr.pdf")


# ==============================================================================
# Option E: Tolerance accuracy (% within ±1, ±2, ±3)
# ==============================================================================
def plot_tolerance_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Show % of predictions within ±1, ±2, ±3 of true rank."""
    tolerances = [0, 1, 2, 3, 5]

    results = []
    for alpha in df["alpha"].unique():
        subset = df[df["alpha"] == alpha]
        for tol in tolerances:
            within_tol = (subset["abs_error"] <= tol).mean()
            results.append({"alpha": alpha, "tolerance": tol, "accuracy": within_tol})

    result_df = pd.DataFrame(results)

    fig, ax = create_figure("single")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$", 1.0: r"$\alpha=1$", 5.0: r"$\alpha=5$"}

    for alpha in [0.1, 1.0, 5.0]:
        subset = result_df[result_df["alpha"] == alpha].sort_values("tolerance")
        ax.plot(subset["tolerance"], subset["accuracy"], "o-",
                color=alpha_colors[alpha], markersize=6, linewidth=2,
                label=alpha_labels[alpha])

    ax.set_xlabel("Tolerance (±ranks)")
    ax.set_ylabel("Proportion within tolerance")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(tolerances)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "tolerance_accuracy.pdf")


# ==============================================================================
# Option F: Signed error distribution by alpha (shows bias)
# ==============================================================================
def plot_signed_error_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot of signed error distribution by alpha."""
    fig, ax = create_figure("wide")

    alphas = sorted(df["alpha"].unique())
    alpha_labels = [r"$\alpha=0.1$", r"$\alpha=1$", r"$\alpha=5$"]

    data = [df[df["alpha"] == a]["signed_error"].values for a in alphas]

    parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.7,
                          showmeans=True, showmedians=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CYCLE[i % len(CYCLE)])
        pc.set_alpha(0.6)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1, zorder=0)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("Signed error (selected − true)")
    ax.set_ylim(-15, 25)

    despine(ax)
    save_figure(fig, output_dir / "signed_error_distribution.pdf")


# ==============================================================================
# Option G: Two-panel publication figure (MAE version)
# ==============================================================================
def plot_two_panel_mae(df: pd.DataFrame, output_dir: Path) -> None:
    """Two-panel figure: MAE by alpha (left) and MAE by SNR (right)."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Panel A: MAE by alpha
    ax = axes[0]
    summary_alpha = df.groupby(["true_rank", "alpha"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$", 1.0: r"$\alpha=1$", 5.0: r"$\alpha=5$"}

    for alpha in [0.1, 1.0, 5.0]:
        subset = summary_alpha[summary_alpha["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=alpha_colors[alpha], markersize=3, linewidth=1.2,
                label=alpha_labels[alpha])
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 8)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    # Panel B: MAE by SNR
    ax = axes[1]
    summary_snr = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    snrs = [0.4, 0.6, 0.8, 1.0]
    snr_colors = {0.4: GRAY, 0.6: CYAN, 0.8: TEAL, 1.0: ROSE}

    for snr in snrs:
        subset = summary_snr[summary_snr["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=3, linewidth=1.2,
                label=f"SNR={snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=snr_colors[snr], alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 8)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "two_panel_mae.pdf")


# ==============================================================================
# Option H: Confusion-style heatmap (P(selected | true))
# ==============================================================================
def plot_confusion_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap showing P(selected_rank | true_rank) - best alpha only."""
    df_best = df[df["alpha"] == 1.0].copy()

    true_ranks = sorted(df_best["true_rank"].unique())

    # Bin selected ranks to match true rank grid
    df_best["selected_binned"] = pd.cut(
        df_best["selected_rank"],
        bins=[-np.inf] + list(np.array(true_ranks) + 1) + [np.inf],
        labels=true_ranks + [true_ranks[-1] + 2]
    ).astype(int)
    df_best["selected_binned"] = df_best["selected_binned"].clip(upper=32)

    # Compute conditional probabilities
    confusion = pd.crosstab(
        df_best["true_rank"],
        df_best["selected_binned"],
        normalize="index"
    )

    fig, ax = create_figure("square")

    im = ax.imshow(confusion.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=0.5, origin="lower")

    # Diagonal line
    ax.plot([-0.5, len(true_ranks)-0.5], [-0.5, len(true_ranks)-0.5],
            "--", color=ROSE, lw=1.5, zorder=5)

    ax.set_xticks(range(0, len(true_ranks), 2))
    ax.set_xticklabels(true_ranks[::2])
    ax.set_yticks(range(0, len(true_ranks), 2))
    ax.set_yticklabels(true_ranks[::2])

    ax.set_xlabel("Selected rank")
    ax.set_ylabel("True rank")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(selected | true)")

    despine(ax, left=False, bottom=False)
    save_figure(fig, output_dir / "confusion_heatmap.pdf")


# ==============================================================================
# Option I: Median + IQR ribbon plot
# ==============================================================================
def plot_median_iqr(df: pd.DataFrame, output_dir: Path) -> None:
    """Median selected rank with IQR ribbon, by alpha."""
    summary = df.groupby(["true_rank", "alpha"])["selected_rank"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    summary.columns = ["true_rank", "alpha", "median", "q25", "q75"]

    fig, ax = create_figure("wide")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
    alpha_labels = {0.1: r"$\alpha=0.1$", 1.0: r"$\alpha=1$", 5.0: r"$\alpha=5$"}

    # Identity line
    ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0,
            label="Ideal")

    for alpha in [0.1, 1.0, 5.0]:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["median"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=alpha_labels[alpha])
        ax.fill_between(
            subset["true_rank"],
            subset["q25"],
            subset["q75"],
            color=alpha_colors[alpha], alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank (median ± IQR)")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 40)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "median_iqr.pdf")


# ==============================================================================
# Option J: Combined boxplot + line overlay
# ==============================================================================
def plot_boxplot_with_ideal(df: pd.DataFrame, output_dir: Path) -> None:
    """Single boxplot for alpha=1, with ideal line and annotations."""
    df_best = df[df["alpha"] == 1.0].copy()

    fig, ax = create_figure("wide")

    ranks = sorted(df_best["true_rank"].unique())
    data = [df_best[df_best["true_rank"] == r]["selected_rank"].values for r in ranks]

    bp = ax.boxplot(data, positions=ranks, widths=1.5, patch_artist=True,
                    showfliers=False, whis=[10, 90])

    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL)
        patch.set_alpha(0.5)
        patch.set_edgecolor(TEAL)
    for median in bp["medians"]:
        median.set_color(ROSE)
        median.set_linewidth(2)
    for whisker in bp["whiskers"]:
        whisker.set_color(GRAY)
    for cap in bp["caps"]:
        cap.set_color(GRAY)

    ax.plot([0, 32], [0, 32], "--", color=GRAY_DARK, lw=2, zorder=0,
            label="Ideal")

    mae = df_best["abs_error"].mean()
    within_2 = (df_best["abs_error"] <= 2).mean()

    ax.text(0.98, 0.05, f"MAE = {mae:.1f}\n±2 accuracy = {within_2:.0%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GRAY_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRAY_LIGHT, alpha=0.8))

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(-2, 42)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "boxplot_best_alpha.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (excluding alpha=10.0)...")
    df = load_data()

    print(f"  {len(df)} observations")
    print(f"  Overall MAE: {df['abs_error'].mean():.2f}")
    print(f"  Within ±1: {(df['abs_error'] <= 1).mean():.1%}")
    print(f"  Within ±2: {(df['abs_error'] <= 2).mean():.1%}")
    print(f"  Within ±3: {(df['abs_error'] <= 3).mean():.1%}")

    print("\nGenerating v2 mockups...")

    plot_boxplot_by_alpha(df, OUTPUT_DIR)
    print("  boxplot_by_alpha.pdf")

    plot_boxplot_by_snr(df, OUTPUT_DIR)
    print("  boxplot_by_snr.pdf")

    plot_mae_by_alpha(df, OUTPUT_DIR)
    print("  mae_by_alpha.pdf")

    plot_mae_by_snr(df, OUTPUT_DIR)
    print("  mae_by_snr.pdf")

    plot_tolerance_accuracy(df, OUTPUT_DIR)
    print("  tolerance_accuracy.pdf")

    plot_signed_error_distribution(df, OUTPUT_DIR)
    print("  signed_error_distribution.pdf")

    plot_two_panel_mae(df, OUTPUT_DIR)
    print("  two_panel_mae.pdf")

    plot_confusion_heatmap(df, OUTPUT_DIR)
    print("  confusion_heatmap.pdf")

    plot_median_iqr(df, OUTPUT_DIR)
    print("  median_iqr.pdf")

    plot_boxplot_with_ideal(df, OUTPUT_DIR)
    print("  boxplot_best_alpha.pdf")

    print(f"\nSaved 10 v2 mockups to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
