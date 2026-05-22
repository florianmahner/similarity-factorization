"""Median + IQR plots for rank detection - separate alpha and SNR versions.

Matches style of other simulation plots (dirichlet_properties, srf_performance).

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_v3.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"] != 5.0].copy()  # Keep 0.1, 1.0, 10.0
    return df


def plot_median_iqr_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Median selected rank with IQR ribbon, by alpha."""
    summary = df.groupby(["true_rank", "alpha"])["selected_rank"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    summary.columns = ["true_rank", "alpha", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 10.0: CYAN}

    # Reference line with annotation (matching other simulation plots)
    ax.plot([0, 32], [0, 32], "--", color=GRAY_LIGHT, lw=1.5, zorder=0)
    ax.text(28, 26, "identity", fontsize=7, color=GRAY, ha="right")

    for alpha in [0.1, 1.0, 10.0]:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["median"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=f"{alpha}")
        ax.fill_between(
            subset["true_rank"],
            subset["q25"],
            subset["q75"],
            color=alpha_colors[alpha], alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title=r"$\alpha$",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "median_iqr_by_alpha.pdf")


def plot_median_iqr_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Median selected rank with IQR ribbon, by SNR."""
    summary = df.groupby(["true_rank", "snr"])["selected_rank"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    summary.columns = ["true_rank", "snr", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    snrs = [0.4, 0.6, 0.8, 1.0]
    snr_colors = {0.4: GRAY, 0.6: CYAN, 0.8: TEAL, 1.0: ROSE}

    # Reference line with annotation
    ax.plot([0, 32], [0, 32], "--", color=GRAY_LIGHT, lw=1.5, zorder=0)
    ax.text(28, 26, "identity", fontsize=7, color=GRAY, ha="right")

    for snr in snrs:
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["median"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"{snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["q25"],
            subset["q75"],
            color=snr_colors[snr], alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title="SNR",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "median_iqr_by_snr.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (excluding alpha=5.0)...")
    df = load_data()
    print(f"  Alphas: {sorted(df['alpha'].unique())}")

    print("\nGenerating median IQR plots...")

    plot_median_iqr_by_alpha(df, OUTPUT_DIR)
    print("  median_iqr_by_alpha.pdf")

    plot_median_iqr_by_snr(df, OUTPUT_DIR)
    print("  median_iqr_by_snr.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
