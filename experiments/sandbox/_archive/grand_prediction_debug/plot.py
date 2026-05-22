"""Plot grand prediction results."""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.figure_theme import (
    apply_theme,
    save_figure,
    despine,
    CMAP,
    GRAY,
)


def plot_overview(df: pd.DataFrame, output_path: Path) -> None:
    """Single overview figure with all correlations grouped by domain."""
    apply_theme()

    # Sort within each domain by correlation
    domains = ["animals", "clothing", "professions", "sports"]
    groups = []
    for domain in domains:
        domain_df = df[df["domain"] == domain].copy()
        domain_df = domain_df.sort_values("correlation", ascending=False)
        groups.append(domain_df)

    df_sorted = pd.concat(groups)

    # Calculate positions with gaps between domains
    positions = []
    domain_centers = {}
    y = 0.0
    gap = 1.0

    for domain in domains:
        domain_df = df_sorted[df_sorted["domain"] == domain]
        n = len(domain_df)
        domain_positions = [y + i for i in range(n)]
        positions.extend(domain_positions)
        domain_centers[domain] = y + (n - 1) / 2
        y += n + gap

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 7))

    # Colors: blue for significant positive, red for significant negative, gray otherwise
    colors = []
    for _, row in df_sorted.iterrows():
        if row["pvalue"] < 0.05:
            colors.append(CMAP[1] if row["correlation"] > 0 else CMAP[0])
        else:
            colors.append(GRAY["light"])

    # Plot bars
    bars = ax.barh(positions, df_sorted["correlation"], color=colors, height=0.7)

    # Add significance markers
    for pos, (_, row) in zip(positions, df_sorted.iterrows()):
        if row["pvalue"] < 0.05:
            x = row["correlation"]
            marker_x = x + 0.03 if x > 0 else x - 0.03
            ha = "left" if x > 0 else "right"
            ax.text(marker_x, pos, "*", fontsize=12, ha=ha, va="center", fontweight="bold")

    # Zero line
    ax.axvline(0, color=GRAY["dark"], linewidth=0.8)

    # Y-axis labels (dimensions)
    ax.set_yticks(positions)
    ax.set_yticklabels(df_sorted["dimension"], fontsize=9)

    # Domain labels on the right
    for domain, center in domain_centers.items():
        ax.text(
            0.72, center, domain.capitalize(),
            transform=ax.get_yaxis_transform(),
            fontsize=10, fontweight="bold",
            ha="left", va="center",
            color=GRAY["dark"],
        )

    # Add subtle horizontal lines between domains
    y = 0
    for i, domain in enumerate(domains[:-1]):
        n = len(df_sorted[df_sorted["domain"] == domain])
        y += n
        ax.axhline(y - 0.5 + gap/2, color=GRAY["faint"], linewidth=0.5, linestyle="-")
        y += gap

    ax.set_xlabel("Correlation", fontsize=10)
    ax.set_xlim(-0.6, 0.7)
    ax.invert_yaxis()

    despine(ax)

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    results_path = Path("sandbox/grand_prediction_debug/outputs/prediction_results.csv")
    output_dir = Path("sandbox/grand_prediction_debug/outputs")

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} results")

    plot_overview(df, output_dir / "prediction_overview.pdf")
    print(f"Created: prediction_overview.pdf")


if __name__ == "__main__":
    main()
