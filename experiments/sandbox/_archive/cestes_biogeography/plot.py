"""
Plotting for Dune Meadow Biogeography Analysis.

Shows environmental validation: factors correlate with moisture gradient.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.figure_theme import (
    CMAP, GRAY, create_figure, save_figure, despine, apply_theme
)


def plot_moisture_validation(site_scores: pd.DataFrame, env_df: pd.DataFrame, output_path: Path) -> None:
    """Plot site factor scores vs moisture gradient - key validation."""
    apply_theme()

    # Find factors with strongest moisture correlation
    moisture = env_df["Moisture"].values.astype(float)

    # Calculate correlations
    corrs = []
    for k in range(site_scores.shape[1]):
        from scipy.stats import spearmanr
        r, p = spearmanr(site_scores.iloc[:, k], moisture)
        corrs.append((k, r, p))

    # Sort by absolute correlation
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Plot top 4 factors
    fig, axes = plt.subplots(2, 2, figsize=(5, 4.5))
    axes = axes.flatten()

    for i, (k, r, p) in enumerate(corrs[:4]):
        ax = axes[i]
        scores = site_scores.iloc[:, k].values

        # Color by management
        mgmt = env_df["Management"].values
        mgmt_colors = {"SF": CMAP[0], "BF": CMAP[1], "HF": CMAP[2], "NM": CMAP[3]}
        colors = [mgmt_colors.get(m, GRAY["medium"]) for m in mgmt]

        ax.scatter(moisture, scores, c=colors, s=50, alpha=0.8)

        # Add trend line
        z = np.polyfit(moisture, scores, 1)
        x_line = np.linspace(moisture.min(), moisture.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color=GRAY["dark"], linestyle="--", linewidth=1)

        sig = "*" if p < 0.05 else ""
        ax.set_title(f"Factor {k}: r={r:.2f}{sig}", fontsize=9)
        ax.set_xlabel("Moisture" if i >= 2 else "")
        ax.set_ylabel("Factor score" if i % 2 == 0 else "")
        despine(ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6, label=m)
                       for m, c in mgmt_colors.items()]
    axes[1].legend(handles=legend_elements, fontsize=6, loc="upper right", title="Mgmt", title_fontsize=6)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_factor_moisture_bars(validation_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of factor-moisture correlations."""
    apply_theme()

    moisture = validation_df[validation_df["variable"] == "Moisture"].copy()
    moisture = moisture.sort_values("factor")

    fig, ax = create_figure("single")

    colors = [CMAP[1] if r > 0 else CMAP[0] for r in moisture["correlation"]]
    bars = ax.bar(moisture["factor"], moisture["correlation"], color=colors, alpha=0.8)

    # Mark significant
    for i, (_, row) in enumerate(moisture.iterrows()):
        if row["p_value"] < 0.05:
            ax.text(row["factor"], row["correlation"] + 0.05 * np.sign(row["correlation"]),
                    "*", ha="center", fontsize=12)

    ax.axhline(0, color=GRAY["medium"], linewidth=0.8)
    ax.axhline(0.3, color=GRAY["light"], linewidth=0.6, linestyle="--")
    ax.axhline(-0.3, color=GRAY["light"], linewidth=0.6, linestyle="--")

    ax.set_xlabel("Factor")
    ax.set_ylabel("Correlation with moisture")
    ax.set_xticks(moisture["factor"])
    ax.set_xticklabels([f"F{i}" for i in moisture["factor"]])
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("Factor-Moisture correlations")

    despine(ax)
    save_figure(fig, output_path)


def plot_species_by_factor(embedding_df: pd.DataFrame, validation_df: pd.DataFrame, output_path: Path) -> None:
    """Show top species per factor, colored by moisture preference."""
    apply_theme()

    moisture = validation_df[validation_df["variable"] == "Moisture"]
    n_factors = len(moisture)
    n_top = 5

    fig, axes = plt.subplots(2, 3, figsize=(7, 4))
    axes = axes.flatten()

    for k in range(min(n_factors, 6)):
        ax = axes[k]
        weights = embedding_df.iloc[:, k]
        top_idx = np.argsort(weights.values)[-n_top:][::-1]
        top_species = embedding_df.index[top_idx]
        top_weights = weights.iloc[top_idx]

        # Get moisture direction for this factor
        moist_r = moisture[moisture["factor"] == k]["correlation"].values[0]
        color = CMAP[1] if moist_r > 0 else CMAP[0]  # Blue=wet, Red=dry

        y_pos = np.arange(n_top)
        ax.barh(y_pos, top_weights, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s[:8] for s in top_species], fontsize=7)
        ax.invert_yaxis()

        direction = "wet" if moist_r > 0 else "dry"
        sig = "*" if moisture[moisture["factor"] == k]["p_value"].values[0] < 0.05 else ""
        ax.set_title(f"F{k} ({direction}){sig}", fontsize=9)
        despine(ax)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_sites_by_moisture(site_scores: pd.DataFrame, env_df: pd.DataFrame, output_path: Path) -> None:
    """2D plot of sites colored by moisture level."""
    apply_theme()

    # Use two factors with strongest opposite moisture correlations
    from scipy.stats import spearmanr
    moisture = env_df["Moisture"].values.astype(float)

    corrs = []
    for k in range(site_scores.shape[1]):
        r, _ = spearmanr(site_scores.iloc[:, k], moisture)
        corrs.append((k, r))

    # Find most positive and most negative
    corrs.sort(key=lambda x: x[1])
    dry_factor = corrs[0][0]
    wet_factor = corrs[-1][0]

    fig, ax = create_figure("square")

    scatter = ax.scatter(
        site_scores.iloc[:, dry_factor],
        site_scores.iloc[:, wet_factor],
        c=moisture,
        cmap="YlGnBu",
        s=80,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Moisture", fontsize=8)

    ax.set_xlabel(f"Factor {dry_factor} (dry)")
    ax.set_ylabel(f"Factor {wet_factor} (wet)")
    ax.set_title("Sites in ecological space")
    despine(ax)

    save_figure(fig, output_path)


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs"
    data_dir = base_dir.parent.parent / "data" / "cestes"

    # Load data
    embedding_df = pd.read_csv(output_dir / "species_embedding.csv", index_col=0)
    site_scores = pd.read_csv(output_dir / "site_scores.csv", index_col=0)
    validation_df = pd.read_csv(output_dir / "validation.csv")
    env_df = pd.read_csv(data_dir / "dune_env.csv", sep="\t", index_col=0)

    print("Generating plots...")

    # 1. Key result: moisture validation scatter
    plot_moisture_validation(site_scores, env_df, output_dir / "moisture_validation.pdf")
    print("  - moisture_validation.pdf")

    # 2. Factor-moisture correlation bars
    plot_factor_moisture_bars(validation_df, output_dir / "moisture_correlations.pdf")
    print("  - moisture_correlations.pdf")

    # 3. Top species per factor with moisture direction
    plot_species_by_factor(embedding_df, validation_df, output_dir / "species_by_factor.pdf")
    print("  - species_by_factor.pdf")

    # 4. Sites in ecological space colored by moisture
    plot_sites_by_moisture(site_scores, env_df, output_dir / "sites_ecological_space.pdf")
    print("  - sites_ecological_space.pdf")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
