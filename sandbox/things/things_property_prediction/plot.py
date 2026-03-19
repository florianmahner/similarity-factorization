"""
Plotting for THINGS Property Prediction Analysis.

Key visualizations:
1. Method comparison (SRF vs SPoSE vs ViCE) for property prediction
2. Dimension interpretability - which SRF dims predict which properties
3. Dimension-property heatmap
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT, setup_style
from src.utils.figure_theme import create_figure, save_figure, despine


PROPERTY_COLS = [
    "manmade_mean", "precious_mean", "lives_mean", "heavy_mean",
    "natural_mean", "moves_mean", "grasp_mean", "hold_mean",
    "be.moved_mean", "pleasant_mean"
]

PROPERTY_NAMES = [
    "Manmade", "Precious", "Lives", "Heavy",
    "Natural", "Moves", "Graspable", "Holdable",
    "Moveable", "Pleasant"
]


def plot_method_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot comparing SRF, SPoSE, ViCE for property prediction."""
    setup_style()

    fig, ax = create_figure("wide", pad_right=0.3)

    pivot = comparison_df.pivot(index="property", columns="method", values="correlation")
    pivot = pivot[["SRF", "SPoSE", "ViCE"]]  # Order

    x = np.arange(len(pivot))
    width = 0.25

    colors = [ROSE, TEAL, CYAN]
    for i, (method, color) in enumerate(zip(pivot.columns, colors)):
        ax.bar(x + i * width, pivot[method], width, label=method, color=color, alpha=0.85)

    ax.set_ylabel("Correlation (r)")
    ax.set_xlabel("")
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color=GRAY_LIGHT, linestyle="--", linewidth=0.8)

    despine(ax)
    save_figure(fig, output_path, tight=True)


def plot_compression_advantage(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Show that SRF achieves similar performance with fewer dimensions."""
    setup_style()

    fig, ax = create_figure("single")

    methods = ["SRF", "SPoSE", "ViCE"]
    dims = [20, 66, 67]  # From the analysis
    mean_corrs = []

    for method in methods:
        mean_corr = comparison_df[comparison_df["method"] == method]["correlation"].mean()
        mean_corrs.append(mean_corr)

    colors = [ROSE, TEAL, CYAN]
    bars = ax.bar(methods, mean_corrs, color=colors, alpha=0.85)

    # Add dimension labels
    for bar, dim in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{dim}D", ha="center", fontsize=8)

    ax.set_ylabel("Mean correlation (r)")
    ax.set_ylim(0, 0.9)
    ax.axhline(0.7, color=GRAY_LIGHT, linestyle="--", linewidth=0.8)

    despine(ax)
    save_figure(fig, output_path, tight=True)


def compute_dimension_property_matrix(
    embedding: np.ndarray,
    properties: np.ndarray,
) -> np.ndarray:
    """Compute correlation matrix between dimensions and properties."""
    n_dims = embedding.shape[1]
    n_props = len(PROPERTY_NAMES)

    corr_matrix = np.zeros((n_dims, n_props))

    for prop_idx in range(n_props):
        y = properties[:, prop_idx]
        valid = ~np.isnan(y)

        for dim in range(n_dims):
            x = embedding[valid, dim]
            corr, _ = spearmanr(x, y[valid])
            corr_matrix[dim, prop_idx] = corr

    return corr_matrix


def plot_dimension_property_heatmap(
    embedding: np.ndarray,
    properties: np.ndarray,
    output_path: Path,
) -> None:
    """Heatmap showing which dimensions predict which properties."""
    setup_style()

    corr_matrix = compute_dimension_property_matrix(embedding, properties)

    # Select top dimensions (those with highest max correlation)
    max_corrs = np.abs(corr_matrix).max(axis=1)
    top_dims = np.argsort(max_corrs)[-12:][::-1]
    corr_subset = corr_matrix[top_dims, :]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    im = ax.imshow(corr_subset, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(PROPERTY_NAMES)))
    ax.set_xticklabels(PROPERTY_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_dims)))
    ax.set_yticklabels([f"D{d}" for d in top_dims], fontsize=8)
    ax.set_ylabel("SRF dimension")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)", fontsize=8)

    # Add correlation values
    for i in range(len(top_dims)):
        for j in range(len(PROPERTY_NAMES)):
            val = corr_subset[i, j]
            if abs(val) > 0.3:
                color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    save_figure(fig, output_path, tight=True)


def plot_interpretable_dimensions(
    embedding: np.ndarray,
    properties: np.ndarray,
    output_path: Path,
) -> None:
    """Show the key interpretable dimensions with scatter plots."""
    setup_style()

    # Find the animacy/natural dimension (highest correlation with Lives or Natural)
    corr_matrix = compute_dimension_property_matrix(embedding, properties)

    # D5: Natural/Lives dimension
    natural_idx = PROPERTY_NAMES.index("Natural")
    lives_idx = PROPERTY_NAMES.index("Lives")
    manmade_idx = PROPERTY_NAMES.index("Manmade")

    animacy_dim = np.argmax(np.abs(corr_matrix[:, natural_idx]))

    # D18: Manipulability dimension (Graspable/Holdable)
    grasp_idx = PROPERTY_NAMES.index("Graspable")
    manip_dim = np.argmax(np.abs(corr_matrix[:, grasp_idx]))

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Plot 1: Animacy dimension vs Natural
    ax = axes[0]
    valid = ~np.isnan(properties[:, natural_idx])
    x = embedding[valid, animacy_dim]
    y = properties[valid, natural_idx]
    ax.scatter(x, y, alpha=0.3, s=10, c=TEAL)
    r, _ = spearmanr(x, y)
    ax.set_xlabel(f"D{animacy_dim}")
    ax.set_ylabel("Natural rating")
    ax.set_title(f"r = {r:.2f}", fontsize=9)
    despine(ax)

    # Plot 2: Animacy vs Manmade (should be negative)
    ax = axes[1]
    valid = ~np.isnan(properties[:, manmade_idx])
    x = embedding[valid, animacy_dim]
    y = properties[valid, manmade_idx]
    ax.scatter(x, y, alpha=0.3, s=10, c=ROSE)
    r, _ = spearmanr(x, y)
    ax.set_xlabel(f"D{animacy_dim}")
    ax.set_ylabel("Manmade rating")
    ax.set_title(f"r = {r:.2f}", fontsize=9)
    despine(ax)

    # Plot 3: Manipulability dimension vs Graspable
    ax = axes[2]
    valid = ~np.isnan(properties[:, grasp_idx])
    x = embedding[valid, manip_dim]
    y = properties[valid, grasp_idx]
    ax.scatter(x, y, alpha=0.3, s=10, c=CYAN)
    r, _ = spearmanr(x, y)
    ax.set_xlabel(f"D{manip_dim}")
    ax.set_ylabel("Graspable rating")
    ax.set_title(f"r = {r:.2f}", fontsize=9)
    despine(ax)

    plt.tight_layout()
    save_figure(fig, output_path, tight=True)


def plot_srf_vs_spose_scatter(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot comparing SRF and SPoSE correlations per property."""
    setup_style()

    fig, ax = create_figure("square")

    srf = comparison_df[comparison_df["method"] == "SRF"].set_index("property")["correlation"]
    spose = comparison_df[comparison_df["method"] == "SPoSE"].set_index("property")["correlation"]

    ax.scatter(spose, srf, s=60, c=TEAL, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Label points
    for prop in srf.index:
        ax.annotate(prop[:4], (spose[prop], srf[prop]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    # Diagonal
    ax.plot([0.5, 0.9], [0.5, 0.9], "--", color=GRAY, linewidth=1)

    ax.set_xlabel("SPoSE (66D)")
    ax.set_ylabel("SRF (20D)")
    ax.set_xlim(0.55, 0.9)
    ax.set_ylim(0.55, 0.9)
    ax.set_aspect("equal")

    despine(ax)
    save_figure(fig, output_path, tight=True)


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs"
    data_dir = base_dir.parent.parent / "data" / "things"

    # Load results
    comparison_df = pd.read_csv(output_dir / "property_prediction_comparison.csv")
    srf_embedding = np.load(output_dir / "srf_embedding.npy")
    properties = pd.read_csv(data_dir / "things_property_ratings.csv")[PROPERTY_COLS].values

    print("Generating plots...")

    # 1. Method comparison bar chart
    plot_method_comparison(comparison_df, output_dir / "method_comparison.pdf")
    print("  - method_comparison.pdf")

    # 2. Compression advantage
    plot_compression_advantage(comparison_df, output_dir / "compression_advantage.pdf")
    print("  - compression_advantage.pdf")

    # 3. Dimension-property heatmap
    plot_dimension_property_heatmap(srf_embedding, properties, output_dir / "dimension_property_heatmap.pdf")
    print("  - dimension_property_heatmap.pdf")

    # 4. Interpretable dimensions scatter
    plot_interpretable_dimensions(srf_embedding, properties, output_dir / "interpretable_dimensions.pdf")
    print("  - interpretable_dimensions.pdf")

    # 5. SRF vs SPoSE scatter
    plot_srf_vs_spose_scatter(comparison_df, output_dir / "srf_vs_spose.pdf")
    print("  - srf_vs_spose.pdf")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
