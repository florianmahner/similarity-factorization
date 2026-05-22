"""Visualize how Dirichlet alpha affects cluster structure and similarity matrix.

Uses the project's figure theme for consistent, publication-quality styling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.figure_theme import (
    CMAP,
    GRAY,
    HEATMAPS,
    apply_theme,
    clean_axis,
    despine,
    save_figure,
)


def simulation_dirichlet(
    n: int,
    k: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate soft cluster memberships from symmetric Dirichlet distribution."""
    factors = rng.dirichlet(np.ones(k) * alpha, size=n)
    dominant = np.argmax(factors, axis=1)
    order = np.argsort(dominant)
    return factors[order]


def factors_to_rsm(factors: np.ndarray) -> np.ndarray:
    """Compute RSM from factor matrix (linear kernel)."""
    rsm = factors @ factors.T
    rsm = (rsm + rsm.T) / 2
    return rsm


def compute_metrics(
    alphas: np.ndarray,
    n: int,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Compute clustering metrics across alpha values."""
    metrics = {
        "max_membership": np.zeros((len(alphas), n_samples)),
        "rsm_contrast": np.zeros((len(alphas), n_samples)),
    }

    for i, alpha in enumerate(alphas):
        for j in range(n_samples):
            factors = simulation_dirichlet(n, k, alpha, rng)
            rsm = factors_to_rsm(factors)

            metrics["max_membership"][i, j] = np.mean(np.max(factors, axis=1))
            metrics["rsm_contrast"][i, j] = rsm.max() - rsm.min()

    return metrics


def plot_alpha_effect_combined(output_path: Path | None = None) -> plt.Figure:
    """Create combined visualization: RSM examples + metric curves."""
    apply_theme()

    example_alphas = [0.1, 1.0, 10.0]
    curve_alphas = np.logspace(-1, 1.5, 25)
    n, k = 16, 4
    n_samples = 50
    rng = np.random.default_rng(42)

    # Compute metrics for curves
    metrics = compute_metrics(curve_alphas, n, k, n_samples, rng)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(5.5, 4.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # Top row: RSM examples at 3 alpha values
    for col, alpha in enumerate(example_alphas):
        ax = fig.add_subplot(gs[0, col])
        rng_example = np.random.default_rng(42 + col)
        factors = simulation_dirichlet(n, k, alpha, rng_example)
        rsm = factors_to_rsm(factors)

        im = ax.imshow(rsm, cmap=HEATMAPS["diverging"], vmin=0, vmax=1)
        clean_axis(ax)
        ax.set_title(rf"$\alpha$ = {alpha}", fontsize=10, fontweight="bold")

    # Add colorbar for RSMs
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Bottom left: Max cluster membership
    ax1 = fig.add_subplot(gs[1, :2])
    y_mean = metrics["max_membership"].mean(axis=1)
    y_sem = metrics["max_membership"].std(axis=1) / np.sqrt(n_samples)

    ax1.plot(curve_alphas, y_mean, "o-", color=CMAP[2], markersize=4, linewidth=1.5)
    ax1.fill_between(curve_alphas, y_mean - y_sem, y_mean + y_sem, color=CMAP[2], alpha=0.2, linewidth=0)

    # Reference line for uniform distribution
    ax1.axhline(1 / k, color=GRAY["light"], linestyle="--", linewidth=0.8, zorder=0)
    ax1.text(0.12, 1 / k + 0.02, f"uniform (1/{k})", fontsize=7, color=GRAY["medium"])

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\alpha$ (Dirichlet concentration)", fontsize=10)
    ax1.set_ylabel("Max cluster membership", fontsize=10)
    ax1.set_xlim(0.1, 30)
    ax1.set_ylim(0.15, 0.9)
    despine(ax1)

    # Bottom right: RSM contrast
    ax2 = fig.add_subplot(gs[1, 2])
    y_mean = metrics["rsm_contrast"].mean(axis=1)
    y_sem = metrics["rsm_contrast"].std(axis=1) / np.sqrt(n_samples)

    ax2.plot(curve_alphas, y_mean, "o-", color=CMAP[3], markersize=4, linewidth=1.5)
    ax2.fill_between(curve_alphas, y_mean - y_sem, y_mean + y_sem, color=CMAP[3], alpha=0.2, linewidth=0)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\alpha$", fontsize=10)
    ax2.set_ylabel("RSM contrast", fontsize=10)
    ax2.set_xlim(0.1, 30)
    ax2.set_ylim(0, 1.05)
    despine(ax2)

    # Add annotation
    fig.text(
        0.5, 0.97,
        r"Sparse $\leftarrow$ Cluster structure $\rightarrow$ Uniform",
        ha="center", fontsize=9, style="italic", color=GRAY["medium"],
    )

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def plot_alpha_effect_minimal(output_path: Path | None = None) -> plt.Figure:
    """Create minimal visualization: W and S matrices only."""
    apply_theme()

    alphas = [0.1, 1.0, 10.0]
    n, k = 16, 4
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, len(alphas), figsize=(5, 3.5))

    # Title
    fig.text(
        0.5, 0.98, r"Sparse $\rightarrow$ Uniform",
        ha="center", fontsize=9, style="italic", color=GRAY["medium"],
    )

    for col, alpha in enumerate(alphas):
        factors = simulation_dirichlet(n, k, alpha, rng)
        rsm = factors_to_rsm(factors)

        # Top row: Factor matrix (W)
        ax = axes[0, col]
        ax.imshow(factors, aspect="auto", cmap=HEATMAPS["sequential"], vmin=0, vmax=1)
        clean_axis(ax)
        ax.set_title(rf"$\alpha$ = {alpha}", fontsize=10, fontweight="bold")

        # Bottom row: RSM (S) with per-panel scaling
        ax = axes[1, col]
        vmin, vmax = rsm.min(), rsm.max()
        im = ax.imshow(rsm, cmap=HEATMAPS["diverging"], vmin=vmin, vmax=vmax)
        clean_axis(ax)

        # Small colorbar showing range
        cax = ax.inset_axes([1.02, 0, 0.06, 1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_ticks([vmin, vmax])
        cbar.ax.yaxis.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

    # Row labels
    axes[0, 0].set_ylabel(r"$W$", fontsize=12, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel(r"$S$", fontsize=12, rotation=0, ha="right", va="center")

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Generate both versions
    plot_alpha_effect_combined(output_dir / "alpha_effect_combined.pdf")
    print(f"Saved: {output_dir / 'alpha_effect_combined.pdf'}")

    plot_alpha_effect_minimal(output_dir / "alpha_effect_minimal.pdf")
    print(f"Saved: {output_dir / 'alpha_effect_minimal.pdf'}")
