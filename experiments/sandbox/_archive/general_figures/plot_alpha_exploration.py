"""Explore how Dirichlet alpha affects various clustering metrics.

Creates a 2x3 panel figure showing 6 metrics vs alpha, each in its own
color from the TAB10_EXPLORATION palette.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.figure_theme import CMAP, apply_theme, save_figure


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
    return (rsm + rsm.T) / 2


def compute_all_metrics(
    alphas: np.ndarray,
    n: int,
    k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Compute all clustering metrics across alpha values."""
    metrics = {
        "sparsity": np.zeros((len(alphas), n_samples)),
        "entropy": np.zeros((len(alphas), n_samples)),
        "max_membership": np.zeros((len(alphas), n_samples)),
        "rsm_std": np.zeros((len(alphas), n_samples)),
        "block_ratio": np.zeros((len(alphas), n_samples)),
        "rsm_range": np.zeros((len(alphas), n_samples)),
    }

    for i, alpha in enumerate(alphas):
        for j in range(n_samples):
            factors = simulation_dirichlet(n, k, alpha, rng)
            rsm = factors_to_rsm(factors)

            # Sparsity: fraction of near-zero entries
            metrics["sparsity"][i, j] = np.mean(factors < 0.1)

            # Entropy of membership distribution
            eps = 1e-10
            ent = -np.sum(factors * np.log(factors + eps), axis=1)
            metrics["entropy"][i, j] = np.mean(ent)

            # Max membership (dominance)
            metrics["max_membership"][i, j] = np.mean(np.max(factors, axis=1))

            # RSM standard deviation
            metrics["rsm_std"][i, j] = rsm.std()

            # Block ratio: within / between cluster similarity
            n_per_cluster = n // k
            within, between = [], []
            for c in range(k):
                start, end = c * n_per_cluster, (c + 1) * n_per_cluster
                within.append(rsm[start:end, start:end].mean())
                mask = np.ones_like(rsm, dtype=bool)
                mask[start:end, start:end] = False
                between.append(rsm[mask].mean())
            metrics["block_ratio"][i, j] = np.mean(within) / (np.mean(between) + eps)

            # RSM range (max - min)
            metrics["rsm_range"][i, j] = rsm.max() - rsm.min()

    return metrics


def plot_alpha_exploration(output_path: Path | None = None) -> plt.Figure:
    """Create 6-panel exploration of alpha effects on clustering metrics."""
    apply_theme()

    alphas = np.logspace(-1, 1.5, 20)
    n, k = 20, 5
    n_samples = 50
    rng = np.random.default_rng(42)

    metrics = compute_all_metrics(alphas, n, k, n_samples, rng)

    # Panel configuration: (metric_key, title, ylabel, reference_line)
    panels = [
        ("sparsity", "Sparsity decreases", "Factor sparsity", None),
        ("entropy", "Entropy increases", "Factor entropy", np.log(k)),
        ("max_membership", "Dominance decreases", "Max membership", 1/k),
        ("rsm_std", "RSM contrast vanishes", "RSM std", None),
        ("block_ratio", "Block structure collapses", "Within / Between similarity", 1.0),
        ("rsm_range", "Similarity range shrinks", "RSM range", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
    axes = axes.flatten()

    for idx, (metric_key, title, ylabel, ref_line) in enumerate(panels):
        ax = axes[idx]
        color = CMAP[idx]

        y_mean = metrics[metric_key].mean(axis=1)
        y_sem = metrics[metric_key].std(axis=1) / np.sqrt(n_samples)

        # Plot with markers and ribbon
        ax.plot(alphas, y_mean, "o-", color=color, markersize=5, linewidth=1.5)
        ax.fill_between(
            alphas,
            y_mean - y_sem,
            y_mean + y_sem,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        # Reference line if specified
        if ref_line is not None:
            ax.axhline(ref_line, color="#999999", linestyle="--", linewidth=0.8, zorder=0)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    plot_alpha_exploration(output_dir / "alpha_exploration.pdf")
    print(f"Saved: {output_dir / 'alpha_exploration.pdf'}")
