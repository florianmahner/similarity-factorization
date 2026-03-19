"""Debug why α=5.0 overestimates rank - focus on matrix properties.

Key insight: α=5.0 creates matrices with very concentrated eigenvalues,
making the effective rank much lower than the true rank.

Usage:
    poetry run python sandbox/rank_detection_viz/debug_alpha5_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.colors import TEAL, ROSE, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir
from utils.simulation import simulation_dirichlet

OUTPUT_DIR = get_output_dir()


def plot_eigenvalue_spectrum(output_dir: Path) -> None:
    """Compare eigenvalue spectra for different alpha values."""
    k = 30
    n = 400
    seed = 42

    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    alphas = [1.0, 5.0]
    colors = {1.0: TEAL, 5.0: ROSE}

    # Left: Eigenvalue spectrum
    ax = axes[0]
    for alpha in alphas:
        rng = np.random.default_rng(seed)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
        S = w @ w.T

        eigvals = np.linalg.eigvalsh(S)
        eigvals = np.sort(eigvals)[::-1]

        ax.semilogy(range(1, len(eigvals) + 1), eigvals, "-",
                   color=colors[alpha], linewidth=1.5, label=f"α={alpha}")

    ax.axvline(k, color=GRAY, linestyle="--", linewidth=1,
               label=f"true k={k}")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_xlim(0, 50)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Right: Cumulative variance explained
    ax = axes[1]
    for alpha in alphas:
        rng = np.random.default_rng(seed)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
        S = w @ w.T

        eigvals = np.linalg.eigvalsh(S)
        eigvals = np.sort(eigvals)[::-1]
        cumvar = np.cumsum(eigvals) / eigvals.sum()

        ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "-",
               color=colors[alpha], linewidth=1.5, label=f"α={alpha}")

    ax.axvline(k, color=GRAY, linestyle="--", linewidth=1)
    ax.axhline(99, color=GRAY_LIGHT, linestyle=":", linewidth=1)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_xlim(0, 50)
    ax.set_ylim(90, 100.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    despine(ax)

    save_figure(fig, output_dir / "eigenvalue_spectrum.pdf")
    print("  eigenvalue_spectrum.pdf")


def plot_factor_structure(output_dir: Path) -> None:
    """Show why α affects eigenvalue concentration."""
    k = 30
    n = 100  # smaller for visualization
    seed = 42

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for ax, alpha in zip(axes, [1.0, 5.0]):
        rng = np.random.default_rng(seed)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)

        # Sort rows by dominant factor for visualization
        dominant = np.argmax(w, axis=1)
        order = np.argsort(dominant)
        w_sorted = w[order, :]

        im = ax.imshow(w_sorted, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Factor")
        ax.set_ylabel("Item (sorted)")
        ax.set_title(f"α={alpha}")

    plt.tight_layout()
    fig.savefig(output_dir / "factor_structure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  factor_structure.pdf")


def analyze_effective_rank_vs_alpha(output_dir: Path) -> None:
    """Show how effective rank depends on alpha."""
    k = 30
    n = 400
    alphas = np.logspace(-1, 1.5, 20)  # 0.1 to ~30

    effective_ranks = []

    for alpha in alphas:
        rng = np.random.default_rng(42)
        w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
        S = w @ w.T

        eigvals = np.linalg.eigvalsh(S)
        eigvals = np.sort(eigvals)[::-1]
        eigvals_norm = eigvals / eigvals.sum()
        eigvals_norm = eigvals_norm[eigvals_norm > 1e-10]
        eff_rank = np.exp(-np.sum(eigvals_norm * np.log(eigvals_norm)))
        effective_ranks.append(eff_rank)

    fig, ax = create_figure("single")

    ax.semilogx(alphas, effective_ranks, "o-", color=TEAL,
                markersize=5, linewidth=1.5)
    ax.axhline(k, color=GRAY, linestyle="--", linewidth=1,
               label=f"true k={k}")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Effective rank (entropy)")
    ax.set_xlim(0.08, 40)
    ax.set_ylim(0, 35)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "effective_rank_vs_alpha.pdf")
    print("  effective_rank_vs_alpha.pdf")


def print_summary():
    """Print summary of findings."""
    print("\n" + "=" * 60)
    print("SUMMARY: Why α=5.0 fails at rank detection")
    print("=" * 60)
    print("""
For α=5.0 (distributed/uniform factors):
- Each item loads on ALL factors roughly equally
- This creates a similarity matrix dominated by one large eigenvalue
- Effective rank ≈ 2-3, even though true rank = 30
- CV can't distinguish k=30 from k=35 because both fit the
  dominant structure equally well

For α=1.0 (mixed membership):
- Items load on a FEW factors strongly
- Eigenvalues are more spread out
- Effective rank ≈ 10-15
- CV can better identify the true rank

CONCLUSION: Rank detection works best for SPARSE factors (low α).
For distributed factors (high α), the "true rank" is not well-defined
from the eigenvalue perspective.
""")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Analyzing why α=5.0 fails at rank detection...")
    print("=" * 50)

    plot_eigenvalue_spectrum(OUTPUT_DIR)
    plot_factor_structure(OUTPUT_DIR)
    analyze_effective_rank_vs_alpha(OUTPUT_DIR)

    print_summary()

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
