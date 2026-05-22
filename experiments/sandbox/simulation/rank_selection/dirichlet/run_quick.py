"""Quick CV curve analysis for specific k/alpha combinations.

Usage:
    poetry run python sandbox/rank_selection/dirichlet/run_quick.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet

PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra
from src.colors import TEAL, ROSE, CYAN, setup_style
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def generate_dirichlet(n: int, k: int, alpha: float, seed: int = 0) -> np.ndarray:
    W = dirichlet.rvs(alpha=[alpha] * k, size=n, random_state=seed)
    return W @ W.T


def plot_cv_curves(n: int, k: int, alphas: list[float], output_dir: Path, title_suffix: str = ""):
    """Plot CV curves for different alpha at fixed n, k."""
    setup_style()

    fig, axes = plt.subplots(1, len(alphas), figsize=(3.5 * len(alphas), 3.5))
    if len(alphas) == 1:
        axes = [axes]

    for i, alpha in enumerate(alphas):
        ax = axes[i]

        S = generate_dirichlet(n, k, alpha, seed=42)
        eig = np.linalg.eigvalsh(S)[::-1]
        gap = eig[k-1] - eig[k] if k < len(eig) else 0

        # Larger range to see full U-curve
        grid = list(range(max(2, k-20), min(k+25, n-1), 2))
        if k not in grid:
            grid.append(k)
        grid = sorted(grid)

        pmin, pmax, _ = estimate_sampling_bounds_ultra(S)
        p_mean = (pmin + pmax) / 2
        if p_mean <= 0.1 or p_mean >= 0.95:
            p_mean = 0.5

        print(f"  alpha={alpha}: p_mean={p_mean:.3f}, gap={gap:.4f}")

        # More repeats for smoother curves
        cv = cross_val_score(
            S, param_grid={'rank': grid},
            sampling_fraction=p_mean, n_repeats=10, n_jobs=-1, verbose=0
        )

        cv_df = cv.cv_results_
        means = np.array([cv_df[cv_df['rank'] == r]['score'].mean() for r in grid])
        stds = np.array([cv_df[cv_df['rank'] == r]['score'].std() for r in grid])

        # Plot raw scores to see U-shape
        ax.errorbar(grid, means, yerr=stds, fmt='o-', color=TEAL,
                   capsize=2, markersize=4, lw=1.5)

        ax.axvline(k, color=ROSE, linestyle='--', lw=2, label=f'True k={k}')

        selected = grid[np.argmin(means)]
        ax.axvline(selected, color=CYAN, linestyle=':', lw=2, label=f'Selected={selected}')

        ax.set_xlabel('Candidate rank')
        if i == 0:
            ax.set_ylabel('CV score (MSE)')
        ax.set_title(f'alpha={alpha}\np={p_mean:.2f}, gap={gap:.3f}', fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        despine(ax)

    fig.suptitle(f'CV curves: n={n}, k={k}{title_suffix}', fontsize=11, y=1.02)
    plt.tight_layout()

    fname = f'cv_curves_n{n}_k{k}.pdf'
    save_figure(fig, output_dir / fname)
    print(f'Saved: {fname}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output: {OUTPUT_DIR}\n')

    # Test 1: k=20, n=300, different alphas
    print("=== Test 1: k=20, n=300 ===")
    plot_cv_curves(n=300, k=20, alphas=[0.1, 1.0, 2.0, 5.0], output_dir=OUTPUT_DIR)

    # Test 2: k=50, n=1000, different alphas (excluding 0.5)
    print("\n=== Test 2: k=50, n=1000 ===")
    plot_cv_curves(n=1000, k=50, alphas=[0.1, 1.0, 2.0, 5.0], output_dir=OUTPUT_DIR)

    print(f'\nAll outputs: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
