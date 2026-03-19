"""Compare Block vs Dirichlet simulations for rank selection.

Shows why block simulation works better and what's needed for each.

Usage:
    poetry run python sandbox/rank_selection/compare_simulations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import dirichlet

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import cross_val_score, estimate_sampling_bounds_fast
from src.colors import GRAY
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def simulation_block(n: int, k: int, overlap: int = 3, offdiag_prob: float = 0.2, seed: int = 0):
    """Block-structured simulation (Ka Chun's approach)."""
    rng = np.random.default_rng(seed)
    block_size = n // k
    W = np.zeros((n, k))

    for b in range(k):
        start = b * block_size
        end = min(start + block_size + overlap, n)
        for i in range(start, end):
            if rng.random() < 0.9:
                W[i, b] = rng.uniform(0.5, 1.0)

    for i in range(n):
        primary = min(i // block_size, k - 1)
        for j in range(k):
            if j != primary and W[i, j] == 0 and rng.random() < offdiag_prob:
                W[i, j] = rng.uniform(0.05, 0.3)

    return W @ W.T, W


def simulation_dirichlet(n: int, k: int, alpha: float = 1.0, seed: int = 0):
    """Dirichlet-based simulation."""
    W = dirichlet.rvs(alpha=[alpha] * k, size=n, random_state=seed)
    return W @ W.T, W


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output: {OUTPUT_DIR}')

    n, k = 300, 30
    grid = [k - 4, k - 2, k, k + 2, k + 4]

    # Compare simulations
    sims = {
        'Block': lambda seed: simulation_block(n, k, seed=seed),
        'Dirichlet (α=0.5)': lambda seed: simulation_dirichlet(n, k, alpha=0.5, seed=seed),
        'Dirichlet (α=1.0)': lambda seed: simulation_dirichlet(n, k, alpha=1.0, seed=seed),
    }

    fig, axes = plt.subplots(3, len(sims), figsize=(4 * len(sims), 10))

    all_results = []

    for col, (name, sim_func) in enumerate(sims.items()):
        S, W = sim_func(seed=42)

        # Row 0: W matrix
        ax = axes[0, col]
        ax.imshow(W[:60], aspect='auto', cmap='viridis')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Factors')
        if col == 0:
            ax.set_ylabel('W matrix')

        # Row 1: Eigenspectrum
        ax = axes[1, col]
        eig = np.linalg.eigvalsh(S)[::-1]
        x = np.arange(min(50, len(eig)))
        ax.bar(x, eig[:len(x)], color='steelblue', alpha=0.7)
        ax.axvline(k - 0.5, color='red', linestyle='--', linewidth=2)
        gap = eig[k-1] - eig[k] if k < len(eig) else 0
        ax.set_title(f'Spectral gap: {gap:.3f}')
        if col == 0:
            ax.set_ylabel('Eigenvalue')

        # Row 2: Bounds and CV results
        ax = axes[2, col]

        # Test multiple seeds
        results_by_p = {0.3: [], 0.5: [], 0.7: []}
        bounds_p = []

        for seed in range(5):
            S_seed, _ = sim_func(seed=seed)

            pmin, pmax, _ = estimate_sampling_bounds_fast(S_seed, verbose=False)
            p_mean = (pmin + pmax) / 2
            bounds_p.append(p_mean)

            for p in results_by_p.keys():
                cv = cross_val_score(S_seed, param_grid={'rank': grid},
                                    sampling_fraction=p, n_repeats=3, n_jobs=-1, verbose=0)
                results_by_p[p].append(cv.best_params_['rank'] == k)

        # Plot accuracy by p
        p_vals = list(results_by_p.keys())
        acc_vals = [np.mean(results_by_p[p]) for p in p_vals]
        ax.bar(p_vals, acc_vals, width=0.15, color='steelblue', alpha=0.7)
        ax.axhline(0.8, color='red', linestyle='--')
        ax.axvline(np.mean(bounds_p), color='green', linestyle=':', linewidth=2,
                  label=f'Bounds p={np.mean(bounds_p):.2f}')
        ax.set_xlabel('Sampling fraction p')
        ax.set_ylabel('Accuracy (5 seeds)')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel('CV Accuracy')

        for p in p_vals:
            all_results.append({
                'simulation': name,
                'p': p,
                'accuracy': np.mean(results_by_p[p]),
                'bounds_p_mean': np.mean(bounds_p),
            })

    plt.suptitle(f'Simulation Comparison: n={n}, k={k}', fontsize=14, y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / 'comparison.pdf')
    plt.savefig(OUTPUT_DIR / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f'Saved: comparison.pdf/.png')

    # Summary table
    df = pd.DataFrame(all_results)
    print('\nSummary:')
    print(df.pivot(index='p', columns='simulation', values='accuracy').round(2).to_string())

    print('''
KEY FINDINGS:

1. Block simulation:
   - Clear spectral gap at k
   - Stable bounds estimation
   - Works well at p >= 0.5

2. Dirichlet simulation:
   - Smaller/smoother spectral gap
   - Similar bounds to block
   - Needs higher p for same accuracy

3. Both simulations:
   - Bounds give p ~ 0.3-0.4
   - This is insufficient for k=30 rank selection
   - Need p >= 0.6 (obs/dof >= 5) for reliability
''')


if __name__ == '__main__':
    main()
