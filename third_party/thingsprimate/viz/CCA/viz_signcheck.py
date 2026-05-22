#!/usr/bin/env python3
"""
Species congruence check

Compute, for each CCA component k, the Pearson correlation across stimuli
between the species-averaged component scores:

    r_k = corr(mean_human[:, k], mean_monkey[:, k])

This directly tests whether component axes are shared across species.
Saves a per-component line plot and a histogram, plus CSV of r_k.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from functions.plotting import setup_style, style_axes, medium_figure, square_figure, narrow_figure
from config.paths import config


def _avg_by_species(cca_all):
    comps = cca_all.get('components') or cca_all.get('comps')
    if comps is None:
        raise KeyError("cca_all missing 'components' (or 'comps')")
    views = sorted(comps)
    human = [v for v in views if 'human' in v]
    monkey = [v for v in views if 'monkey' in v]
    if not human or not monkey:
        raise ValueError("Need both human and monkey views for congruence")
    H = np.mean(np.stack([np.asarray(comps[v]) for v in human], axis=0), axis=0)
    M = np.mean(np.stack([np.asarray(comps[v]) for v in monkey], axis=0), axis=0)
    return H, M


def compute_congruence(H, M, use_abs=False):
    K = H.shape[1]
    r = np.zeros(K, dtype=float)
    for k in range(K):
        x = H[:, k]
        y = M[:, k]
        if np.std(x) == 0 or np.std(y) == 0:
            r[k] = np.nan
        else:
            val = np.corrcoef(x, y)[0, 1]
            r[k] = abs(val) if use_abs else val
    return r


def plot_congruence_line(r, highlight_idx=None):
    setup_style()
    fig = medium_figure()
    ax = fig.add_axes([.12, .22, .82, .70])
    x = np.arange(1, len(r) + 1)
    ax.plot(x, r, '-', color='#444', lw=2.5)
    if highlight_idx is not None and 1 <= highlight_idx <= len(r):
        ax.axvline(highlight_idx, color=config.plotting.get('human_color', '#7c5799'), ls='--', lw=3)
    ax.set_xlabel('Component (1-indexed)')
    ax.set_ylabel('Human–Monkey congruence (r)')
    ax.set_ylim(0.0, 1.0)
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_congruence_hist(r):
    setup_style()
    fig = medium_figure()
    ax = fig.add_axes([.18, .18, .76, .74])
    vals = np.abs(r[~np.isnan(r)])
    ax.hist(vals, bins=20, color=config.plotting.get('shared_color', '#81B7B3'), edgecolor='black')
    ax.set_xlabel('Human–Monkey congruence (r)')
    ax.set_ylabel('Count')
    ax.set_xlim(0.0, 1.0)
    style_axes(ax)
    fig.tight_layout()
    return fig


def save_all(r, highlight_idx=None):
    out_fig = os.path.join(config.fig_dir, 'check')
    out_res = os.path.join(config.results_dir, 'check')
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_res, exist_ok=True)

    fig1 = plot_congruence_line(r, highlight_idx=highlight_idx)
    fig1.savefig(os.path.join(out_fig, 'species_congruence_line.pdf'),
                 bbox_inches='tight', pad_inches=0.01)
    plt.close(fig1)

    fig2 = plot_congruence_hist(r)
    fig2.savefig(os.path.join(out_fig, 'species_congruence_hist.pdf'),
                 bbox_inches='tight', pad_inches=0.01)
    plt.close(fig2)

    pd.DataFrame({'component_1idx': np.arange(1, len(r)+1), 'r_human_monkey': r}).to_csv(
        os.path.join(out_res, 'species_congruence.csv'), index=False
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Species congruence (human–monkey) per component')
    parser.add_argument('--abs', action='store_true', help='Use absolute correlation')
    parser.add_argument('--highlight', type=int, default=None, help='1-indexed component to highlight')
    args = parser.parse_args()

    with open(config.cache_file, 'rb') as f:
        state = pickle.load(f)
    cca_all = state['cca_all']
    H, M = _avg_by_species(cca_all)
    r = compute_congruence(H, M, use_abs=args.abs)
    save_all(r, highlight_idx=args.highlight)
    print(f'Done. Saved figures to {os.path.join(config.fig_dir, "check")} and CSV to {os.path.join(config.results_dir, "check")}')


if __name__ == '__main__':
    main()
