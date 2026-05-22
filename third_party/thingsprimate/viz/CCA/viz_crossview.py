# crossview_cca_viz_withshuff.py
# -*- coding: utf-8 -*-
"""Simplified cross-view CCA visualisation with shuffling results."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys, os
from statsmodels.stats.multitest import multipletests

# ──────────────────────────────────────────────────────────────────────
proj_root = Path(__file__).resolve().parents[2]
if str(proj_root) not in sys.path:
    sys.path.append(str(proj_root))

from functions.plotting import setup_style, style_axes, format_axes, square_panel
from config.paths import config

# ──────────────────────────────────────────────────────────────────────
setup_style()
RES = config.results_dir / 'crossview_results.pkl'
with open(RES, 'rb') as f:
    res = pickle.load(f)

FAMS = {
    'Cross-species': res['cross-species'],
    'Human-only'  : res['human_only'],
    'Monkey-only' : res['monkey_only']
}

NULL_DISTS = res['null_distributions']
lbls_map = res['view_label_map']
alphas = res['alphas']
n_cc = res['num_cc']

col_hum = config.plotting['human_color']
col_mon = config.plotting['monkey_color']
col_sig = config.plotting['significant']

OUT = config.fig_dir / 'crossview'
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# COMPUTE SIGNIFICANCE AND FDR CORRECTION
alpha = 0.05  # Default significance threshold

def compute_sig_and_fdr(obs_corrs, null_corrs, alpha=0.05):
    """Compute p-values, FDR correction, and significance thresholds."""
    # Compute p-values for each component
    p_vals = []
    for comp in range(len(obs_corrs)):
        p_val = np.mean(null_corrs[:, comp] >= obs_corrs[comp])
        p_vals.append(p_val)
    
    # FDR correction using Benjamini-Hochberg
    p_vals = np.array(p_vals)
    rejected, q_vals, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')
    
    # Find FDR threshold
    fdr_sig_comps = np.where(rejected)[0].tolist()
    
    if len(fdr_sig_comps) > 0:
        # Find the minimum observed correlation among significant components
        # This represents the FDR threshold
        fdr_thresh_corr = np.min(obs_corrs[rejected])
    else:
        fdr_thresh_corr = None
    
    return {
        'p_values': p_vals.tolist(),
        'q_values': q_vals.tolist(),
        'significant_fdr': fdr_sig_comps,
        'fdr_threshold_corr': fdr_thresh_corr,
        'optimal_n_cc': len(fdr_sig_comps)
    }

def find_first_contig_stretch(sig_comps):
    """Find the highest component number in the first contiguous stretch."""
    if not sig_comps:
        return 0
    
    # Sort to ensure order
    sorted_comps = sorted(sig_comps)
    
    # Find the first contiguous stretch
    first_stretch_end = sorted_comps[0]
    
    for i in range(1, len(sorted_comps)):
        if sorted_comps[i] == sorted_comps[i-1] + 1:
            first_stretch_end = sorted_comps[i]
        else:
            break
    
    return first_stretch_end + 1  # Convert to 1-indexed component number

# Compute significance for each family
SIG_RESULTS = {}
for fam_key, null_key in [('Cross-species', 'cross-species'), 
                         ('Human-only', 'human_only'), 
                         ('Monkey-only', 'monkey_only')]:
    SIG_RESULTS[null_key] = compute_sig_and_fdr(
        FAMS[fam_key]['comp_corrs'], 
        NULL_DISTS[null_key], 
        alpha
    )

print(f"\n=== FDR CORRECTION RESULTS (alpha={alpha}) ===")
for fam_key, sig in SIG_RESULTS.items():
    fdr_sig = len(sig['significant_fdr'])
    print(f"{fam_key.capitalize()}: {fdr_sig} significant components after FDR: {sig['significant_fdr']}")

# Print neat table showing first contiguous stretch
print(f"\n=== FIRST CONTIGUOUS COMPONENTS (α={alpha}) ===")
print(f"{'Family':<12} {'Total':<8} {'Contiguous':<12} {'Range':<10}")
print("-" * 45)
for fam_key, sig in SIG_RESULTS.items():
    total_sig = len(sig['significant_fdr'])
    first_contig = find_first_contig_stretch(sig['significant_fdr'])
    
    if first_contig > 0:
        range_str = f"1-{first_contig}"
    else:
        range_str = "None"
    
    family_display = {
        'cross-species': 'Cross-species',
        'human_only': 'Human',
        'monkey_only': 'Monkey'
    }.get(fam_key, fam_key.replace('_', ' ').title())
    print(f"{family_display:<12} {total_sig:<8} {first_contig:<12} {range_str:<10}")
print()

# ──────────────────────────────────────────────────────────────────────
def scree_plot(fam_res, null_dist, sig_res, title, show_legend=False):
    """Create scree plot with null distribution, FDR threshold, and significance highlighting."""
    fig, ax = square_panel()  # same inner panel as ridge

    x = np.arange(1, n_cc + 1)
    y = fam_res['comp_corrs']
    sig_comps = [comp + 1 for comp in sig_res['significant_fdr']]  # Convert to 1-indexed
    
    # Null distribution 95th percentile
    null_95 = np.percentile(null_dist, 95, axis=0)
    
    # Gradient background
    y_fill = np.linspace(0, max(y)*1.1, 50)
    for i, y_val in enumerate(y_fill[:-1]):
        # Flipped gradient: darker at top, lighter at bottom
        alpha_val = 0.2 * (1 - y_val / max(y_fill))
        ax.fill_between(x, y_val, y_fill[i+1], alpha=alpha_val, color='gray', edgecolor='none', zorder=0)
    
    # 95% significance threshold line
    ax.plot(x, null_95, color=col_sig, linestyle='-', linewidth=2.25, 
           zorder=4, label='Null 95th percentile')
    
    # Significant component highlighting - continuous ranges (back layer)
    if sig_comps:
        # Find continuous ranges of significant components
        ranges = []
        start = sig_comps[0]
        end = sig_comps[0]
        
        for i in range(1, len(sig_comps)):
            if sig_comps[i] == end + 1:
                end = sig_comps[i]
            else:
                ranges.append((start, end))
                start = sig_comps[i]
                end = sig_comps[i]
        ranges.append((start, end))
        
        # Fill continuous ranges with transparent purple
        for start_comp, end_comp in ranges:
            if start_comp <= len(y) and end_comp <= len(y):
                x_range = np.arange(start_comp, end_comp + 1)
                y_range = y[start_comp-1:end_comp]  # Convert back to 0-indexed for array indexing
                ax.fill_between(x_range, 0, y_range, alpha=0.4, color='#7a3843', 
                              edgecolor='none', zorder=1)
    
    # Main observed correlations line (single color)
    ax.plot(x, y, 'k-', lw=3, zorder=5)
    
    ax.set_xlim(1, n_cc)
    ax.set_ylim(0, 0.95)
    # Explicit ticks: always show component 1 plus every 25th component thereafter
    tick_marks = [1]
    tick_marks.extend(t for t in range(25, n_cc + 1, 25))
    ax.set_xticks(sorted(set(tick_marks)))
    
    # Clean formatting
    format_axes(ax)
    ax.set_xlabel('Component')
    ax.set_ylabel('Cross-view similarity\n(Pearson\'s $r$)')
    # ax.set_ylabel('Cross-view similarity)')
    ax.set_title(f'{title}')
    
    # Add legend only to cross-species plot
    if show_legend:
        # Create a proper legend with only the relevant elements
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#7a3843', alpha=0.4, edgecolor='none', label='p < 0.05'),
            Line2D([0], [0], color=col_sig, lw=2.25, label='95th percentile H$_0$')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False)
    
    style_axes(ax)
    # plt.tight_layout()  # Removed to avoid content-dependent resizing
    
    return fig

def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# Generate individual plots
figs = {}
for name, fam_key in [('cross-species', 'cross-species'), 
                     ('human', 'human_only'), 
                     ('monkey', 'monkey_only')]:
    title = {'cross-species': 'Cross-species', 'human': 'Human', 'monkey': 'Monkey'}[name]
    # Add legend only to cross-species plot
    show_legend = (name == 'cross-species')
    figs[name] = scree_plot(
        FAMS[title if title != 'Human' and title != 'Monkey' else f'{title}-only'], 
        NULL_DISTS[fam_key], 
        SIG_RESULTS[fam_key], 
        title,
        show_legend
    )

for k, f in figs.items():
    save_fig(f, k)

print(f"\nFigures saved to: {OUT}")
