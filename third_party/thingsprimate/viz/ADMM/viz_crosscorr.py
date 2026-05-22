#!/usr/bin/env python3
"""
Cross-family ANLS component correlation visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import MDS

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from config.paths import config
from functions.plotting import setup_style, square_figure, large_figure

setup_style()
np.random.seed(42)

# Colors
H_COL = config.plotting['human_color']
M_COL = config.plotting['monkey_color'] 
X_COL = '#7f7f7f'  # Cross-species

FAM_COLORS = {'all': X_COL, 'human': H_COL, 'monkey': M_COL}
FAM_LABELS = {'all': 'Cross-species', 'human': 'Human', 'monkey': 'Monkey'}

def load_data():
    """Load ANLS results."""
    file_path = os.path.join(config.results_dir, 'anls_results', 'anls_summary.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ANLS summary not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_corrs(comp_dict):
    """Compute correlation matrix across families."""
    all_comps = []
    fam_labels = []
    fam_idx = {}
    start = 0
    
    for fam, comps in comp_dict.items():
        n = comps.shape[1]
        all_comps.append(comps)
        fam_labels.extend([fam] * n)
        fam_idx[fam] = (start, start + n)
        start += n
    
    all_data = np.hstack(all_comps)
    corr_mat = np.corrcoef(all_data.T)
    
    return corr_mat, {'labels': fam_labels, 'idx': fam_idx, 'n': {f: comp_dict[f].shape[1] for f in comp_dict.keys()}}

def plot_heatmap(corr_mat, fam_info):
    """Detailed correlation heatmap."""
    fam_order = ['all', 'human', 'monkey']
    fams = [f for f in fam_order if f in fam_info['idx']]
    n_total = corr_mat.shape[0]
    
    fig = large_figure()
    ax = fig.add_subplot(111)
    
    # Lower triangle only
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    plot_mat = corr_mat.copy()
    plot_mat[mask] = np.nan
    
    im = ax.imshow(plot_mat, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
    
    # Family boundaries
    bounds = []
    for fam in fams:
        start, end = fam_info['idx'][fam]
        bounds.append((fam, start, end))
        if end < n_total:
            ax.axhline(end - 0.5, color='black', linewidth=3)
            ax.axvline(end - 0.5, color='black', linewidth=3)
    
    # Labels
    ticks, labels = [], []
    for fam, start, end in bounds:
        ticks.append((start + end - 1) / 2)
        labels.append(f"{FAM_LABELS[fam]}\n({end - start})")
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, ha='center', fontweight='bold')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, va='center', fontweight='bold')
    
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(n_total - 0.5, -0.5)
    ax.set_title('Cross-Family Component Correlations', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.7, label='Correlation')
    
    # Family color bars - add as inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_top = inset_axes(ax, width="100%", height="15%", loc='upper center', 
                       bbox_to_anchor=(0, 1.05), bbox_transform=ax.transAxes)
    ax_top.set_xlim(-0.5, n_total - 0.5)
    for fam, start, end in bounds:
        color = FAM_COLORS[fam]
        ax_top.barh(0.5, end - start, left=start, height=0.8, color=color, alpha=0.8)
        mid = (start + end) / 2
        abbrev = fam.upper()[0] if fam != 'all' else 'X'
        ax_top.text(mid, 0.5, abbrev, ha='center', va='center', fontweight='bold', color='white')
    ax_top.axis('off')
    
    return fig

def plot_mds_pairwise(corr_mat, fam_info):
    """MDS plot with family centroids."""
    # Convert correlation to distance and apply MDS
    dist_mat = 1 - np.abs(corr_mat)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_mat)
    
    # Calculate family centroids
    fam_centroids = {}
    for fam, (start, end) in fam_info['idx'].items():
        fam_coords = coords[start:end]
        fam_centroids[fam] = fam_coords.mean(axis=0)
    
    # Create visualization
    fig = large_figure()
    
    # MDS plot
    ax1 = fig.add_subplot(121)
    
    # Plot family clusters
    for fam, (start, end) in fam_info['idx'].items():
        fam_coords = coords[start:end]
        color = FAM_COLORS[fam]
        ax1.scatter(fam_coords[:, 0], fam_coords[:, 1], 
                   c=color, alpha=0.3, s=30, edgecolors='none')
    
    # Plot centroids
    for fam, centroid in fam_centroids.items():
        color = FAM_COLORS[fam]
        ax1.scatter(centroid[0], centroid[1], c=color, s=200, marker='*', 
                   edgecolors='black', linewidth=2, alpha=0.9)
        ax1.text(centroid[0], centroid[1] + 0.1, FAM_LABELS[fam], 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_xlabel('MDS Dimension 1')
    ax1.set_ylabel('MDS Dimension 2')
    ax1.set_title('Family Centroids in MDS Space')
    ax1.grid(True, alpha=0.3)
    
    # Family correlations
    ax2 = fig.add_subplot(122)
    
    # Calculate mean correlations between families
    fams = list(fam_info['idx'].keys())
    n_fams = len(fams)
    fam_corr_mat = np.zeros((n_fams, n_fams))
    
    for i, fam1 in enumerate(fams):
        start1, end1 = fam_info['idx'][fam1]
        for j, fam2 in enumerate(fams):
            start2, end2 = fam_info['idx'][fam2]
            
            if i == j:
                # Within-family correlation (excluding diagonal)
                within_corrs = corr_mat[start1:end1, start2:end2]
                mask = ~np.eye(within_corrs.shape[0], dtype=bool)
                fam_corr_mat[i, j] = within_corrs[mask].mean() if mask.sum() > 0 else 0
            else:
                # Between-family correlation
                between_corrs = corr_mat[start1:end1, start2:end2]
                fam_corr_mat[i, j] = np.mean(between_corrs)
    
    # Create lower triangle mask
    mask = np.triu(np.ones_like(fam_corr_mat, dtype=bool), k=1)
    fam_corr_mat[mask] = np.nan
    
    # Plot heatmap
    im = ax2.imshow(fam_corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(n_fams))
    ax2.set_yticks(range(n_fams))
    ax2.set_xticklabels([FAM_LABELS[f] for f in fams])
    ax2.set_yticklabels([FAM_LABELS[f] for f in fams])
    
    # Add correlation values
    for i in range(n_fams):
        for j in range(n_fams):
            if i >= j:  # Lower triangle and diagonal
                text = ax2.text(j, i, f'{fam_corr_mat[i, j]:.3f}', 
                              ha="center", va="center", fontweight='bold',
                              color='white' if abs(fam_corr_mat[i, j]) > 0.5 else 'black')
    
    ax2.set_title('Cross-Family Correlations')
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Mean Correlation')
    
    return fig

def save_fig(fig, name):
    """Save figure."""
    if not fig: return
    
    fig_dir = os.path.join(config.fig_dir, 'crosscorr_analysis')
    os.makedirs(fig_dir, exist_ok=True)
    
    fmt = config.plotting.get('savefig_format', 'png')
    path = os.path.join(fig_dir, f'{name}.{fmt}')
    
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"    Saved {name}")
    plt.close(fig)

def main():
    """Main analysis."""
    try:
        data = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Extract components
    comp_dict = {}
    for fam in ['all', 'human', 'monkey']:
        if fam in data['results']:
            comp_dict[fam] = data['results'][fam]['components']
            print(f"{FAM_LABELS[fam]}: {comp_dict[fam].shape}")
    
    print(f"\nComputing correlations...")
    corr_mat, fam_info = compute_corrs(comp_dict)
    print(f"Matrix: {corr_mat.shape}, Total: {sum(fam_info['n'].values())}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    save_fig(plot_heatmap(corr_mat, fam_info), 'heatmap')
    save_fig(plot_mds_pairwise(corr_mat, fam_info), 'mds_pairwise')
    
    print(f"\nComplete! Saved to {os.path.join(config.fig_dir, 'crosscorr_analysis')}")

if __name__ == "__main__":
    main() 
