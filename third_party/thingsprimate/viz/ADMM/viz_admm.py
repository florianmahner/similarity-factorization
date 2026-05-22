#!/usr/bin/env python3
"""
Visualization for ADMM Analysis Results
Creates figures showing CV scores and component visualizations using the same approach as CCA families.
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# Add project root to path before importing project modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model.feature.core.ridge_utils import ridge_cv_fold
from config.paths import config
from functions.plotting import narrow_figure, medium_figure, setup_style, style_axes, format_axes, large_figure, square_panel, wide_figure, add_panel_axes
from functions.viz_common import save_fig_std, plot_duplets_from_enc_scores, make_legend, load_enc_scores

# Border thickness for image mosaics
MOSAIC_BORDER_WIDTH = 4.0

# Components overview grid parameters
OVERVIEW_COLS = 30
OVERVIEW_IMG_SIZE = 100
OVERVIEW_ROW_SPACING = 15  # Pixels of white space between component rows
GUARD_COL_WIDTH = OVERVIEW_IMG_SIZE
GUARD_GAP = 12
GUARD_FOLDS = 5

# Toggle for per-component plots (not selected)
PLOT_ALL = False
FIRST_N_COMPONENTS = 20

# Component selections (1-indexed) per family for ADMM
SELECTED_COMPONENTS = {
    'all':    [1, 10, 20, 47],
    'human':  [13, 27, 28, 33],
    'monkey': [12, 42, 46, 53],
}

# Grid size for selected component image mosaics
SELECTED_ROWS = 2
SELECTED_COLS = 6
RSM_TOP_N = 200

# Top features count
TOP_K = 5

# Subsampling settings
TOP_PCT = 1.0  # Percentile for subsampling top images

# Duplet barplot styling (parallel to CCA)
BONE_SEM_RANGE = (0.25, 0.45)
BONE_VIS_RANGE = (0.65, 0.75)

def subsample_top_pct(vec, n_cells, pct=TOP_PCT, pos=True):
    n_top = max(1, int(len(vec) * pct / 100))
    sorted_idx = np.argsort(vec)[::-1] if pos else np.argsort(vec)
    top_cand = sorted_idx[:n_top]
    return np.random.choice(top_cand, size=min(n_cells, len(top_cand)), replace=False)

def load_admm_results():
    """Load comprehensive ADMM analysis results."""
    output_dir = os.path.join(config.results_dir, 'admm_results')
    results_file = os.path.join(output_dir, 'admm_all_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"ERROR: ADMM results not found at {results_file}")
        print("Please run CCA_ADMM.py first.")
        return None
    
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check which families are available
    available_families = list(data['results'].keys())
    all_families = ['all', 'human', 'monkey']
    missing_families = [f for f in all_families if f not in available_families]
    
    print(f"Available families: {available_families}")
    if missing_families:
        print(f"Missing families: {missing_families}")
        print("(These can be added by re-running CCA_ADMM.py)")
    
    return data

def plot_cv_scores(results, fig_dir):
    """Create narrow figures showing CV scores for each family following gridsearch style."""
    setup_style()
    
    # Color palette from config
    color_palette = {
        'all': config.plotting.get('shared_color', '#81B7B3'), 
        'human': config.plotting.get('human_color', '#7c5799'), 
        'monkey': config.plotting.get('monkey_color', '#bda855')
    }
    
    for family_name, result in results.items():
        cv_df = result['cv_results']
        grouped = cv_df.groupby('rank')['score'].agg(['mean', 'std']).reset_index()
        
        fig = narrow_figure()
        ax = fig.add_axes([.1, .1, .85, .78])
        
        # Get data
        rank_vals = grouped['rank'].values
        score_vals = grouped['mean'].values
        std_vals = grouped['std'].values
        
        # Set y-axis range with padding (±20% around min/max)
        y_min, y_max = (score_vals - std_vals).min(), (score_vals + std_vals).max()
        y_range = y_max - y_min
        y_padding = y_range * 0.2
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Plot the main line with error bars in black (like gridsearch)
        ax.errorbar(rank_vals, score_vals, yerr=std_vals,
                   marker='o', linestyle='-', color='black', linewidth=3, 
                   markersize=9, capsize=4, capthick=1.5, elinewidth=2, zorder=5)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('CV Score (MSE)')
        ax.set_title(f'{family_name.replace("_", " ").title()} ADMM')
        
        # Format x-axis with integer ticks and set range starting from 10
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_xlim(left=10)
        
        # Format y-axis to show appropriate precision
        def y_formatter(x, pos):
            if abs(x) < 0.01:
                return f'{x:.2e}'
            else:
                return f'{x:.3f}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # Mark best rank with family-specific color
        best_rank = result['best_rank']
        family_color = color_palette.get(family_name, 'steelblue')
        ax.axvline(best_rank, color=family_color, linestyle='--', alpha=0.8, 
                   linewidth=3.5, zorder=6)
        
        style_axes(ax)
        
        # Save to standardized path under admm_analysis/<family>/cv_scores
        save_fig_std('admm', family_name, 'cv_scores', fig)

def get_viz_rank(family_name, best_rank, available_ranks):
    """Get visualization rank from config or use CV best, ensuring it's available."""
    custom_ranks = config.hyperparameters.get('admm', {}).get('custom_ranks', {})
    target_rank = custom_ranks.get(family_name, best_rank)
    
    # Find closest available rank
    if target_rank in available_ranks:
        return target_rank
    else:
        closest_rank = min(available_ranks, key=lambda x: abs(x - target_rank))
        if target_rank != best_rank:
            print(f"  Custom rank {target_rank} not available, using closest: {closest_rank}")
        return closest_rank

def _square_and_resize(path, target):
    """Load → center-crop to a square → resize to `target`×`target` px."""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, upper = (w - side) // 2, (h - side) // 2
        im = im.crop((left, upper, left + side, upper + side))
        im = im.resize((target, target), Image.BILINEAR)
        
        # Add dark grey inward border
        im_arr = np.array(im)
        thick = max(2, target // 60)  # Scale border with image size
        border_color = (64, 64, 64)  # Dark grey
        
        # Top and bottom borders
        im_arr[:thick, :] = border_color
        im_arr[-thick:, :] = border_color
        
        # Left and right borders  
        im_arr[:, :thick] = border_color
        im_arr[:, -thick:] = border_color
        
        return Image.fromarray(im_arr)
    except:
        # Return gray square with border if image can't be loaded
        blank = np.full((target, target, 3), 128, np.uint8)
        thick = max(2, target // 60)
        border_color = (64, 64, 64)
        blank[:thick, :] = border_color
        blank[-thick:, :] = border_color
        blank[:, :thick] = border_color
        blank[:, -thick:] = border_color
        return Image.fromarray(blank)

def _build_mosaic(paths, rows, cols, px):
    """
    Return a single numpy array of shape (rows*px, cols*px, 3)
    that contains all images tiled left-to-right, top-to-bottom.
    Missing tiles get mid-grey.
    """
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [_square_and_resize(p, px) if p is not None else Image.fromarray(blank) 
             for p in paths]
    # Pad with blank tiles if we don't have enough images
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]

    grid = np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])
    return grid

def _border(ax, color='black', lw=3):
    """
    Add a rectangle that runs exactly around the axes (0‒1 in Axes coords).
    Because it lives in ax.transAxes it is immune to GridSpec rounding.
    """
    ax.add_patch(
        Rectangle((0, 0), 1, 1,
                  transform=ax.transAxes,  # <- axes-relative
                  facecolor='none',
                  edgecolor=color,
                  linewidth=lw,
                  clip_on=False)           # let the stroke stick out
    )

def _bone_shades(n: int, start: float, end: float):
    tvals = [0.5 * (start + end)] if n <= 1 else np.linspace(start, end, n)
    cols = []
    for t in tvals:
        r, g, b, _ = cm.bone(float(t))
        cols.append((r, g, b))
    return cols


def _guard_tile(val, color, width, height):
    v = float(np.clip(val, -0.2, 1.0)) if np.isfinite(val) else np.nan
    base = np.asarray(mpl.colors.to_rgb(color))
    white = np.ones(3)
    if not np.isfinite(v):
        rgb = np.ones(3) * 0.9
    elif v <= 0.0:
        t = min(1.0, -v)
        rgb = white * (1.0 - 0.35 * t)
    else:
        t = min(1.0, v)
        mix = 0.85 * t
        rgb = white * (1.0 - mix) + base * mix
    tile = (rgb * 255.0).astype(np.uint8)
    arr = np.tile(tile, (height, width, 1))
    b = max(1, width // 60)
    arr[:b, :, :] = 64
    arr[-b:, :, :] = 64
    arr[:, :b, :] = 64
    arr[:, -b:, :] = 64
    return arr


def _guard_text_color(val):
    if not np.isfinite(val):
        return 'black'
    return 'white' if val >= 0.55 else 'black'


def ridge_guard(W, cca_res, n_folds=GUARD_FOLDS):
    if cca_res is None:
        return None
    comps = cca_res.get('components') or {}
    views = sorted(comps)
    if not views:
        return None
    humans = [v for v in views if v.lower().startswith('human')]
    monkeys = [v for v in views if v.lower().startswith('monkey')]
    want = humans + monkeys
    if not want:
        return None
    mats = {v: np.asarray(comps[v], float) for v in want}
    n_comp = W.shape[1]
    n_jobs = int(config.analysis.get('n_jobs', 4))

    def _fit(idx):
        y = W[:, idx]
        return {v: ridge_cv_fold(mats[v], y, n_folds=n_folds)[0] for v in want}

    rows = Parallel(n_jobs=n_jobs)(delayed(_fit)(i) for i in range(n_comp))
    per_view = {v: np.array([row[v] for row in rows], float) for v in want}

    labels, colors, vals = [], [], []
    if humans:
        labels.append('Human')
        colors.append(config.plotting.get('human_color', '#7c5799'))
        vals.append(np.nanmean([per_view[v] for v in humans], axis=0))
    if monkeys:
        labels.append('Monkey')
        colors.append(config.plotting.get('monkey_color', '#bda855'))
        vals.append(np.nanmean([per_view[v] for v in monkeys], axis=0))
    if not vals:
        return None
    guard_vals = np.stack(vals, axis=1)
    return {
        'values': guard_vals,
        'labels': labels,
        'colors': colors,
        'per_view': per_view,
        'views': want,
    }


def _draw_mini_profile(ax, imp_row, std_row, names, groups,
                        feature_label_fontsize=8, ylim_max=None):
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])

    vis_shades = _bone_shades(3, *vis_range)
    beh_shades = _bone_shades(3, *beh_range)

    idx_vis_all = [i for i, g in enumerate(groups) if g == 'visual']
    idx_beh_all = [i for i, g in enumerate(groups) if g == 'behavioral']

    vals = imp_row
    idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:3]
    idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:3]

    vals_v = [float(vals[i]) for i in idx_v]
    vals_b = [float(vals[i]) for i in idx_b]
    stds_v = [float(std_row[i]) if std_row is not None else 0.0 for i in idx_v]
    stds_b = [float(std_row[i]) if std_row is not None else 0.0 for i in idx_b]
    labs_v = [names[i] for i in idx_v]
    labs_b = [names[i] for i in idx_b]

    heights = vals_v + vals_b
    yerrs = stds_v + stds_b
    colors = vis_shades[:len(vals_v)] + beh_shades[:len(vals_b)]

    xpos = np.arange(len(heights))
    ax.bar(xpos, heights, color=colors, edgecolor='black', linewidth=1.0, zorder=2)
    for x, h, ye in zip(xpos, heights, yerrs):
        if ye and ye > 0:
            ax.errorbar(x, h, yerr=ye, fmt='none', ecolor='black', elinewidth=1.0, capsize=0, zorder=3)

    if ylim_max is None:
        ymax = max([h + ye for h, ye in zip(heights, yerrs)] or [1.0]) * 1.35
    else:
        ymax = float(ylim_max)
    ax.set_ylim(0, ymax)
    ax.margins(y=0)
    ax.set_xticks([])
    ax.set_ylabel(r'$\Delta R^2$', fontsize=9)

    # Labels above bars
    yrange = ymax
    base_pad = (max(heights) if heights else 1.0) * 0.01
    extra_gap = yrange * 0.01
    short = []
    for name in (labs_v + labs_b):
        lab = name
        for ch in ['-', '/', ' ']:
            if ch in lab:
                lab = lab.split(ch)[0]
                break
        short.append(lab.replace('_', ' '))
    for x, h, ye, lab in zip(xpos, heights, yerrs, short):
        ax.text(x, h + ye + base_pad + extra_gap, lab, rotation=90, ha='center', va='bottom', fontsize=feature_label_fontsize)
    style_axes(ax); format_axes(ax)

def _build_positive_mosaic_for_comp_admm(W, ci, stims, cats, img_dir, rows=2, cols=6, img_px=180):
    weights = W[:, ci]
    n_cells = rows * cols
    ipos = subsample_top_pct(weights, n_cells)
    paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    return _build_mosaic(paths, rows, cols, img_px)

def plot_top_components_profiles_and_grids_admm(fam_key, W, stims, cats, imp_mean, imp_std, names, groups,
                                                img_dir, top_k=5, rows=2, cols=6, img_px=180):
    # Rank by maximum ΔR² over all features
    scores = np.max(imp_mean, axis=1)
    order = np.argsort(scores)[::-1]
    chosen = order[:min(top_k, imp_mean.shape[0])]

    # Shared y-limit for mini profiles
    ymax = 0.0
    for ci in chosen:
        vals = imp_mean[ci]
        stds = imp_std[ci] if imp_std is not None else np.zeros_like(vals)
        idx_vis_all = [i for i, g in enumerate(groups) if g == 'visual']
        idx_beh_all = [i for i, g in enumerate(groups) if g == 'behavioral']
        idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:3]
        idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:3]
        local_max = 0.0
        for j in (idx_v + idx_b):
            local_max = max(local_max, float(vals[j] + (stds[j] if stds is not None else 0.0)))
        ymax = max(ymax, local_max)
    ymax = ymax * 1.35 if ymax > 0 else 1.0

    fig_h = max(4, 1.8 * len(chosen))
    fig_w = 10
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = GridSpec(len(chosen), 2, width_ratios=[1.0, 3.0], hspace=0.4, wspace=0.25, figure=fig)

    for r, ci in enumerate(chosen):
        ax_prof = fig.add_subplot(gs[r, 0])
        _draw_mini_profile(ax_prof, imp_mean[ci], imp_std[ci] if imp_std is not None else None,
                           names, groups, feature_label_fontsize=8, ylim_max=ymax)
        ax_prof.set_title(f'C{ci+1} profile', fontsize=10)

        ax_grid = fig.add_subplot(gs[r, 1])
        mosaic = _build_positive_mosaic_for_comp_admm(W, ci, stims, cats, img_dir, rows=rows, cols=cols, img_px=img_px)
        ax_grid.imshow(mosaic, interpolation='nearest', aspect='auto')
        ax_grid.axis('off')
        _border(ax_grid, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)

    fig.suptitle(f"{fam_key.upper()} ADMM – Top {len(chosen)} Components by Max ΔR²", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, [(int(ci)+1, float(scores[ci])) for ci in chosen]

def plot_duplets_from_joint(imp_mean: np.ndarray, names: list, groups: list,
                             comp_indices: list, std: np.ndarray = None,
                             feature_label_fontsize: float = 11,
                             extra_label_gap_frac: float = 0.01):
    """
    Grouped bars from joint matrix with visual/behavioral/semantic tags.
    - Bars are CENTER-aligned; error bars and labels use the same center x.
    - No whitespace at the x-axis (ax.margins(y=0)).
    - Tight layout to avoid bleed.
    - Feature labels above bars use a slightly larger fontsize (only these labels).
    - A tiny extra gap between error bar and label (extra_label_gap_frac of y-range).
    """
    if imp_mean is None or names is None or groups is None or not comp_indices:
        return None

    # Load color ranges from config
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])
    sem_range = config.plotting.get('semantic_range', [0.2, 0.22])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    bar_width = 0.5
    gap_between_groups = 0.6
    gap_within_groups = 0.0  # No gap between v/b/s pairs
    bars_per_group = 6  # 2 per group × 3 groups
    positions, heights, colors, labels, group_centers, yerrs = [], [], [], [], [], []

    idx_vis_all = [i for i,g in enumerate(groups) if g == 'visual']
    idx_beh_all = [i for i,g in enumerate(groups) if g == 'behavioral']
    idx_sem_all = [i for i,g in enumerate(groups) if g == 'semantic']

    # Load behavioral dimension names for proper labeling
    try:
        from functions.viz_common import load_behav_dim_names
        beh_dim_names = load_behav_dim_names()
    except:
        beh_dim_names = []

    for gi, ci in enumerate(comp_indices):
        vals = imp_mean[ci]
        idx_v = sorted(idx_vis_all, key=lambda i: vals[i], reverse=True)[:2]
        idx_b = sorted(idx_beh_all, key=lambda i: vals[i], reverse=True)[:2]
        idx_s = sorted(idx_sem_all, key=lambda i: vals[i], reverse=True)[:2]
        names_v = [names[j] for j in idx_v]
        names_b = [names[j] for j in idx_b]
        names_s = [names[j] for j in idx_s]
        vals_v = vals[idx_v] if len(idx_v)>0 else []
        vals_b = vals[idx_b] if len(idx_b)>0 else []
        vals_s = vals[idx_s] if len(idx_s)>0 else []
        stds_v = std[ci][idx_v] if (std is not None and len(idx_v)>0) else []
        stds_b = std[ci][idx_b] if (std is not None and len(idx_b)>0) else []
        stds_s = std[ci][idx_s] if (std is not None and len(idx_s)>0) else []

        group_start = gi * (bars_per_group * bar_width + gap_between_groups)

        # Visual bars
        for k in range(2):
            positions.append(group_start + k * bar_width)
            heights.append(float(vals_v[k]) if k < len(vals_v) else 0.0)
            colors.append(vis_shades[k])
            labels.append(names_v[k] if k < len(names_v) else '')
            yerrs.append(float(stds_v[k]) if (k < len(stds_v)) else 0.0)

        # Behavioral bars (with gap after visual)
        beh_start = group_start + 2 * bar_width + gap_within_groups
        for k in range(2):
            positions.append(beh_start + k * bar_width)
            heights.append(float(vals_b[k]) if k < len(vals_b) else 0.0)
            colors.append(beh_shades[k])
            # Use actual behavioral dimension names if available
            if k < len(names_b) and beh_dim_names and names_b[k].startswith('SPOSE'):
                try:
                    idx = int(names_b[k].split()[-1]) - 1
                    if 0 <= idx < len(beh_dim_names):
                        beh_name = beh_dim_names[idx]
                    else:
                        beh_name = names_b[k]
                except:
                    beh_name = names_b[k]
            else:
                beh_name = names_b[k]
            labels.append(beh_name)
            yerrs.append(float(stds_b[k]) if (k < len(stds_b)) else 0.0)

        # Semantic bars (with gap after behavioral)
        sem_start = beh_start + 2 * bar_width + gap_within_groups
        for k in range(2):
            positions.append(sem_start + k * bar_width)
            heights.append(float(vals_s[k]) if k < len(vals_s) else 0.0)
            colors.append(sem_shades[k])
            labels.append(names_s[k] if k < len(names_s) else '')
            yerrs.append(float(stds_s[k]) if (k < len(stds_s)) else 0.0)

        group_centers.append(group_start + (bars_per_group - 1) * bar_width / 2)

    fig = wide_figure()
    ax = fig.add_subplot(111)

    # Bars centered at 'positions'
    ax.bar(positions, heights, width=bar_width, color=colors,
           edgecolor='black', linewidth=1.2, align='center', zorder=2)

    # Error bars centered on the same x
    for x, h, ye in zip(positions, heights, yerrs):
        if ye and ye > 0:
            ax.errorbar(x, h, yerr=ye, fmt='none', ecolor='black',
                        elinewidth=1.0, capsize=0, zorder=3)

    # Determine y headroom and padding
    ymax_data = max([h + ye for h, ye in zip(heights, yerrs)] or [1.0])
    ax.set_ylim(0, ymax_data * 1.60)
    ax.margins(y=0)

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    base_pad = (max(heights) if heights else 1.0) * 0.012
    extra_gap = yrange * float(extra_label_gap_frac)

    # Labels centered above the bars (slightly larger fontsize)
    for x, h, name, ye in zip(positions, heights, labels, yerrs):
        if name:
            lab = name
            for ch in ['-', '/', ' ']:
                if ch in lab:
                    lab = lab.split(ch)[0]
                    break
            ax.text(x, h + ye + base_pad + extra_gap, lab.replace('_',' '), rotation=90,
                    ha='center', va='bottom', fontsize=feature_label_fontsize, clip_on=False)

    ax.set_ylabel(r'$\Delta R^2$')
    style_axes(ax); format_axes(ax, precision=3)

    # Component numbering under groups
    ax.set_xticks(group_centers)
    comp_labels = [str(ci+1) for ci in comp_indices]
    ax.set_xticklabels(comp_labels)
    ax.set_xlabel('Component')

    for label in ax.get_xticklabels():
        label.set_fontstyle('normal')
        label.set_fontweight('normal')

    plt.tight_layout(pad=0.6)
    return fig

def make_legend():
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])
    sem_range = config.plotting.get('semantic_range', [0.2, 0.22])

    vis_shades = _bone_shades(2, *vis_range)
    beh_shades = _bone_shades(2, *beh_range)
    sem_shades = _bone_shades(2, *sem_range)

    fig = narrow_figure()
    ax = fig.add_subplot(111)
    handles = [
        plt.Rectangle((0,0),1,1, color=vis_shades[0], ec='black', lw=1.0, label='Visual'),
        plt.Rectangle((0,0),1,1, color=beh_shades[0], ec='black', lw=1.0, label='Behavioral'),
        plt.Rectangle((0,0),1,1, color=sem_shades[0], ec='black', lw=1.0, label='Semantic'),
    ]
    ax.axis('off')
    _ = ax.legend(handles=handles, frameon=False, loc='center')
    plt.tight_layout(pad=0.2)
    return fig

def _plot_distribution_positive_only(ax, vec, order, ipos, inv, pos_cand=None):
    """Draw positive-only density curve and brackets for ADMM components."""
    vals = vec[order]
    total = len(vals)
    
    def _smooth(y):
        return gaussian_filter1d(y, sigma=len(y) / 50) if len(y) > 0 else y
    
    # Only positive side since ADMM components are non-negative
    x = np.linspace(0, total - 1, 500)
    y = _smooth(np.interp(x, np.arange(total), vals))
    if y.max() > 0:
        y *= (vals.max() or 1) / y.max()
    
    for i in range(len(x) - 1):
        frac = x[i] / total
        ax.fill_between(x[i:i + 2], 0, y[i:i + 2], color=cm.Reds(0.3 + 0.7 * frac), lw=0)
    
    ax.axhline(0, color='k', lw=0)
    ax.set_xlim(0, total)
    y0, y1 = ax.get_ylim()
    dy = y1 - y0
    ax.set_ylim(y0 - 0.05 * dy, y1 + 0.05 * dy)
    for sp in ax.spines.values():
        sp.set_visible(False)
    
    # Brackets & arrows for positive selection
    if pos_cand is not None:
        xp_all = inv[pos_cand]
    else:
        xp_all = inv[ipos]
    
    xp0, xp1 = xp_all.min(), xp_all.max()
    
    # Update y-limits after adjusting
    y0, y1 = ax.get_ylim()
    
    # Get positive color from config
    pos_color = config.plotting.get('positive_color', '#8B0000')
    
    ax.plot([xp0, xp1], [0, 0], color=pos_color, lw=4)
    ax.plot([xp0, xp0], [0, y1], color=pos_color, lw=4)
    ax.plot([xp1, xp1], [0, y1], color=pos_color, lw=4)

def load_img_simple(path, size):
    """Load and prep single image for overview."""
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)
        
        # Add border
        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(1, size // 60)
        arr[:bw, :] = b_color
        arr[-bw:, :] = b_color
        arr[:, :bw] = b_color
        arr[:, -bw:] = b_color
        
        return Image.fromarray(arr)
    except:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)

def plot_components_overview(W, all_stims, cats, img_dir, save_path, family_name, viz_rank,
                             cols=OVERVIEW_COLS, img_size=OVERVIEW_IMG_SIZE,
                             imp_mean=None, feature_names=None, feature_groups=None,
                             guard=None):
    """ADMM overview: left feature bars + labels (if available) and right image rows (positive-only)."""
    n_comps = W.shape[1]

    guard_vals = guard.get('values') if guard else None
    guard_labels = guard.get('labels') if guard else []
    guard_colors = guard.get('colors') if guard else []
    guard_cols = guard_vals.shape[1] if isinstance(guard_vals, np.ndarray) else 0
    guard_w = guard_cols * GUARD_COL_WIDTH + (GUARD_GAP if guard_cols else 0)
    guard_centers = [GUARD_COL_WIDTH * (i + 0.5) for i in range(guard_cols)]
    guard_gap_tile = np.full((img_size, GUARD_GAP, 3), 255, np.uint8) if guard_cols else None

    # Build image rows (positive-only)
    all_rows = []
    for comp_idx in range(n_comps):
        w = W[:, comp_idx]
        top_idx = subsample_top_pct(w, cols)
        row_tiles = []
        if guard_cols:
            gvals = guard_vals[comp_idx]
            for gval, col in zip(gvals, guard_colors):
                row_tiles.append(_guard_tile(gval, col, GUARD_COL_WIDTH, img_size))
            if guard_gap_tile is not None:
                row_tiles.append(guard_gap_tile.copy())
        for idx in top_idx:
            stim = all_stims[idx]
            cat = cats[idx]
            path = os.path.join(img_dir, cat, f"{stim}.jpg")
            row_tiles.append(np.array(load_img_simple(path, img_size)))
        row = np.hstack(row_tiles)
        all_rows.append(row)
        if comp_idx < n_comps - 1:
            all_rows.append(np.full((OVERVIEW_ROW_SPACING, row.shape[1], 3), 255, np.uint8))

    grid = np.vstack(all_rows)
    H_img, W_img = grid.shape[:2]

    # No importance: compose grid and overlay component numbers
    if imp_mean is None or feature_names is None or feature_groups is None:
        canvas = grid
        guard_x0 = 0
        mosaic_x0 = guard_w
        fig_w = canvas.shape[1] / 100.0
        fig_h = canvas.shape[0] / 100.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(canvas); ax.axis('off')
        if guard_cols:
            hdr_y = -max(8, GUARD_GAP)
            for gi, lab in enumerate(guard_labels):
                ax.text(guard_x0 + guard_centers[gi], hdr_y, lab, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
        for i in range(n_comps):
            y0 = i * (img_size + OVERVIEW_ROW_SPACING)
            if guard_cols:
                for gi, val in enumerate(guard_vals[i]):
                    txt = '--' if not np.isfinite(val) else f"{val:.2f}"
                    ax.text(guard_x0 + guard_centers[gi], y0 + img_size * 0.5, txt,
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            color=_guard_text_color(val))
            ax.text(mosaic_x0 + 8, y0 + 14, f'C{i+1}', ha='left', va='top', fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6))
        plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close(); return

    # Build small panels per component (bars + two-column labels)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg
    import matplotlib.pyplot as _plt

    def _short(name):
        lab = name
        for ch in ['-', '/', ' ', '_']:
            if ch in lab:
                lab = lab.split(ch)[0]
                break
        return lab

    bar_panel_height = img_size
    BAR_PANEL_WIDTH_PX = 240
    BAR_GAP_PX = 30
    vis_range = config.plotting.get('visual_range', [0.9, 0.92])
    beh_range = config.plotting.get('behavioral_range', [0.6, 0.62])

    vis_shades = [cm.bone(t) for t in np.linspace(vis_range[0], vis_range[1], 3)]
    beh_shades = [cm.bone(t) for t in np.linspace(beh_range[0], beh_range[1], 3)]

    panels = []
    n_pan = min(n_comps, imp_mean.shape[0])
    for comp_idx in range(n_pan):
        vals = np.array(imp_mean[comp_idx], dtype=float)
        idx_vis = [i for i, g in enumerate(feature_groups) if g == 'visual']
        idx_beh = [i for i, g in enumerate(feature_groups) if g == 'behavioral']
        top_vis = sorted(idx_vis, key=lambda i: vals[i], reverse=True)[:3]
        top_beh = sorted(idx_beh, key=lambda i: vals[i], reverse=True)[:3]
        bar_vals = [vals[i] for i in top_vis] + [vals[i] for i in top_beh]
        bar_cols = [vis_shades[k][:3] for k in range(len(top_vis))] + [beh_shades[k][:3] for k in range(len(top_beh))]
        vis_labels = [feature_names[i] for i in top_vis]
        beh_labels = [feature_names[i] for i in top_beh]

        fig_w_in = BAR_PANEL_WIDTH_PX / 100.0
        fig_h_in = bar_panel_height / 100.0
        _fig = _plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)
        bar_ax = _fig.add_axes([0.08, 0.15, 0.50, 0.70])
        txt_ax = _fig.add_axes([0.63, 0.15, 0.34, 0.70])
        x = np.arange(len(bar_vals))
        bar_ax.bar(x, bar_vals, color=bar_cols, edgecolor='black', linewidth=0.8)
        bar_ax.set_xticks([])
        bar_ax.tick_params(axis='y', labelsize=6, length=3)
        bar_ax.set_ylabel(r'$\Delta R^2$', fontsize=7)
        for sp in ['top', 'right']:
            bar_ax.spines[sp].set_visible(False)
        ymax = max(bar_vals) * 1.15 if len(bar_vals) and max(bar_vals) > 0 else 1.0
        bar_ax.set_ylim(0, ymax)
        bar_ax.margins(x=0)

        txt_ax.axis('off')
        txt_ax.text(0.00, 0.97, 'Visual', fontsize=8, fontweight='bold', va='top')
        txt_ax.text(0.55, 0.97, 'Behav', fontsize=8, fontweight='bold', va='top')
        for r in range(max(len(vis_labels), len(beh_labels))):
            y = 0.90 - r * 0.35
            if r < len(vis_labels):
                txt_ax.text(0.00, y, _short(vis_labels[r]), fontsize=8, fontweight='bold', va='top')
            if r < len(beh_labels):
                txt_ax.text(0.55, y, _short(beh_labels[r]), fontsize=8, fontweight='bold', va='top')

        canvas = _FigureCanvasAgg(_fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = canvas.get_width_height()
        panel = buf.reshape(h, w, 3)
        _plt.close(_fig)
        if panel.shape[0] != bar_panel_height or panel.shape[1] != BAR_PANEL_WIDTH_PX:
            panel = np.array(Image.fromarray(panel).resize((BAR_PANEL_WIDTH_PX, bar_panel_height), Image.BILINEAR))
        panels.append(panel)

    total_w = BAR_PANEL_WIDTH_PX + BAR_GAP_PX + W_img
    canvas = np.full((H_img, total_w, 3), 255, np.uint8)
    for comp_idx in range(n_comps):
        y0 = comp_idx * (img_size + OVERVIEW_ROW_SPACING)
        if comp_idx < len(panels):
            canvas[y0:y0 + img_size, :BAR_PANEL_WIDTH_PX] = panels[comp_idx]
    x_cursor = BAR_PANEL_WIDTH_PX + BAR_GAP_PX
    canvas[:, x_cursor:x_cursor + W_img] = grid

    fig_w = total_w / 100.0
    fig_h = H_img / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(canvas); ax.axis('off')
    guard_x0 = x_cursor
    mosaic_x0 = x_cursor + guard_w
    if guard_cols:
        hdr_y = -max(8, GUARD_GAP)
        for gi, lab in enumerate(guard_labels):
            ax.text(guard_x0 + guard_centers[gi], hdr_y, lab, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    for i in range(n_comps):
        y0 = i * (img_size + OVERVIEW_ROW_SPACING)
        if guard_cols:
            for gi, val in enumerate(guard_vals[i]):
                txt = '--' if not np.isfinite(val) else f"{val:.2f}"
                ax.text(guard_x0 + guard_centers[gi], y0 + img_size * 0.5, txt,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color=_guard_text_color(val))
        ax.text(mosaic_x0 + 8, y0 + 14, f'C{i+1}', ha='left', va='top', fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', edgecolor='none', alpha=0.6))
    plt.tight_layout(); plt.savefig(save_path, dpi=180, bbox_inches='tight'); plt.close()

def plot_comp_imgs_positive_only(comp_idx, W, stims, cats, img_dir=None, title=None, rows=4, cols=6, 
                                img_px=180, dpi=300, subsample=True, sub_n=100, random_seed=42):
    """Plot ADMM component images with positive-only grid (no negative box)"""
    if img_dir is None:
        img_dir = os.path.join(config.data_dir, 'things', 'images')
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    vec = W[:, comp_idx]
    n_cells = rows * cols
    order = np.argsort(vec)

    # Choose only positive examples (ADMM components are non-negative)
    if subsample:
        n_total = len(vec)
        n_cand = min(sub_n, n_total // 2)
        pos_cand = order[-n_cand:]
        ipos = np.random.choice(pos_cand, size=min(n_cells, len(pos_cand)), replace=False)
    else:
        ipos = subsample_top_pct(vec, n_cells)
    
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    if subsample:
        xp_all = inv[pos_cand]
    else:
        xp_all = inv[ipos]

    # Figure with only positive grid and distribution (no negative)
    fw = cols * img_px / dpi  # Keep thumbnails square
    fh = (rows + 0.5) * img_px / dpi  # Only positive grid + distribution
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, constrained_layout=False)
    gs_master = GridSpec(
        2, 1,
        height_ratios=[rows, 0.5],  # positive grid – distribution
        hspace=0,
        figure=fig
    )

    # 1) Positive grid (top) - build mosaic
    pos_paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    pos_mosaic = _build_mosaic(pos_paths, rows, cols, img_px)
    ax_top = fig.add_subplot(gs_master[0])
    ax_top.imshow(pos_mosaic, interpolation='nearest', aspect='auto')
    ax_top.axis('off')
    ax_top.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_top.set_ylim(pos_mosaic.shape[0] - 0.5, -0.5)
    _border(ax_top, color=config.plotting.get('positive_color', '#8B0000'), 
            lw=MOSAIC_BORDER_WIDTH)

    # 2) Distribution axis (bottom)
    ax_center = fig.add_subplot(gs_master[1])
    ax_center.set_xlim(-0.5, pos_mosaic.shape[1] - 0.5)
    ax_center.margins(x=0, y=0)  # no padding left/right or top/bottom
    _plot_distribution_positive_only(ax_center, vec, order, ipos, inv, 
                                   pos_cand if subsample else None)
    ax_center.set_yticks([])
    ax_center.set_xticks([])

    if title:
        fig.suptitle(f"{title} – Component {comp_idx+1}", fontsize=14, y=0.99)
    
    return fig

def save_admm_fig(fig, fig_type, family_name, comp_idx=None, extra_info=""):
    """Save ADMM analysis figure with structured path and name in admm_analysis folder."""
    if not fig:
        return
        
    # Save in admm_analysis subfolder of figures directory
    fig_dir = os.path.join(config.fig_dir, 'admm_analysis', family_name, fig_type)
    os.makedirs(fig_dir, exist_ok=True)
    
    fmt = config.plotting.get('savefig_format', 'pdf')
    
    # Special-case duplets to mirror CCA naming
    if fig_type == 'duplets' and not extra_info and comp_idx is None:
        fname = 'bar_duplet.pdf'
        fmt = 'pdf'
    elif comp_idx is not None:
        fname = f'comp_{comp_idx+1}.{fmt}'
    elif extra_info:
        fname = f'{extra_info}.{fmt}'
    else:
        fname = f'{fig_type}.{fmt}'

    fig_path = os.path.join(fig_dir, fname)
    
    # Use higher DPI for image plots
    save_dpi = 600 if fig_type == 'images' else 300
    fig.savefig(fig_path, bbox_inches='tight', dpi=save_dpi)
    print(f"    Saved to {fig_path}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# RSM / RDM visualizations

def _load_cca_state():
    """Load the CCA state used to build RSMs (robust to two common locations)."""
    # Prefer explicit cache_file if present
    state_fp = getattr(config, 'cache_file', None)
    if state_fp is None or not os.path.exists(state_fp):
        # Fallback to legacy path used in viz_cca_families
        maybe_fp = os.path.join(config.results_dir, 'cca_state.pkl')
        state_fp = maybe_fp if os.path.exists(maybe_fp) else None
    if not state_fp:
        print("ERROR: Could not locate CCA state file (config.cache_file or results/cca_state.pkl)")
        return None
    try:
        with open(state_fp, 'rb') as f:
            s = pickle.load(f)
        return s
    except Exception as e:
        print(f"ERROR: Failed to read CCA state from {state_fp}: {e}")
        return None

def _compute_mean_rsm(cca_results, fisher=True, view_weights=None):
    """Match CCA_ADMM.compute_mean_rsm: Fisher-z average of per-view cosine RSMs, rescaled to [0,1]."""
    from sklearn.metrics.pairwise import cosine_similarity
    views = sorted(cca_results['components'])
    sims = []
    for v in views:
        comps = cca_results['components'][v]
        sim = cosine_similarity(comps)
        sims.append(sim)
    sims = np.stack(sims)

    if view_weights is None:
        view_weights = np.ones(len(views)) / len(views)
    else:
        view_weights = np.asarray(view_weights) / np.sum(view_weights)

    if fisher:
        sims_clipped = np.clip(sims, -0.9999, 0.9999)
        sims_z = np.arctanh(sims_clipped)
        mean_z = (view_weights[:, None, None] * sims_z).sum(axis=0)
        mean_sim = np.tanh(mean_z)
    else:
        mean_sim = (view_weights[:, None, None] * sims).sum(axis=0)

    # Rescale to [0,1]
    mn, mx = mean_sim.min(), mean_sim.max()
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    mean_sim = (mean_sim - mn) / denom
    return mean_sim

def _plot_matrix(mat, title=None, cmap='magma', vmin=0.0, vmax=1.0, *, bare=False, outline=True, top_n=None):
    """Plot a square matrix with magma colormap and compact styling.

    bare=True suppresses title, axis labels, and colorbar (useful for insets).
    outline=True draws a black rectangle around the matrix area.
    top_n: if provided, crops to the top-left top_n×top_n submatrix.
    """
    if top_n is not None and top_n > 0:
        mat = mat[:top_n, :top_n]
    fig, ax = square_panel()
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if not bare:
        if title:
            ax.set_title(title)
        ax.set_xlabel('Stimuli')
        ax.set_ylabel('Stimuli')
    # Hide tick labels for compactness
    ax.set_xticks([]); ax.set_yticks([])
    style_axes(ax)
    # Colorbar (thin) unless bare
    if not bare:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(length=3)
    # Black outline around matrix area
    if outline:
        h, w = mat.shape[:2]
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor='black', linewidth=4.0, zorder=5))
    return fig


# ──────────────────────────────────────────────────────────────────────
# RSM / RDM visualizations

def visualize_rsms(results):
    """Compute and save the RSM (only) for each family, matching ADMM settings.

    We output only the RSM because ADMM operates on a similarity matrix.
    """
    setup_style()
    s = _load_cca_state()
    if s is None:
        print("Skipping RSM/RDM plots: no CCA state available")
        return

    fisher_z = config.hyperparameters.get('admm', {}).get('use_fisher_z', True)

    fam_map = {
        'all': s.get('cca_all'),
        'human': s.get('cca_hum'),
        'monkey': s.get('cca_mon'),
    }

    for fam_key in results.keys():
        cca_res = fam_map.get(fam_key)
        if cca_res is None:
            print(f"RSM: missing CCA results for family '{fam_key}', skipping")
            continue

        mean_rsm = _compute_mean_rsm(cca_res, fisher=fisher_z)

        # RSM figure (bare inset style, cropped to first RSM_TOP_N entries)
        fig_rsm = _plot_matrix(mean_rsm, title=None, cmap='magma', vmin=0.0, vmax=1.0,
                               bare=True, outline=True, top_n=RSM_TOP_N)
        save_fig_std('admm', fam_key, 'rsm', fig_rsm, extra=f"{fam_key}_RSM")

def visualize_selected_components(results, all_stims, fig_dir,
                                  selection_map, rows=4, cols=6):
    """Visualize user-selected components (positive-only) for each family.

    selection_map: dict with keys in {'all','human','monkey'} and values as 1-indexed lists.
    rows, cols: grid size for the positive-only mosaics.
    """
    cats = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in all_stims]
    img_dir = os.path.join(config.data_dir, 'things', 'images')

    for fam_key, result in results.items():
        # Normalize family key to our selection keys
        sel_key = fam_key  # expected to already be 'all' | 'human' | 'monkey'
        chosen = selection_map.get(sel_key, []) or []
        if not chosen:
            continue

        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(fam_key, result['best_rank'], available_ranks)
        W = result['components'][viz_rank]

        print(f"\nSelected components for {fam_key} (rank {viz_rank}): {chosen}")

        # Save selected component grids
        for comp_id in chosen:
            # Convert to 0-indexed; skip invalid
            ci = int(comp_id) - 1
            if ci < 0 or ci >= W.shape[1]:
                print(f"  Warning: component {comp_id} out of range (1..{W.shape[1]}), skipping")
                continue

            title = f"{fam_key.upper()} Component {comp_id} (rank {viz_rank})"
            fig = plot_comp_imgs_positive_only(
                ci, W, all_stims, cats,
                img_dir=img_dir,
                title=title,
                rows=rows, cols=cols,
                img_px=360, dpi=300
            )
            # Save into a dedicated 'selected' folder
            save_fig_std('admm', fam_key, 'selected', fig, comp_idx=ci, image_high_dpi=True)

def visualize_components(results, all_stims, fig_dir, cca_state=None):
    """Visualize ADMM components using selected ranks."""
    cats = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in all_stims]
    cca_all = cca_state.get('cca_all') if isinstance(cca_state, dict) else None

    for family_name, result in results.items():
        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(family_name, result['best_rank'], available_ranks)
        W = result['components'][viz_rank]

        custom_ranks = config.hyperparameters.get('admm', {}).get('custom_ranks', {})
        is_custom = family_name in custom_ranks
        rank_info = f"custom rank {viz_rank}" if is_custom else f"best rank {viz_rank}"
        
        print(f"Visualizing {family_name} components ({rank_info})")
        
        # Create components overview
        print(f"  Creating components overview (guard-enhanced where applicable)...")
        img_dir = os.path.join(config.data_dir, 'things', 'images')
        overview_path = os.path.join(fig_dir, family_name, 'overview', 'components_overview.png')
        os.makedirs(os.path.dirname(overview_path), exist_ok=True)
        guard_data = None
        if family_name == 'all' and cca_all is not None:
            print("    Computing ridge guard (cross-species views)...")
            guard_data = ridge_guard(W, cca_all)
            if guard_data:
                guard_dir = os.path.dirname(overview_path)
                df_guard = pd.DataFrame({'Component': np.arange(1, guard_data['values'].shape[0] + 1)})
                for gi, lab in enumerate(guard_data['labels']):
                    df_guard[f"{lab}_R2"] = guard_data['values'][:, gi]
                guard_csv = os.path.join(guard_dir, 'ridge_guard.csv')
                df_guard.to_csv(guard_csv, index=False)
                mean_vals = [f"{lab}:{np.nanmean(guard_data['values'][:, gi]):.3f}" for gi, lab in enumerate(guard_data['labels'])]
                print(f"      Guard means {' | '.join(mean_vals)}")
                print(f"      Guard summary saved to {guard_csv}")
        need_overview = guard_data is not None or not os.path.exists(overview_path)
        if need_overview:
            plot_components_overview(W, all_stims, cats, img_dir, overview_path, family_name, viz_rank, guard=guard_data)
            print(f"    Saved overview to {overview_path}")
        else:
            print(f"    Overview exists, skipping: {overview_path}")
        
        # Individual component plots
        if PLOT_ALL:
            n_viz = min(FIRST_N_COMPONENTS, W.shape[1])
            for comp_idx in range(n_viz):
                print(f"  Component {comp_idx+1}: Generating image plot...")
                fig_images = plot_comp_imgs_positive_only(
                    comp_idx, W, all_stims, cats,
                    img_dir=img_dir,
                    title=f"{family_name.upper()} Component {comp_idx+1} (rank {viz_rank})",
                    img_px=360, dpi=300
                )
                save_fig_std('admm', family_name, 'images', fig_images, comp_idx=comp_idx, image_high_dpi=True)
            print(f"  Saved {n_viz} component plots for {family_name}")
        else:
            print(f"  Skipping 'plot all' individual component plots (PLOT_ALL=False)")

def create_rank_summary(results, fig_dir):
    """Create rank selection summary table."""
    table_data = []
    custom_ranks = config.hyperparameters.get('admm', {}).get('custom_ranks', {})
    
    for family_name, result in results.items():
        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(family_name, result['best_rank'], available_ranks)
        is_custom = family_name in custom_ranks
        
        table_data.append({
            'Family': family_name.upper(),
            'CV Best Rank': result['best_rank'],
            'Viz Rank': viz_rank,
            'Custom': 'Yes' if is_custom else 'No',
            'CV Score': f"{result['best_score']:.4f}"
        })
    
    df = pd.DataFrame(table_data)
    
    summary_dir = os.path.join(fig_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, 'rank_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print("RANK SELECTION SUMMARY")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    print(f"Table saved: {csv_path}")

def main():
    """Main visualization function."""
    print("Loading ADMM results...")
    data = load_admm_results()
    if data is None: return
    
    results = data['results']
    all_stims = data['all_stims']
    
    # Check available families
    available_families = list(results.keys())
    if not available_families:
        print("ERROR: No families found in results.")
        return
    
    fig_dir = os.path.join(config.fig_dir, 'admm_analysis')
    if os.path.exists(fig_dir):
        import shutil
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"Creating visualizations in {fig_dir}...")
    print(f"Processing {len(available_families)} families: {available_families}")
    
    print("\nStep 1: CV scores (full rank range)")
    plot_cv_scores(results, fig_dir)
    
    print("\nStep 2: Component visualization (selected ranks)")
    cca_state = _load_cca_state()
    visualize_components(results, all_stims, fig_dir, cca_state=cca_state)

    # Selected components (positive-only) if not plotting all
    if not PLOT_ALL:
        print("\nStep 3: Selected components (positive-only)")
        visualize_selected_components(
            results, all_stims, fig_dir,
            selection_map=SELECTED_COMPONENTS,
            rows=SELECTED_ROWS, cols=SELECTED_COLS
        )
    else:
        print("\nStep 3: Skipping selected components (PLOT_ALL=True)")

    # RSM used for analyses
    # Duplets (ΔR²) for selected components, per family
    print("\nStep 4: Duplets (ΔR²) for selected components")
    for fam_key, result in results.items():
        sel = SELECTED_COMPONENTS.get(fam_key, []) or []
        if not sel:
            print(f"  {fam_key}: no selected components, skipping duplets")
            continue

        # Skip: encoding scores loaded by plot_duplets_from_enc_scores
        if imp_mean is None or names is None or groups is None:
            print(f"  {fam_key}: incomplete importance payload, skipping duplets")
            continue

        # Convert selection to 0-indexed and filter to valid range
        zero_idx = [int(c)-1 for c in sel if isinstance(c, (int, np.integer)) and (1 <= int(c) <= imp_mean.shape[0])]
        if not zero_idx:
            print(f"  {fam_key}: no valid selected components within 1..{imp_mean.shape[0]}, skipping duplets")
            continue

        print(f"  {fam_key}: duplets for components {sel} (valid: {[z+1 for z in zero_idx]})")
        sel = SELECTED_COMPONENTS.get(fam_key, [1,2,3,4])
        fig_duplets = plot_duplets_from_enc_scores('admm', fam_key, sel,
            feature_label_fontsize=9, extra_label_gap_frac=0.012
        )
        save_fig_std('admm', fam_key, 'duplets', fig_duplets)

        # Save legend alongside in the duplets folder
        fig_legend = make_legend()
        save_fig_std('admm', fam_key, 'duplets', fig_legend, extra='duplet_legend')

    # Top features (profiles + grids) based on joint ΔR²
    print("\nStep 5: Top features (profiles + grids)")
    img_dir = os.path.join(config.data_dir, 'things', 'images')
    cats_all = ['_'.join(s.split('_')[:-1]) if '_' in s else s for s in all_stims]
    for fam_key, result in results.items():
        # Skip: old importance system removed
        available_ranks = list(result['components'].keys())
        viz_rank = get_viz_rank(fam_key, result['best_rank'], available_ranks)
        W = result['components'][viz_rank]

        imp_mean = imp.get('importance_mean')
        imp_std = imp.get('importance_std')
        names = imp.get('feature_names')
        groups = imp.get('feature_groups')
        if imp_mean is None or names is None or groups is None:
            print(f"  {fam_key}: incomplete importance payload, skipping top features")
            continue

        fig_top, summary = plot_top_components_profiles_and_grids_admm(
            fam_key, W, all_stims, cats_all, imp_mean, imp_std, names, groups,
            img_dir, top_k=TOP_K, rows=SELECTED_ROWS, cols=SELECTED_COLS, img_px=180
        )
        save_fig_std('admm', fam_key, 'top_features', fig_top, extra='top5_profiles_grids')

        # Save CSV summary
        top_dir = os.path.join(config.fig_dir, 'admm_analysis', fam_key, 'top_features')
        os.makedirs(top_dir, exist_ok=True)
        df = pd.DataFrame(summary, columns=['Component', 'MaxDeltaR2'])
        csv_path = os.path.join(top_dir, 'top5_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"    Saved summary to {csv_path}")

    print("\nStep 6: RSM matrices")
    visualize_rsms(results)

    print("\nStep 7: Summary")
    create_rank_summary(results, fig_dir)
    
    print(f"\nVisualization complete. Plots saved in: {fig_dir}")
    print("To change viz rank: edit config.toml [hyperparameters.admm.custom_ranks]")
    
    # Summary of what was processed
    all_families = ['all', 'human', 'monkey']
    missing_families = [f for f in all_families if f not in available_families]
    if missing_families:
        print(f"\nNote: Missing families {missing_families} - run CCA_ADMM.py to complete")

if __name__ == "__main__":
    main()
