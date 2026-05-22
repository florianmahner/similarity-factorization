#!/usr/bin/env python3
"""
Top SymNMF components with ridge predictability bars.

Generates publication-ready mosaics plus guard insets for each family.
"""

import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm

from config.paths import config
from nmf.utils import (
    FAMILIES,
    load_admm_results,
    pick_rank,
    build_img_paths,
    ensure_dir,
    load_guard_table,
    derive_categories,
)
from nmf.viz_grid import plot_comp_imgs_positive_only
from functions.plotting import setup_style, style_axes, format_axes, narrow_figure, wide_figure, square_figure
from functions.viz_common import save_fig_std
from matplotlib import font_manager
from matplotlib import colormaps
from nmf.gallery import GALLERY_SPECS

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

# Import ADMM viz function to ensure exact visual match
from viz.ADMM import viz_admm as admmviz

# Component selection
TOP_DEFAULT = 20
CUSTOM_COMPONENTS = {
    'all': [],
    'human': [],
    'monkey': [],
}

# Mosaic layout (match viz_admm selected grids)
ROWS = 2
COLS = 6
IMG_PX = 360

# Guard inset layout
BAR_PANEL_PX = 240
BAR_PAD = 0.15

# Output folders per family
FAMILY_DIR = {'all': 'cross', 'human': 'human', 'monkey': 'monkey'}

# Colors
HUM_COLOR = config.plotting.get('human_color', '#7c5799')
MON_COLOR = config.plotting.get('monkey_color', '#bda855')
POS_COLOR = config.plotting.get('positive_color', '#A8393F')

# Category enrichment settings
ENRICH_TOP_FRAC = 0.10  # legacy lift fraction (unused for AUROC but kept for potential reference)
ENRICH_TOP_N = 5
ENRICH_MIN_COUNT = 5

HERE = Path(__file__).resolve().parent
OUT_DIR = ensure_dir(HERE / 'figures' / 'topnmf')

SPEC_BY_FAMILY = {spec['family']: spec for spec in GALLERY_SPECS.values()}


def _fmt(val):
    return "n/a" if not np.isfinite(val) else f"{val:.3f}"


def parse_custom(overrides):
    custom = {k: list(v) for k, v in CUSTOM_COMPONENTS.items()}
    for entry in overrides or []:
        if ':' not in entry:
            continue
        fam, comps = entry.split(':', 1)
        fam = fam.strip().lower()
        if fam not in custom:
            continue
        ids = []
        for token in comps.replace(';', ',').split(','):
            token = token.strip()
            if token.isdigit():
                ids.append(int(token))
        if ids:
            custom[fam] = ids
    return custom


def select_components(family, guard_df, spec, top_n, custom_ids):
    if custom_ids:
        return [int(c) for c in custom_ids]
    df = guard_df.copy()
    mask = spec['filter'](df)
    df = df[mask].copy()
    if df.empty:
        return []
    df['score'] = df.apply(spec['score'], axis=1)
    df = df.sort_values('score', ascending=False)
    return df['component'].astype(int).tolist()[:top_n]




def guard_stats(guard_df, comp_idx):
    row = guard_df[guard_df['component'] == comp_idx + 1]
    if row.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    r = row.iloc[0]
    return (
        float(r.get('human_r2', np.nan)),
        float(r.get('human_r2_std', np.nan)),
        float(r.get('monkey_r2', np.nan)),
        float(r.get('monkey_r2_std', np.nan)),
    )


def draw_guard_bar(ax, human, human_std, monkey, monkey_std):
    vals = [human, monkey]
    errs = [
        human_std if np.isfinite(human_std) and human_std > 0 else 0.0,
        monkey_std if np.isfinite(monkey_std) and monkey_std > 0 else 0.0,
    ]
    colors = [HUM_COLOR, MON_COLOR]
    labels = ['Human', 'Monkey']
    xpos = np.arange(2)
    ax.bar(xpos, vals, color=colors, edgecolor='black', linewidth=1.2)
    for x, val, err in zip(xpos, vals, errs):
        if err > 0:
            ax.errorbar(x, val, yerr=err, fmt='none', ecolor='black',
                        elinewidth=1.2, capsize=3, capthick=1.0)
    ymax = max([val + err for val, err in zip(vals, errs)] + [0.5]) * 1.25
    ax.set_ylim(0, ymax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$R^2$', fontsize=11)
    style_axes(ax)
    format_axes(ax, precision=3)
    ax.margins(x=0.2)


def _lighten(color, mix=0.35):
    base = np.array(mcolors.to_rgb(color))
    return tuple(base * (1 - mix) + mix)


def _get_font_path(preferred=('Arial', 'Liberation Sans', 'DejaVu Sans')):
    for name in preferred:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path:
                return path
        except Exception:
            continue
    try:
        return font_manager.findfont('sans-serif')
    except Exception:
        return None


def _trunc_cmap(name='Reds', minval=0.75, maxval=1.0, n=256):
    base = colormaps.get_cmap(name)
    colors = base(np.linspace(minval, maxval, n))
    return cm.colors.ListedColormap(colors)




def compute_enrichment(weights, cats, cat_counts):
    n = len(weights)
    if n == 0:
        return []
    order = np.argsort(weights)
    sorted_w = weights[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_w[j] == sorted_w[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    aucs = []
    for cat, base in cat_counts.items():
        n_pos = base
        n_neg = n - n_pos
        if n_pos < ENRICH_MIN_COUNT or n_neg <= 0:
            continue
        mask = np.array([c == cat for c in cats])
        sum_ranks = ranks[mask].sum()
        auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        if np.isfinite(auc):
            aucs.append((cat, float(auc)))
    aucs.sort(key=lambda x: x[1], reverse=True)
    return aucs[:ENRICH_TOP_N]


def plot_enrichment(enrichment, comp_idx):
    if not enrichment:
        return None
    cats, aucs = zip(*enrichment)
    fig = square_figure()
    ax = fig.add_subplot(111)
    colors = [_lighten(POS_COLOR, mix=0.15 + 0.6 * (i / max(1, len(aucs) - 1))) for i in range(len(aucs))]
    x = np.arange(len(aucs))
    ax.bar(x, aucs, color=colors, edgecolor='black', linewidth=1.0)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=3.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right')
    ax.set_ylabel('AUROC')
    ax.set_title(f'Category AUROC – C{comp_idx+1}')
    ax.set_ylim(0.4, 1.0)
    style_axes(ax)
    format_axes(ax)
    return fig


def plot_wordcloud(enrichment, comp_idx):
    if WordCloud is None or not enrichment:
        return None
    freqs = {cat.replace('_', ' '): max(auc, 0.51) for cat, auc in enrichment if auc > 0.5}
    if not freqs:
        return None
    cmap = _trunc_cmap('Reds', minval=0.75, maxval=1.0)
    min_f = min(freqs.values())
    max_f = max(freqs.values())
    wc = WordCloud(
        width=900,
        height=900,
        background_color=None,
        mode='RGBA',
        colormap=cmap,
        prefer_horizontal=1.0,
        min_font_size=100,
        font_path=_get_font_path()
    )
    wc.generate_from_frequencies(freqs)
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        val = freqs.get(word, min_f)
        if max_f == min_f:
            t = 1.0
        else:
            t = (val - min_f) / (max_f - min_f)
        r, g, b, _ = cmap(t)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    wc.recolor(color_func=color_func)
    fig = square_figure()
    ax = fig.add_subplot(111)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def render_component(family, comp_idx, rank_idx, W, stims, img_paths, guard_df, spec, cats, cat_counts):
    fig_grid = plot_comp_imgs_positive_only(
        comp_idx, W, stims, cats,
        img_dir=os.path.join(config.data_dir, 'things', 'images'),
        title=None,
        rows=ROWS,
        cols=COLS,
        img_px=IMG_PX,
        dpi=300
    )
    fname = f"n{rank_idx}_comp{comp_idx + 1}"
    save_fig_std('nmf', family, 'selected', fig_grid, extra=fname, image_high_dpi=True)

    # Minimal, separate ridge bar plot to the right (standalone PDF)
    human, human_std, monkey, monkey_std = guard_stats(guard_df, comp_idx)
    fig_bar = narrow_figure()
    ax = fig_bar.add_subplot(111)
    draw_guard_bar(ax, human, human_std, monkey, monkey_std)
    save_fig_std('nmf', family, 'ridge_bars', fig_bar, extra=fname)

    enrichment = compute_enrichment(W[:, comp_idx], cats, cat_counts)
    fig_enrich = plot_enrichment(enrichment, comp_idx)
    if fig_enrich is not None:
        save_fig_std('nmf', family, 'enrichment', fig_enrich, extra=fname)
    fig_wc = plot_wordcloud(enrichment, comp_idx)
    if fig_wc is not None:
        save_fig_std('nmf', family, 'wordcloud', fig_wc, extra=fname)
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="SymNMF top component visualization with guard insets.")
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES,
                        help='Families to process (default: all).')
    parser.add_argument('--top', type=int, default=TOP_DEFAULT,
                        help='Number of components per family when no custom list is provided.')
    parser.add_argument('--custom', action='append', default=[],
                        help='Override components per family, e.g., "human:5,7,9". Can be repeated.')
    args = parser.parse_args()

    setup_style()
    custom = parse_custom(args.custom)

    results, stims = load_admm_results()
    img_paths = build_img_paths(stims)
    cats = derive_categories(stims)
    cat_counts = Counter(cats)

    for family in args.families:
        spec = SPEC_BY_FAMILY.get(family)
        if spec is None:
            print(f"[viz_topnmf] No spec available for family '{family}', skipping.")
            continue
        guard_df = load_guard_table(family)
        selected = select_components(family, guard_df, spec, args.top, custom.get(family))
        if not selected:
            print(f"[viz_topnmf] {family}: no components matched criteria.")
            continue
        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)

        print(f"[viz_topnmf] {family}: rendering components {selected}")
        for rank_idx, comp in enumerate(selected, start=1):
            idx = int(comp) - 1
            if idx < 0 or idx >= W.shape[1]:
                print(f"[viz_topnmf] {family}: component {comp} out of range, skipping.")
                continue
            render_component(family, idx, rank_idx, W, stims, img_paths, guard_df, spec, cats, cat_counts)


if __name__ == '__main__':
    main()
