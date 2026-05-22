#!/usr/bin/env python3
"""
SymNMF component gallery builder.

Expects ridge guard CSVs produced by nmf/ridge_guard.py and renders
large overview grids for shared, human-specific, and monkey-specific factors.
"""

from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
from matplotlib import font_manager
from PIL import Image

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

from config.paths import config
from nmf.utils import (
    load_admm_results,
    pick_rank,
    build_img_paths,
    ensure_dir,
    load_guard_table,
    derive_categories,
)

HERE = Path(__file__).resolve().parent
RESULT_DIR = ensure_dir(HERE / 'results')
ensure_dir(RESULT_DIR / 'guard')
FIG_DIR = ensure_dir(config.fig_dir / 'nmf_analysis')

# Gallery layout
TOP_IMAGES = 40
IMG_PX = 150
LABEL_W = 150
BAR_W = 240
WC_W = 320
GAP_SMALL = 20
GAP_IMG = 28
ROW_GAP = 22
DPI = 150
PANEL_DPI = 150

# Thresholds
R2_SHARED = 0.20
R2_SPEC = 0.20
R2_OTHER_MAX = 0.10

HUM_COLOR = config.plotting.get('human_color', '#7c5799')
MON_COLOR = config.plotting.get('monkey_color', '#bda855')
POS_COLOR = config.plotting.get('positive_color', '#A8393F')

GALLERY_SPECS = {
    'shared': {
        'family': 'all',
        'title': 'Shared components (cross-species SymNMF)',
        'filename': 'shared_gallery.pdf',
        'summary': 'shared_gallery.csv',
        'filter': lambda df: (df['human_r2'] >= R2_SHARED) & (df['monkey_r2'] >= R2_SHARED),
        'score': lambda row: np.minimum(row['human_r2'], row['monkey_r2']),
        'label_prefix': 'ALL',
    },
    'human': {
        'family': 'human',
        'title': 'Human-specific components (human SymNMF)',
        'filename': 'human_gallery.pdf',
        'summary': 'human_gallery.csv',
        'filter': lambda df: (df['human_r2'] >= R2_SPEC) & (df['monkey_r2'] < R2_OTHER_MAX),
        'score': lambda row: row['human_r2'] - row['monkey_r2'],
        'label_prefix': 'HUM',
    },
    'monkey': {
        'family': 'monkey',
        'title': 'Monkey-specific components (monkey SymNMF)',
        'filename': 'monkey_gallery.pdf',
        'summary': 'monkey_gallery.csv',
        'filter': lambda df: (df['monkey_r2'] >= R2_SPEC) & (df['human_r2'] < R2_OTHER_MAX),
        'score': lambda row: row['monkey_r2'] - row['human_r2'],
        'label_prefix': 'MON',
    },
}

IMG_CACHE = {}


def load_component_matrix(results, family, rank):
    return np.asarray(results[family]['components'][rank], float)


def _blank_tile(size):
    return np.full((size, size, 3), 200, np.uint8)


def load_tile(path):
    key = str(path)
    if key in IMG_CACHE:
        return IMG_CACHE[key]
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((IMG_PX, IMG_PX), Image.BILINEAR)
        arr = np.array(im)
        bw = max(1, IMG_PX // 60)
        arr[:bw] = 64
        arr[-bw:] = 64
        arr[:, :bw] = 64
        arr[:, -bw:] = 64
    except Exception:
        arr = _blank_tile(IMG_PX)
    IMG_CACHE[key] = arr
    return arr


def _fig_to_array(fig, target_width):
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    if h != IMG_PX or w != target_width:
        arr = np.asarray(Image.fromarray(arr).resize((target_width, IMG_PX), Image.BILINEAR))
    return arr


def _text_panel(width, text):
    fig = plt.figure(figsize=(width / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    return _fig_to_array(fig, width)


def render_bar_panel(human, monkey):
    fig = plt.figure(figsize=(BAR_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    vals = [float(human), float(monkey)]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]
    colors = [HUM_COLOR, MON_COLOR]
    xpos = np.arange(2)
    ax.bar(xpos, vals, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r'$R^2$', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4)
    return _fig_to_array(fig, BAR_W)


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
    return mpl.cm.colors.ListedColormap(colors)


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
        if n_pos < 5 or n_neg <= 0:
            continue
        mask = np.array([c == cat for c in cats])
        sum_ranks = ranks[mask].sum()
        auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        if np.isfinite(auc):
            aucs.append((cat, float(auc)))
    aucs.sort(key=lambda x: x[1], reverse=True)
    return aucs[:5]


def render_wordcloud(enrichment):
    freqs = {cat.replace('_', ' '): max(auc, 0.51) for cat, auc in enrichment if auc > 0.5}
    if WordCloud is None:
        return _text_panel(WC_W, 'Install wordcloud for enrichment view')
    if not freqs:
        return _text_panel(WC_W, 'Enrichment < 0.5 AUROC')
    cmap = _trunc_cmap('Reds', minval=0.75, maxval=1.0)
    wc = WordCloud(
        width=900,
        height=900,
        background_color='white',
        colormap=cmap,
        prefer_horizontal=1.0,
        min_font_size=60,
        font_path=_get_font_path()
    )
    wc.generate_from_frequencies(freqs)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        val = freqs.get(word, 0.5)
        keys = list(freqs.values())
        mn, mx = min(keys), max(keys)
        if mx == mn:
            t = 1.0
        else:
            t = (val - mn) / (mx - mn)
        r, g, b, _ = cmap(t)
        red = int(r * 255)
        green = int(g * 255)
        blue = int(b * 255)
        return "rgb({},{},{})".format(red, green, blue)

    wc.recolor(color_func=color_func)
    fig = plt.figure(figsize=(WC_W / PANEL_DPI, IMG_PX / PANEL_DPI), dpi=PANEL_DPI)
    ax = fig.add_subplot(111)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return _fig_to_array(fig, WC_W)


def build_entries(spec_key, spec, guard_df, W, cats, cat_counts):
    df = guard_df.copy()
    mask = spec['filter'](df)
    df = df[mask].copy()
    if df.empty:
        return []
    df['score'] = df.apply(spec['score'], axis=1)
    df.sort_values('score', ascending=False, inplace=True)
    entries = []
    for rank_idx, (_, row) in enumerate(df.iterrows(), start=1):
        idx = int(row['component']) - 1
        weights = W[:, idx]
        comp_id = int(row['component'])
        entries.append({
            'comp_idx': idx,
            'label': f"{spec['label_prefix']} N{rank_idx} C{comp_id}",
            'score': float(row['score']),
            'human_r2': float(row['human_r2']),
            'human_r2_std': float(row.get('human_r2_std', np.nan)),
            'monkey_r2': float(row['monkey_r2']),
            'monkey_r2_std': float(row.get('monkey_r2_std', np.nan)),
            'weights': weights,
            'enrichment': compute_enrichment(weights, cats, cat_counts),
        })
    return entries


def assemble_canvas(entries, img_paths):
    if not entries:
        return None, None
    width_images = TOP_IMAGES * IMG_PX
    gap_small = np.full((IMG_PX, GAP_SMALL, 3), 255, np.uint8)
    gap_img = np.full((IMG_PX, GAP_IMG, 3), 255, np.uint8)
    label_block = np.full((IMG_PX, LABEL_W, 3), 255, np.uint8)
    total_width = LABEL_W + 2 * GAP_SMALL + BAR_W + WC_W + GAP_IMG + width_images
    gap_row = np.full((ROW_GAP, total_width, 3), 255, np.uint8)

    parts = []
    row_meta = []
    cursor = 0

    for ent in entries:
        weights = ent['weights']
        order = np.argsort(weights)[::-1][:TOP_IMAGES]
        tiles = [load_tile(img_paths[i]) for i in order]
        while len(tiles) < TOP_IMAGES:
            tiles.append(_blank_tile(IMG_PX))

        img_block = np.hstack(tiles)
        bar_panel = render_bar_panel(ent['human_r2'], ent['monkey_r2'])
        wc_panel = render_wordcloud(ent['enrichment'])

        row = np.hstack([
            label_block.copy(),
            gap_small.copy(),
            bar_panel,
            gap_small.copy(),
            wc_panel,
            gap_img.copy(),
            img_block,
        ])
        parts.append(row)
        row_meta.append({
            'y': cursor,
            'label': ent['label'],
        })
        cursor += IMG_PX
        parts.append(gap_row.copy())
        cursor += ROW_GAP

    canvas = np.vstack(parts[:-1])
    return canvas, row_meta


def render_gallery(spec_key, entries, img_paths, title, fname):
    canvas, meta = assemble_canvas(entries, img_paths)
    if canvas is None:
        print(f"[gallery] {spec_key}: no components matched thresholds.")
        return

    h, w = canvas.shape[:2]
    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax.imshow(canvas)
    ax.axis('off')

    label_x = LABEL_W * 0.5
    bar_left = LABEL_W + GAP_SMALL
    bar_x = bar_left + BAR_W * 0.5
    wc_left = bar_left + BAR_W + GAP_SMALL
    wc_x = wc_left + WC_W * 0.5
    img_left = wc_left + WC_W + GAP_IMG
    img_x = img_left + (TOP_IMAGES * IMG_PX) * 0.5

    for info in meta:
        y = info['y'] + IMG_PX * 0.5
        ax.text(label_x, y, info['label'], ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(w * 0.5, -10, title, ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(bar_x, -2, 'Predictability (R²)', ha='center', va='bottom', fontsize=11)
    ax.text(wc_x, -2, 'Word cloud', ha='center', va='bottom', fontsize=11)
    ax.text(img_x, -2, 'Top stimuli', ha='center', va='bottom', fontsize=11)
    out_fp = FIG_DIR / fname
    fig.savefig(out_fp, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[gallery] saved {out_fp}")


def save_summary(entries, path):
    if not entries:
        return
    rows = []
    for ent in entries:
        rows.append({
            'label': ent['label'],
            'component': ent['comp_idx'] + 1,
            'human_r2': ent['human_r2'],
            'human_r2_std': ent.get('human_r2_std'),
            'monkey_r2': ent['monkey_r2'],
            'monkey_r2_std': ent.get('monkey_r2_std'),
            'score': ent['score'],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Render SymNMF galleries from guard CSVs.")
    parser.add_argument('--which', nargs='+', choices=list(GALLERY_SPECS.keys()), default=None,
                        help='Which galleries to build (default: all).')
    args = parser.parse_args()

    results, stims = load_admm_results()
    img_paths = build_img_paths(stims)
    cats = derive_categories(stims)
    cat_counts = Counter(cats)

    targets = args.which or list(GALLERY_SPECS.keys())

    for key in targets:
        spec = GALLERY_SPECS[key]
        family = spec['family']
        guard_df = load_guard_table(family)
        rank = pick_rank(family, results[family])
        W = load_component_matrix(results, family, rank)
        entries = build_entries(key, spec, guard_df, W, cats, cat_counts)

        render_gallery(key, entries, img_paths, spec['title'], spec['filename'])
        summary_fp = RESULT_DIR / spec['summary']
        save_summary(entries, summary_fp)
        if entries:
            print(f"[gallery] {key}: {len(entries)} components listed; summary -> {summary_fp}")


if __name__ == '__main__':
    main()
