#!/usr/bin/env python3
"""
ADMM-style component grids for SymNMF families.

Reproduces the exact visualization from viz/ADMM/viz_admm.py but
orders the top ROWS×COLS images deterministically (no subsampling).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.paths import config
from nmf.utils import (
    FAMILIES,
    load_admm_results,
    pick_rank,
    ensure_dir,
    load_guard_table,
    derive_categories,
)
from nmf.gallery import GALLERY_SPECS
from functions.plotting import setup_style
from functions.viz_common import save_fig_std

# Layout mirrors viz/ADMM/viz_admm.py selected-component figures
ROWS = 2
COLS = 6
IMG_PX = 360
MOSAIC_BORDER_WIDTH = 10.0
TOP_DEFAULT = 20


def subsample_top_pct(vec, n_cells):
    idx = np.argsort(vec)[::-1]
    return idx[:n_cells]


def load_img_simple(path, size):
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)

        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(1, size // 60)
        arr[:bw, :] = b_color
        arr[-bw:, :] = b_color
        arr[:, :bw] = b_color
        arr[:, -bw:] = b_color

        return Image.fromarray(arr)
    except Exception:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)


def _build_mosaic(paths, rows, cols, px):
    blank = np.full((px, px, 3), 128, np.uint8)
    tiles = [load_img_simple(p, px) if p is not None else Image.fromarray(blank)
             for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]

    grid = np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])
    return grid


def _border(ax, color='black', lw=3):
    ax.add_patch(
        Rectangle((0, 0), 1, 1,
                  transform=ax.transAxes,
                  facecolor='none',
                  edgecolor=color,
                  linewidth=lw,
                  clip_on=False)
    )


def _plot_distribution_positive_only(ax, vec, order, ipos, inv):
    vals = vec[order]
    total = len(vals)

    def _smooth(y):
        return gaussian_filter1d(y, sigma=len(y) / 50) if len(y) > 0 else y

    x = np.linspace(0, total - 1, 500)
    y = _smooth(np.interp(x, np.arange(total), vals))
    if y.max() > 0:
        y *= (vals.max() or 1) / y.max()

    for i in range(len(x) - 1):
        frac = x[i] / total if total else 0
        ax.fill_between(x[i:i + 2], 0, y[i:i + 2], color=cm.Reds(0.3 + 0.7 * frac), lw=0)

    ax.axhline(0, color='k', lw=0)
    ax.set_xlim(0, total)
    y0, y1 = ax.get_ylim()
    dy = y1 - y0
    ax.set_ylim(y0 - 0.05 * dy, y1 + 0.05 * dy)
    for sp in ax.spines.values():
        sp.set_visible(False)

    xp_all = inv[ipos]
    xp0, xp1 = xp_all.min(), xp_all.max()
    y0, y1 = ax.get_ylim()
    pos_color = config.plotting.get('positive_color', '#8B0000')
    ax.plot([xp0, xp1], [0, 0], color=pos_color, lw=7)
    ax.plot([xp0, xp0], [0, y1], color=pos_color, lw=7)
    ax.plot([xp1, xp1], [0, y1], color=pos_color, lw=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _build_positive_mosaic(W, ci, stims, cats, img_dir, rows=ROWS, cols=COLS, img_px=IMG_PX):
    weights = W[:, ci]
    n_cells = rows * cols
    ipos = subsample_top_pct(weights, n_cells)
    paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ipos]
    mosaic = _build_mosaic(paths, rows, cols, img_px)
    return mosaic, ipos, weights


def plot_comp_imgs_positive_only(comp_idx, W, stims, cats,
                                 img_dir=None, title=None,
                                 rows=ROWS, cols=COLS,
                                 img_px=IMG_PX, dpi=300):
    if img_dir is None:
        img_dir = os.path.join(config.data_dir, 'things', 'images')

    mosaic, ipos, vec = _build_positive_mosaic(W, comp_idx, stims, cats, img_dir, rows, cols, img_px)
    order = np.argsort(vec)
    inv = np.empty(len(vec), int)
    inv[order] = np.arange(len(vec))

    density_h = 200
    fig_h = (rows * img_px + density_h) / 100
    fig_w = (cols * img_px) / 100
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[rows * img_px, density_h], hspace=0.02)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(mosaic, interpolation='nearest', aspect='auto')
    ax_img.axis('off')
    _border(ax_img, color=config.plotting.get('positive_color', '#8B0000'), lw=MOSAIC_BORDER_WIDTH)
    if title:
        ax_img.set_title(title, fontsize=14)

    ax_den = fig.add_subplot(gs[1])
    _plot_distribution_positive_only(ax_den, vec, order, ipos, inv)
    ax_den.axis('off')
    return fig


def select_components(family, guard_df, spec, top_n):
    if guard_df is None or guard_df.empty:
        return []
    df = guard_df.copy()
    mask = spec['filter'](df)
    df = df[mask].copy()
    if df.empty:
        return []
    df['score'] = df.apply(spec['score'], axis=1)
    df = df.sort_values('score', ascending=False)
    return df['component'].astype(int).tolist()[:top_n]


def main():
    parser = argparse.ArgumentParser(description="ADMM-style SymNMF component grids.")
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES,
                        help='Families to visualize (default: all).')
    parser.add_argument('--top', type=int, default=TOP_DEFAULT,
                        help='Number of components per family.')
    args = parser.parse_args()

    setup_style()
    results, stims = load_admm_results()
    cats = derive_categories(stims)
    img_dir = os.path.join(config.data_dir, 'things', 'images')

    for family in args.families:
        spec = GALLERY_SPECS.get('shared' if family == 'all' else family)
        if spec is None:
            print(f"[viz_grid] No spec for family '{family}', skipping.")
            continue
        guard_df = load_guard_table(family)
        selected = select_components(family, guard_df, spec, args.top)
        if not selected:
            print(f"[viz_grid] {family}: no components matched criteria.")
            continue
        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        print(f"[viz_grid] {family}: rank {rank}, components {selected}")

        for rank_idx, comp in enumerate(selected, start=1):
            ci = comp - 1
            if ci < 0 or ci >= W.shape[1]:
                continue
            fig = plot_comp_imgs_positive_only(
                ci, W, stims, cats,
                img_dir=img_dir, rows=ROWS, cols=COLS,
                img_px=IMG_PX, dpi=300
            )
            fname = f"n{rank_idx}_comp{comp}"
            save_fig_std('nmf', family, 'selected', fig, extra=fname, image_high_dpi=True)


if __name__ == '__main__':
    main()
