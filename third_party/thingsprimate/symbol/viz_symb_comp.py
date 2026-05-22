#!/usr/bin/env python3
"""
Negative-pole mosaics for cross-species CCA components 7 and 8.

Matches the mosaic style of category/things_bodyparts.py, but stripped down to
just the grid + bounding box. Adds arrows per thumbnail to indicate symbol/icon/
numerosity membership (from data/things/annotations.csv). Grid is 2x6 with
square thumbnails.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import to_rgb
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.paths import config
from functions.plotting import setup_style

# Layout (mirrors category/things_bodyparts.py)
ROWS = 2
COLS = 6
IMG_SIZE = 800
IMG_BORDER_FRAC = 1.0 / 35.0        # tile border relative to side (wider)
MOSAIC_BORDER_FRAC = 0.08           # outline relative to tile size (wider)
MOSAIC_BORDER_WIDTH = 10.0
TOP_N = ROWS * COLS

# Arrow styling (scaled by IMG_SIZE)
ARROW_OFFSETS = [-0.2, 0.0, 0.2]
ARROW_LEN_FRAC = 0.30
ARROW_PAD_FRAC = 0.10
ARROW_MUT_SCALE_FRAC = 0.50
ARROW_LINEWIDTH_FRAC = 0.020

COMPONENTS = [7, 8]          # 1-indexed cross-species components
STATE_FP = Path(config.results_dir) / 'cca_state.pkl'
IMG_DIR = Path(config.data_dir) / 'things' / 'images'
OUT_DIR = Path(config.fig_dir) / 'symbol' / 'symb_comp'
ANN_PATH = Path(config.things_dir) / 'annotations.csv'

NEG_COLOR = config.plotting.get('negative_color', '#000080')


def lighten_color(color, mix=0.35):
    mix = max(0.0, min(1.0, mix))
    rgb = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple(rgb * (1 - mix) + white * mix)


def darken_color(color, mix=0.25):
    mix = max(0.0, min(1.0, mix))
    rgb = np.array(to_rgb(color))
    black = np.zeros(3)
    return tuple(rgb * (1 - mix) + black * mix)


def get_palette():
    base_icon = config.plotting.get('symbol_color', '#2874A6')
    symbol_color = lighten_color(base_icon, 0.35)
    icon_color = to_rgb(base_icon)
    numerosity_color = darken_color(base_icon, 0.25)
    return {
        'symbol': symbol_color,
        'icon': icon_color,
        'numerosity': numerosity_color,
    }


# --------------------------------------------------------------------- helpers
def load_state(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_annotations():
    ann = pd.read_csv(ANN_PATH)
    ann['stim'] = ann['filename'].astype(str)
    return ann.set_index('stim')


def load_img_simple(path, size):
    try:
        im = Image.open(path).convert('RGB')
        w0, h0 = im.size
        side = min(w0, h0)
        left = (w0 - side) // 2
        top = (h0 - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BILINEAR)

        arr = np.array(im)
        b_color = (64, 64, 64)
        bw = max(2, int(round(size * IMG_BORDER_FRAC)))
        arr[:bw, :] = b_color; arr[-bw:, :] = b_color
        arr[:, :bw] = b_color; arr[:, -bw:] = b_color
        return Image.fromarray(arr)
    except Exception:
        blank = np.full((size, size, 3), 128, np.uint8)
        return Image.fromarray(blank)


def _build_mosaic(paths, rows, cols, size):
    blank = np.full((size, size, 3), 128, np.uint8)
    tiles = [load_img_simple(p, size) if p is not None else Image.fromarray(blank) for p in paths]
    tiles += [Image.fromarray(blank)] * (rows * cols - len(tiles))
    tiles = [np.asarray(t) for t in tiles]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    return grid


def _border(ax, color='black', lw=3):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor='none', edgecolor=color,
                           linewidth=lw, clip_on=False))


def _build_negative_mosaic(scores, stims, cats, img_dir, rows=ROWS, cols=COLS, img_px=IMG_SIZE):
    order = np.argsort(scores)  # most negative first
    ineg = order[:rows * cols]
    paths = [os.path.join(img_dir, cats[i], f"{stims[i]}.jpg") for i in ineg]
    mosaic = _build_mosaic(paths, rows, cols, img_px)
    return mosaic, ineg


# ------------------------------------------------------------------- plotting
def _arrow_positions(rows, cols, img_px):
    pos = []
    for k in range(rows * cols):
        r = k // cols
        c = k % cols
        x0 = c * img_px
        y0 = r * img_px
        pos.append((x0, y0))
    return pos


def _draw_arrows(ax, pos_idx, stims, ann, rows, cols, img_px, palette):
    grid_pos = _arrow_positions(rows, cols, img_px)
    offsets = [-0.2, 0.0, 0.2]
    feats = ['symbol', 'icon', 'numerosity']
    arrow_len = img_px * 0.30
    for k, idx in enumerate(pos_idx):
        stim = stims[idx]
        rec = ann.loc[str(stim)] if str(stim) in ann.index else None
        if rec is None:
            continue
        flags = [bool(rec.get(f, 0)) for f in feats]
        if not any(flags):
            continue
        x0, y0 = grid_pos[k]
        x_mid = x0 + img_px / 2
        top_y = y0 + img_px * 0.05
        pad = img_px * 0.10
        on = 0
        for off, feat, flag in zip(offsets, feats, flags):
            if not flag:
                continue
            x = x_mid + off * img_px
            arrow = FancyArrowPatch(
                (x, top_y - pad), (x, top_y + arrow_len),
                arrowstyle='simple', mutation_scale=60,
                linewidth=2.0, facecolor=palette[feat],
                edgecolor='#3a3a3a', alpha=0.95
            )
            ax.add_patch(arrow)
            on += 1


def plot_component_negative(scores, stims, cats, comp_id, ann, palette, out_path):
    if scores is None or len(scores) == 0:
        print(f'[comp{comp_id:02d}] no scores found; skipping.')
        return

    mosaic, ineg = _build_negative_mosaic(scores, stims, cats, IMG_DIR)

    setup_style()
    fig_w = (COLS * IMG_SIZE) / 120
    fig_h = (ROWS * IMG_SIZE) / 120
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(mosaic, interpolation='nearest')
    ax.axis('off')
    ax.set_xlim(-0.5, mosaic.shape[1] - 0.5)
    ax.set_ylim(mosaic.shape[0] - 0.5, -0.5)
    ax.set_aspect('equal')
    mosaic_outline = max(3.0, mosaic.shape[1] * MOSAIC_BORDER_FRAC)
    _border(ax, color=NEG_COLOR, lw=mosaic_outline)
    _draw_arrows(ax, ineg, stims, ann, ROWS, COLS, IMG_SIZE, palette)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[comp{comp_id:02d}] saved figure to {out_path}")


# ----------------------------------------------------------------------- main
def main():
    if not STATE_FP.exists():
        print(f"CCA state file not found: {STATE_FP}")
        return

    print(f"Loading CCA state from {STATE_FP}...")
    state = load_state(STATE_FP)
    cca_all = state['cca_all']
    stims = np.array(state['all_stims'])
    views_all = sorted(cca_all['components'].keys())
    comps_all = np.mean([cca_all['components'][v] for v in views_all], axis=0)
    cats = ['_'.join(str(sv).split('_')[:-1]) if '_' in str(sv) else str(sv) for sv in stims]
    ann = load_annotations()
    palette = get_palette()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for comp_id in COMPONENTS:
        ci = comp_id - 1
        if ci < 0 or ci >= comps_all.shape[1]:
            print(f"[comp{comp_id:02d}] out of range; skipping.")
            continue
        scores = comps_all[:, ci]
        out_path = OUT_DIR / f"comp{comp_id:02d}_negative.pdf"
        plot_component_negative(scores, stims, cats, comp_id, ann, palette, out_path)


if __name__ == '__main__':
    main()
