#!/usr/bin/env python3
"""Component 8 loading bar visualization with image strip."""
import os
import sys
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from PIL import Image

_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up - 1]
    if (cand / 'config').exists():
        if str(cand) not in sys.path:
            sys.path.append(str(cand))
        break

from config.paths import config
from functions.plotting import setup_style, wide_figure

COMPONENT = 7
N_PER_SIDE = 100  # number of stimuli shown on each side (negative/positive)
TOP_K = 5
IMG_PX = 240
DPI = 600
DEFAULT_BAR_COLOR = '#b3b3b3'
SYMBOL_LIGHTEN_MIX = 0.35
NUMEROSITY_DARKEN_MIX = 0.25
OUT_DIR = config.fig_dir / 'symbol' / 'symb'
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    symbol_color = lighten_color(base_icon, SYMBOL_LIGHTEN_MIX)
    icon_color = to_rgb(base_icon)
    numerosity_color = darken_color(base_icon, NUMEROSITY_DARKEN_MIX)
    return {
        'symbol': symbol_color,
        'icon': icon_color,
        'numerosity': numerosity_color,
    }


def square_and_resize(path, target):
    """Load → center-crop to square → resize with border (mirrors viz_symb)."""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, upper = (w - side) // 2, (h - side) // 2
        im = im.crop((left, upper, left + side, upper + side))
        im = im.resize((target, target), Image.LANCZOS)
        arr = np.array(im, copy=True)
    except Exception:
        arr = np.full((target, target, 3), 128, dtype=np.uint8)

    border = max(4, target // 36)
    color = (48, 48, 48)
    arr[:border, :] = color
    arr[-border:, :] = color
    arr[:, :border] = color
    arr[:, -border:] = color
    return Image.fromarray(arr)


def build_strip(paths, px):
    tiles = [square_and_resize(p, px) for p in paths]
    tiles = [np.asarray(t) for t in tiles if t is not None]
    if not tiles:
        return np.full((px, px, 3), 128, dtype=np.uint8)
    return np.hstack(tiles)


def load_state():
    with open(config.cache_file, 'rb') as f:
        return pickle.load(f)


def load_annotations(stims):
    ann = pd.read_csv(config.things_dir / 'annotations.csv')
    ann['stim'] = ann['filename'].astype(str)
    return ann.set_index('stim').reindex(stims)


def normalize_views(components, view_names):
    mats = []
    for v in view_names:
        mat = components[v]
        mu = mat.mean(axis=0, keepdims=True)
        sd = mat.std(axis=0, keepdims=True)
        sd[sd == 0] = 1
        mats.append((mat - mu) / sd)
    return np.mean(np.stack(mats, axis=0), axis=0)


def prepare_subset(scores, annotations):
    order = np.argsort(scores)
    if 2 * N_PER_SIDE > len(order):
        sel_idx = order
    else:
        sel_idx = np.concatenate([order[:N_PER_SIDE], order[-N_PER_SIDE:]])
    return sel_idx


def make_figure(scores, stims, cats, annotations):
    palette = get_palette()
    sel_idx = prepare_subset(scores, annotations)
    sel_scores = scores[sel_idx]

    labels = annotations.loc[stims[sel_idx], ['symbol', 'icon', 'numerosity']].fillna(False).astype(bool)
    positions = np.arange(len(sel_idx))

    setup_style()
    fig, ax_bar = plt.subplots(figsize=(8.2, 3.0))
    bar_width = 0.9
    feature_order = ['symbol', 'icon', 'numerosity']
    for pos, score, row in zip(positions, sel_scores, labels.itertuples(index=False)):
        cats = [feat for feat in feature_order if getattr(row, feat)]
        x_left = pos - bar_width / 2
        if not cats:
            ax_bar.bar(pos, score, color=DEFAULT_BAR_COLOR, edgecolor='none', linewidth=0, width=bar_width, zorder=3)
            continue
        if score >= 0:
            seg_height = score / len(cats) if len(cats) else score
            curr = 0.0
            for cat in cats:
                color = palette.get(cat, DEFAULT_BAR_COLOR)
                rect = Rectangle((x_left, curr), bar_width, seg_height,
                                 facecolor=color, edgecolor='none', zorder=3)
                ax_bar.add_patch(rect)
                curr += seg_height
        else:
            seg_height = score / len(cats) if len(cats) else score
            curr = 0.0
            for cat in cats:
                color = palette.get(cat, DEFAULT_BAR_COLOR)
                rect = Rectangle((x_left, curr + seg_height), bar_width, -seg_height,
                                 facecolor=color, edgecolor='none', zorder=3)
                ax_bar.add_patch(rect)
                curr += seg_height

    ax_bar.axhline(0, color='#222222', linewidth=2.5)
    max_abs = np.max(np.abs(sel_scores)) if sel_scores.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    margin = max_abs * 0.05
    ax_bar.set_ylim(-max_abs - margin, max_abs + margin)
    ax_bar.set_xlim(-0.5, len(positions) - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_axis_off()

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.10)

    legend_path = OUT_DIR / 'component_legend.pdf'
    if not legend_path.exists():
        legend_fig, legend_ax = plt.subplots(figsize=(3.0, 1.6))
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette['symbol'], ec='black', lw=0.6, label='Symbol'),
            plt.Rectangle((0, 0), 1, 1, color=palette['icon'], ec='black', lw=0.6, label='Icon'),
            plt.Rectangle((0, 0), 1, 1, color=palette['numerosity'], ec='black', lw=0.6, label='Numerosity')
        ]
        legend_ax.legend(handles=handles, frameon=False, loc='center', ncol=1)
        legend_ax.axis('off')
        legend_fig.savefig(legend_path, dpi=DPI, bbox_inches='tight')
        plt.close(legend_fig)

    out_path = OUT_DIR / f'component_{COMPONENT:02d}_loading_strip.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def save_negative_strip(scores, stims, cats, annotations, feature, used_images):
    palette = get_palette()
    mask = annotations[feature].fillna(False).astype(bool).to_numpy()
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    order = np.argsort(scores[idx])  # most negative first
    sel = []
    img_dir = config.things_dir / 'images'
    for i in idx[order]:
        img_name = stims[i]
        if img_name in used_images:
            continue
        sel.append(i)
        used_images.add(img_name)
        if len(sel) >= TOP_K:
            break
    if not sel:
        return

    paths = [str(img_dir / cats[i] / f"{stims[i]}.jpg") for i in sel]
    strip = build_strip(paths, IMG_PX)

    setup_style()
    fig = plt.figure(figsize=(strip.shape[1] / DPI, strip.shape[0] / DPI), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.imshow(strip, aspect='auto')
    ax.axis('off')
    strip_border = max(4, IMG_PX // 36)
    ax.add_patch(Rectangle((0, 0), strip.shape[1], strip.shape[0], fill=False, edgecolor=(48/255,48/255,48/255), linewidth=strip_border / 2.0))
    fig.subplots_adjust(0, 0, 1, 1)
    out_path = OUT_DIR / f'component_{COMPONENT:02d}_{feature}_neg_strip.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    state = load_state()
    stims = np.array(state['all_stims'])
    components = state['cca_all']['components']

    view_names = ['human_01', 'human_02', 'human_03', 'monkey_N', 'monkey_F']
    comps = normalize_views(components, view_names)
    scores = comps[:, COMPONENT - 1]

    ann = load_annotations(stims)
    cats = np.array([s.rsplit('_', 1)[0] for s in stims])

    make_figure(scores, stims, cats, ann)
    used_images = set()
    for feat in ['symbol', 'icon', 'numerosity']:
        save_negative_strip(scores, stims, cats, ann, feat, used_images)


if __name__ == '__main__':
    main()
