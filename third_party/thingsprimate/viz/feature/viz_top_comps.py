#!/usr/bin/env python3
from __future__ import annotations

import os, sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Limit thread fan-out
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
os.environ.setdefault('OMP_PROC_BIND', 'TRUE')

_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up-1]
    if (cand / 'config').exists():
        if str(cand) not in sys.path:
            sys.path.append(str(cand))
        break

from config.paths import config
from functions.plotting import setup_style, style_axes, format_axes, wide_figure

TOP_N = 10
RES_DIR = config.results_dir / 'feature' / 'species' / 'cca'
FIG_DIR = config.fig_dir / 'feature'
FIG_DIR.mkdir(parents=True, exist_ok=True)

_ARCH_CONF = config.plotting.get('model_arch', {})
_ARCH_BLOCKS = []
_CANON_KEYS = []
for arch_key, ent in _ARCH_CONF.items():
    canon_models = [str(m).lower() for m in ent.get('models', [])]
    if not canon_models:
        continue
    block = {
        'arch': str(arch_key),
        'canon_models': canon_models,
        'cmap': str(ent.get('cmap', config.plotting.get('model_cmap', 'magma'))),
        'tmin': float(ent.get('tmin', config.plotting.get('model_cmap_tmin', 0.25))),
        'tmax': float(ent.get('tmax', config.plotting.get('model_cmap_tmax', 0.85))),
    }
    _ARCH_BLOCKS.append(block)
    _CANON_KEYS.extend(canon_models)

_MODEL_HATCH = {str(k).lower(): str(v) for k, v in config.plotting.get('model_hatch', {}).items()}


def _canon_model_name(name: str) -> str:
    n = str(name).lower()
    for canon in _CANON_KEYS:
        if n == canon or canon in n:
            return canon
    raise KeyError(f"Model '{name}' not registered in config.plotting.model_arch")


def ordered_models(models: Iterable[str]) -> List[str]:
    unique = list(dict.fromkeys(list(models)))
    out = []
    for block in _ARCH_BLOCKS:
        for canon in block['canon_models']:
            for m in unique:
                if m in out:
                    continue
                try:
                    if _canon_model_name(m) == canon:
                        out.append(m)
                except KeyError:
                    continue
    remaining = [m for m in unique if m not in out]
    if remaining:
        raise KeyError(f"Missing model_arch config for: {', '.join(remaining)}")
    return out


def _soft_color(col, mix=0.2):
    arr = np.array(col[:3], float)
    muted = arr * (1.0 - mix) + mix
    alpha = col[3] if len(col) == 4 else 1.0
    return (muted[0], muted[1], muted[2], alpha)


def model_colors(models: Iterable[str]) -> Dict[str, tuple]:
    models = ordered_models(models)
    if not models:
        return {}
    out = {}
    for block in _ARCH_BLOCKS:
        block_models = [m for m in models if _canon_model_name(m) in block['canon_models']]
        if not block_models:
            continue
        cmap_name = block['cmap']
        tmin = block['tmin']
        tmax = block['tmax']
        try:
            cmap = plt.colormaps.get_cmap(cmap_name)
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
        ts = np.linspace(tmin, tmax, max(2, len(block_models)))
        for i, m in enumerate(block_models):
            out[m] = _soft_color(cmap(float(ts[i])))
    return out


def _model_hatch(name: str) -> str:
    try:
        canon = _canon_model_name(name)
    except KeyError:
        return ''
    return _MODEL_HATCH.get(canon, '')


def _load_cached_scores() -> pd.DataFrame:
    # Prefer the aggregated all-rows CSV; fallback to 'best' if present
    all_csv = RES_DIR / 'dnn_component_layer_scores.csv'
    best_csv = RES_DIR / 'dnn_component_layer_best.csv'
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        need = {'family','component','model','layer_key','metric_mean','metric_std'}
        if set(df.columns) >= need:
            df['metric_type'] = df.get('metric_type', 'r')
            return df
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        if set(df.columns) >= {'family','component','model','metric_mean','metric_std','metric_type'}:
            return df[df['metric_type'] == 'r'].copy()
    return pd.DataFrame()


def _select_best_per_component(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df.copy()
    sub = sub.sort_values('metric_mean', ascending=False).drop_duplicates(['family','component'])
    return sub


def compute_best_layers() -> pd.DataFrame:
    df = _load_cached_scores()
    if df.empty:
        return df
    if 'metric_type' in df.columns:
        df = df[df['metric_type'] == 'r'] if 'r' in set(df['metric_type']) else df
    return _select_best_per_component(df)


def plot_top(best: pd.DataFrame) -> None:
    families = [
        ('monkey', 'Macaque', 'M'),
        ('cross-species', 'Cross-species', 'X'),
        ('human', 'Human', 'H'),
    ]
    model_list = list(best['model'].unique())
    models = ordered_models(model_list) if model_list else []
    colors = model_colors(models)
    records = []
    for fam_key, title, abbrev in families:
        sub = best[best['family'] == fam_key]
        if sub.empty:
            continue
        sub = sub.sort_values('metric_mean', ascending=False).drop_duplicates('component')
        sub = sub[sub['component'] <= TOP_N].sort_values('component')
        for _, row in sub.iterrows():
            comp = int(row['component'])
            records.append({
                'family': fam_key,
                'title': title,
                'abbr': abbrev,
                'component': comp,
                'metric_mean': float(row['metric_mean']),
                'metric_std': float(row['metric_std']),
                'model': row['model'],
            })

    if not records:
        print('No component-layer scores available for plotting.')
        return

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    width = 0.32
    gap_between_families = 1.2
    pos, heights, errs, bar_colors, labels, hatches = [], [], [], [], [], []
    x = 0.0
    family_centers = []

    for fam_key, title, abbrev in families:
        fam_records = [r for r in records if r['family'] == fam_key]
        if not fam_records:
            continue
        fam_records.sort(key=lambda r: r['component'])
        start_idx = len(pos)
        for rec in fam_records:
            pos.append(x)
            heights.append(rec['metric_mean'])
            errs.append(rec['metric_std'])
            bar_colors.append(colors.get(rec['model'], '#777777'))
            labels.append(f"{rec['abbr']}{rec['component']}")
            hatches.append(_model_hatch(rec['model']))
            x += width
        end_idx = len(pos) - 1
        family_centers.append((pos[start_idx] + pos[end_idx]) / 2)
        x += gap_between_families

    bars = ax.bar(pos, heights, width=width * 0.9, color=bar_colors, edgecolor='black', linewidth=1.2)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.errorbar(pos, heights, yerr=errs, fmt='none', ecolor='black', elinewidth=2.0, capsize=0, zorder=5)

    xtick_pos = [p for p, lbl in zip(pos, labels) if lbl.endswith('1') or lbl.endswith('10')]
    xtick_lab = ['1' if lbl.endswith('1') else '10' for lbl in labels if lbl.endswith('1') or lbl.endswith('10')]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, rotation=45, ha='right')
    ax.set_ylabel('Encoding (r)')
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel('Components')

    family_names = []
    for fam_key, title, _abbr in families:
        if any(r['family'] == fam_key for r in records):
            family_names.append(title)
    if family_centers:
        for center, name in zip(family_centers, family_names):
            ax.text(center, 1.02, name, ha='center', va='bottom', fontsize=11, transform=ax.get_xaxis_transform())

    style_axes(ax); format_axes(ax, axis='y', precision=3)
    fig.tight_layout(pad=0.4)

    handles = []
    for m in models:
        if m not in colors:
            continue
        rect = plt.Rectangle((0, 0), 1, 1, color=colors[m], ec='black', lw=1.0, label=m)
        hatch = _model_hatch(m)
        if hatch:
            rect.set_hatch(hatch)
        handles.append(rect)
    if handles:
        legend_fig = plt.figure(figsize=(3, 3.5))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.legend(handles=handles, frameon=False, loc='center', ncol=1)
        legend_ax.axis('off')
        legend_dir = FIG_DIR / 'legend'
        legend_dir.mkdir(parents=True, exist_ok=True)
        legend_path = legend_dir / 'top_components_combined_legend.pdf'
        legend_fig.savefig(legend_path, bbox_inches='tight', dpi=600)
        plt.close(legend_fig)
        print(f'Saved legend: {legend_path}')

    out_path = FIG_DIR / 'top_components_combined.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    best = compute_best_layers()
    if best.empty:
        print('No component-layer scores available.')
        return
    best = best.sort_values('metric_mean', ascending=False).drop_duplicates(['family', 'component'])
    plot_top(best)


if __name__ == '__main__':
    main()
