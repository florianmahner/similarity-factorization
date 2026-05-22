#!/usr/bin/env python3
"""
Behavioral body-part encoding of species-specific SymNMF components (SFig. 6).

Mirrors the Fig. 3 ridge bar style: univariate ridge CV (R²) of the body-part
SPoSE dimension against ordered human/monkey SymNMF components (ordered as in
gallery.py). Highlights the top-ranked human-only component.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.paths import config
from functions.plotting import setup_style, style_axes, format_axes, wide_roomy_figure
from model.feature.core.feature_io import load_beh
from model.feature.core.ridge_utils import get_ridge_params
from nmf.utils import load_admm_results, pick_rank, load_guard_table, ensure_dir
from nmf.gallery import GALLERY_SPECS

FMT = config.plotting.get('savefig_format', 'pdf')
OUT_DIR = ensure_dir(config.fig_dir / 'nmf_analysis')
N_FOLDS = 10
BASE_GREY = '#b3b3b3'
HUM_COLOR = config.plotting.get('human_color', '#7c5799')


def _find_bodypart_idx(names, key='body part'):
    key = key.lower()
    for i, name in enumerate(names):
        if key in name.lower():
            return i
    return None


def load_bodypart_scores(stims):
    X, names, _ = load_beh(stims)
    if X.size == 0 or not names:
        raise RuntimeError('behavioral embedding missing or empty')
    idx = _find_bodypart_idx(names)
    if idx is None:
        raise RuntimeError('body-part dimension not found in behavioral labels')
    return np.asarray(X[:, idx], float), names[idx]


def order_components(family, guard_df, n_components):
    key = 'human' if family == 'human' else 'monkey'
    spec = GALLERY_SPECS[key]
    if guard_df is None or guard_df.empty:
        return list(range(1, n_components + 1))
    df = guard_df.copy()
    df = df[spec['filter'](df)]
    if df.empty:
        return list(range(1, n_components + 1))
    df['score'] = df.apply(spec['score'], axis=1)
    df = df.sort_values('score', ascending=False)
    return df['component'].astype(int).tolist()


def ridge_body_to_components(body, W, comp_ids, kf):
    X = np.asarray(body, float).reshape(-1, 1)
    valid = [c for c in comp_ids if 1 <= c <= W.shape[1]]
    alphas, _ = get_ridge_params()

    r2_folds = np.zeros((len(valid), N_FOLDS))
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        xs = StandardScaler().fit(X[tr])
        Xtr = xs.transform(X[tr]).ravel()
        Xte = xs.transform(X[te]).ravel()
        for j, cid in enumerate(valid):
            ci = cid - 1
            ytr = W[tr, ci]
            yte = W[te, ci]
            alpha = RidgeCV(alphas=alphas, cv=3).fit(Xtr.reshape(-1, 1), ytr).alpha_
            r2 = Ridge(alpha=alpha).fit(Xtr.reshape(-1, 1), ytr).score(Xte.reshape(-1, 1), yte)
            r2_folds[j, fold_idx] = r2

    means = r2_folds.mean(axis=1)
    stds = r2_folds.std(axis=1)
    return valid, means, stds, r2_folds


def plot_family(family, comp_ids, means, stds, highlight=None, label='Body part'):
    setup_style()
    fig = wide_roomy_figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(comp_ids) + 1)
    colors = [BASE_GREY] * len(comp_ids)
    if highlight is not None and highlight in comp_ids:
        colors[comp_ids.index(highlight)] = HUM_COLOR

    edge_lw = 0.8 if family == 'monkey' else 1.2
    ax.bar(x, means, color=colors, edgecolor='black', linewidth=edge_lw, width=0.82)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black',
                elinewidth=edge_lw + 0.4, capsize=0, zorder=3)

    style_axes(ax)
    format_axes(ax, precision=3)
    ax.set_xlabel(f"{family.title()} SymNMF components (ranked)")
    ax.set_ylabel('Encoding (R²)')
    ticks = [1] + [t for t in range(5, len(comp_ids) + 1, 5)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 0.1)
    ax.set_title('Body part dimension Encoding')

    fname = f'sfig6_bodypart_{family}.{FMT}'
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'[body-encoding] saved {out}')


def summarize_stats(family, comp_ids, means, stds, folds):
    if len(comp_ids) <= 1:
        return
    target_idx = 0
    target_label = comp_ids[target_idx]
    target_mean = means[target_idx]
    target_std = stds[target_idx]
    target_folds = folds[target_idx]

    other = []
    p_vals = []
    for j in range(1, len(comp_ids)):
        t_stat, p_val = ttest_rel(target_folds, folds[j])
        other.append({'comp': comp_ids[j], 't': t_stat, 'p': p_val})
        p_vals.append(p_val)

    rejected, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    for k, info in enumerate(other):
        info['q'] = q_vals[k]
        info['sig'] = rejected[k] and info['t'] > 0

    n_sig = sum(1 for o in other if o['sig'])
    min_t = min(o['t'] for o in other if o['t'] > 0) if any(o['t'] > 0 for o in other) else np.nan
    max_q_sig = max((o['q'] for o in other if o['sig']), default=np.nan)

    print(f"[body-encoding] {family}: rank1 (comp {target_label}) R² = {target_mean:.3f} ± {target_std:.3f}")
    print(f"[body-encoding] {family}: rank1 vs others -> {n_sig}/{len(other)} higher at FDR<0.05; min t = {min_t:.3f}, max sig q = {max_q_sig:.3e}")


def summarize_joint(hum_ids, hum_means, hum_stds, hum_folds,
                    mon_ids, mon_means, mon_stds, mon_folds):
    if not hum_ids:
        return
    target_mean = hum_means[0]
    target_std = hum_stds[0]
    target_folds = hum_folds[0]
    target_label = hum_ids[0]

    others = []
    p_vals = []
    for cid, f in zip(hum_ids[1:], hum_folds[1:]):
        t_stat, p_val = ttest_rel(target_folds, f)
        others.append({'who': f'human {cid}', 't': t_stat, 'p': p_val})
        p_vals.append(p_val)
    for cid, f in zip(mon_ids, mon_folds):
        t_stat, p_val = ttest_rel(target_folds, f)
        others.append({'who': f'monkey {cid}', 't': t_stat, 'p': p_val})
        p_vals.append(p_val)

    if not others:
        return
    rejected, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    for k, info in enumerate(others):
        info['q'] = q_vals[k]
        info['sig'] = rejected[k] and info['t'] > 0

    n_sig = sum(1 for o in others if o['sig'])
    min_t = min(o['t'] for o in others if o['t'] > 0) if any(o['t'] > 0 for o in others) else np.nan
    max_q_sig = max((o['q'] for o in others if o['sig']), default=np.nan)

    print(f"[body-encoding] joint: human rank1 (comp {target_label}) R² = {target_mean:.3f} ± {target_std:.3f}")
    print(f"[body-encoding] joint: rank1 vs all other human+monkey comps -> {n_sig}/{len(others)} higher at FDR<0.05; min t = {min_t:.3f}, max sig q = {max_q_sig:.3e}")


def main():
    results, stims = load_admm_results()
    body, label = load_bodypart_scores(stims)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fam_data = {}

    for family in ('human', 'monkey'):
        if family not in results:
            print(f"[body-encoding] {family}: missing in ADMM results, skipping.")
            continue
        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        try:
            guard_df = load_guard_table(family)
        except FileNotFoundError:
            guard_df = None

        comp_order = order_components(family, guard_df, W.shape[1])
        comp_ids, means, stds, folds = ridge_body_to_components(body, W, comp_order, kf)
        highlight = comp_ids[0] if family == 'human' and comp_ids else None
        plot_family(family, comp_ids, means, stds, highlight=highlight, label=label)
        summarize_stats(family, comp_ids, means, stds, folds)
        fam_data[family] = (comp_ids, means, stds, folds)

    if 'human' in fam_data:
        hum_ids, hum_means, hum_stds, hum_folds = fam_data['human']
        mon_ids, mon_means, mon_stds, mon_folds = fam_data.get('monkey', ([], [], [], np.zeros((0, N_FOLDS))))
        summarize_joint(hum_ids, hum_means, hum_stds, hum_folds,
                        mon_ids, mon_means, mon_stds, mon_folds)


if __name__ == '__main__':
    main()
