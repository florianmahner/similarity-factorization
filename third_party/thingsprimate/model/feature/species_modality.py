#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break

from config.paths import config
from functions.plotting import setup_style, medium_panel, style_axes, panel_figure, narrow_figure
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
try:
    from scipy import stats
except Exception:
    stats = None
from model.feature.core.feature_io import load_cca_components, load_vis, load_sem
from model.feature.core.ridge_utils import get_ridge_params

# ---- Key choices ----
DOMAIN = 'cca'
FAMILIES = ['human', 'monkey']  # species-specific spaces
N_FOLDS = 10
SEED = 42
MAX_COMPONENTS = 30  # plotting crop only

# Local color scheme (distinct from global plotting defaults)
VISUAL_COLOR = '#55889E'
SEMANTIC_COLOR = '#C7522E'
SHARED_COLOR = '#B2996E'  # darker than #E1D3B8


def _family_color(fam: str):
    plot_cfg = getattr(config, 'plotting', {}) or {}
    return {
        'human': plot_cfg.get('human_color', '#7c5799'),
        'monkey': plot_cfg.get('monkey_color', '#bda855'),
        'cross-species': plot_cfg.get('shared_color', '#81B7B3'),
        'all': plot_cfg.get('shared_color', '#81B7B3'),
    }.get(fam, '#4c5b6b')


def _family_label(fam: str):
    return {
        'human': 'Human',
        'monkey': 'Monkey',
        'cross-species': 'Cross-species',
        'all': 'All',
    }.get(fam, fam)


def _cca_family_to_key(family: str) -> str:
    return {'cross-species': 'all', 'human': 'human', 'monkey': 'monkey'}.get(family, 'all')


def _load_comp_weights() -> dict:
    """Load held-out canonical correlations from crossview_results.pkl.

    Returns dict: {'cross-species': arr, 'human': arr, 'monkey': arr}.
    If file is missing, returns empty dict to trigger equal-weight fallback.
    """
    p = config.results_dir / 'crossview_results.pkl'
    if not p.exists():
        print(f"[species_modality] crossview_results not found at {p}; using equal weights")
        return {}
    import pickle
    with open(p, 'rb') as f:
        d = pickle.load(f)
    out = {}
    if 'cross-species' in d:
        out['cross-species'] = np.asarray(d['cross-species'].get('comp_corrs', []), float)
    if 'human_only' in d:
        out['human'] = np.asarray(d['human_only'].get('comp_corrs', []), float)
    if 'monkey_only' in d:
        out['monkey'] = np.asarray(d['monkey_only'].get('comp_corrs', []), float)
    return out


def _ridge_cv_metrics(X, y, n_folds=10, seed=42, alphas=None):
    alphas = np.asarray(alphas if alphas is not None else [0.001,0.01,0.1,1.0,10.0], float)
    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    vals_r2, vals_r = [], []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        xs = StandardScaler().fit(Xtr)
        Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
        rc = RidgeCV(alphas=alphas, cv=3)
        rc.fit(Xtr, ytr)
        yp = rc.predict(Xte)
        r2 = r2_score(yte, yp)
        vals_r2.append(float(r2))
        vals_r.append(float(np.sqrt(max(r2, 0.0))))
    vr2 = np.array(vals_r2, float)
    vr = np.array(vals_r, float)
    return (
        float(vr.mean()), float(vr.std()), vr.tolist(),
        float(vr2.mean()), float(vr2.std()), vr2.tolist()
    )


def _venn_for_family(family: str, n_folds=N_FOLDS, seed=SEED):
    key = _cca_family_to_key(family)
    Y, stims = load_cca_components(key)  # (n_stim, n_comp)
    Xv, _, _ = load_vis(stims)
    Xs, _, _ = load_sem(stims)
    if Xv.shape[1] == 0 or Xs.shape[1] == 0:
        print(f"[species_modality] {family}: missing visual or semantic features")
        return None

    alphas, _ = get_ridge_params()
    r_v = []; r_s = []; r_vs = []
    r2_v = []; r2_s = []; r2_vs = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xv, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_v.append(mr); r2_v.append(mr2)
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xs, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_s.append(mr); r2_s.append(mr2)
        Xvs = np.hstack([Xv, Xs])
        mr, _, _, mr2, _, _ = _ridge_cv_metrics(Xvs, y, n_folds=n_folds, seed=seed, alphas=alphas)
        r_vs.append(mr); r2_vs.append(mr2)
        if (k+1) % 10 == 0 or (k+1) == Y.shape[1]:
            print(f"[species_modality] {family}: comp {k+1}/{Y.shape[1]}")
    r_v = np.asarray(r_v, float)
    r_s = np.asarray(r_s, float)
    r_vs = np.asarray(r_vs, float)
    r2_v = np.asarray(r2_v, float)
    r2_s = np.asarray(r2_s, float)
    r2_vs = np.asarray(r2_vs, float)

    uniq_v = r2_vs - r2_s
    uniq_s = r2_vs - r2_v
    shared = (r2_v + r2_s) - r2_vs
    shared_r = np.sqrt(np.clip(shared, 0.0, None))

    # Weights from crossview held-out component correlations
    w_all = _load_comp_weights()
    w = np.asarray(w_all.get(family, []), float)
    if w.size == 0:
        w = np.ones(Y.shape[1], float)
    if w.size < Y.shape[1]:
        pad = np.ones(Y.shape[1] - w.size, float)
        w = np.concatenate([w, pad])
    w = np.maximum(w[:Y.shape[1]], 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    uv_r2 = float((w * uniq_v).sum())
    us_r2 = float((w * uniq_s).sum())
    sh_r2 = float((w * shared).sum())
    out = {
        'family': family,
        'n_components': int(Y.shape[1]),
        'weight_sum': float(w.sum()),
        'r_visual_weighted': float((w * r_v).sum()),
        'r_semantic_weighted': float((w * r_s).sum()),
        'r_joint_weighted': float((w * r_vs).sum()),
        'r2_visual_weighted': float((w * r2_v).sum()),
        'r2_semantic_weighted': float((w * r2_s).sum()),
        'r2_joint_weighted': float((w * r2_vs).sum()),
        'unique_visual_r2': uv_r2,
        'unique_semantic_r2': us_r2,
        'shared_r2': sh_r2,
        'unique_visual_r': float(np.sqrt(max(uv_r2, 0.0))),
        'unique_semantic_r': float(np.sqrt(max(us_r2, 0.0))),
        'shared_r': float(np.sqrt(max(sh_r2, 0.0))),
    }
    per_comp = pd.DataFrame({
        'component': np.arange(1, Y.shape[1]+1, dtype=int),
        'r_visual': r_v,
        'r_semantic': r_s,
        'r_shared': shared_r,
        'r_joint': r_vs,
        'r2_visual': r2_v,
        'r2_semantic': r2_s,
        'r2_shared': shared,
        'r2_joint': r2_vs,
    })
    return out, per_comp


def _plot_venn(vals: dict):
    setup_style()
    fig, ax = panel_figure(width='square')
    ax.set_axis_off()

    # Signed values for labels; non-negative for area (use r^2 for areas/labels)
    uv_signed = float(vals.get('unique_visual_r2', 0.0))
    us_signed = float(vals.get('unique_semantic_r2', 0.0))
    sh_signed = float(vals.get('shared_r2', 0.0))
    a = max(uv_signed, 0.0)
    b = max(us_signed, 0.0)
    ab = max(sh_signed, 0.0)

    v = venn2(subsets=(a, b, ab), set_labels=('Visual', 'Semantic'), ax=ax)

    if v.get_patch_by_id('10'):
        p = v.get_patch_by_id('10')
        p.set_facecolor(VISUAL_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)
    if v.get_patch_by_id('01'):
        p = v.get_patch_by_id('01')
        p.set_facecolor(SEMANTIC_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)
    if v.get_patch_by_id('11'):
        p = v.get_patch_by_id('11')
        p.set_facecolor(SHARED_COLOR)
        p.set_edgecolor('black')
        p.set_alpha(0.7)

    # Replace subset labels with signed values (3 decimals, r^2 units)
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(f"{uv_signed:.3f}")
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(f"{us_signed:.3f}")
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(f"{sh_signed:.3f}")
    # Tweak set labels
    if v.set_labels[0]: v.set_labels[0].set_color('black')
    if v.set_labels[1]: v.set_labels[1].set_color('black')

    ax.set_title(f"{_family_label(vals['family'])} weighted variance partition (r)", pad=2)

    out_dir = config.fig_dir / 'feature' / 'venn'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"venn_{vals['family']}.pdf"
    fig.savefig(out_p, bbox_inches='tight')
    plt.close(fig)
    print(f"[species_modality] saved {out_p}")


def _plot_component_lines(family: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    setup_style()
    fig = narrow_figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.78])

    if MAX_COMPONENTS and MAX_COMPONENTS > 0:
        df_plot = df.head(MAX_COMPONENTS)
    else:
        df_plot = df
    x = df_plot['component'].to_numpy()
    ax.plot(x, df_plot['r_visual'], color=VISUAL_COLOR, label='Visual', linewidth=3.4)
    ax.plot(x, df_plot['r_semantic'], color=SEMANTIC_COLOR, label='Semantic', linewidth=3.4)
    ax.plot(x, df_plot['r_shared'], color=SHARED_COLOR, label='Shared', linewidth=3.4)

    ax.set_xlabel('Component')
    ax.set_ylabel('Cross-validated partial r')
    ax.set_title(f"{_family_label(family)} component-wise fit", pad=4)
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, loc='upper right')
    style_axes(ax)

    out_dir = config.fig_dir / 'feature' / 'venn'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"component_traces_{family}.pdf"
    fig.savefig(out_p, bbox_inches='tight')
    plt.close(fig)
    print(f"[species_modality] saved {out_p}")


def _component_stats(per_comp: dict, summaries: dict, out_fig_dir):
    if 'human' not in per_comp or 'monkey' not in per_comp:
        return
    dh = per_comp['human']; dm = per_comp['monkey']
    if dh.empty or dm.empty:
        return
    # Align by component index
    n = int(min(dh.shape[0], dm.shape[0]))
    dh = dh.iloc[:n].copy(); dm = dm.iloc[:n].copy()
    d_h = dh['r_semantic'] - dh['r_visual']
    d_m = dm['r_semantic'] - dm['r_visual']
    inter = d_h - d_m

    def _summ(label, fam, d):
        v = d.to_numpy().astype(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {'label': label, 'family': fam, 'n_components': 0}
        tval = np.nan; pval = np.nan
        if stats is not None and v.size > 1:
            tval, pval = stats.ttest_1samp(v, 0.0)
        return {
            'label': label,
            'family': fam,
            'n_components': int(v.size),
            'mean_delta_r': float(v.mean()),
            'std_delta_r': float(v.std(ddof=1)) if v.size>1 else 0.0,
            't_value_vs_zero': float(tval) if np.isfinite(tval) else np.nan,
            'p_value_vs_zero': float(pval) if np.isfinite(pval) else np.nan,
        }

    rows = []
    # Weighted roll-ups (per full CCA space)
    for fam in ('human','monkey'):
        s = summaries.get(fam, {})
        if not s: continue
        rows.append({
            'label': f'{fam}_weighted',
            'family': fam,
            'n_components': int(s.get('n_components', 0)),
            'r_visual_weighted': float(s.get('r_visual_weighted', np.nan)),
            'r_semantic_weighted': float(s.get('r_semantic_weighted', np.nan)),
            'r_joint_weighted': float(s.get('r_joint_weighted', np.nan)),
            'unique_visual_r': float(s.get('unique_visual_r', np.nan)),
            'unique_semantic_r': float(s.get('unique_semantic_r', np.nan)),
            'shared_r': float(s.get('shared_r', np.nan)),
            'unique_visual_r2': float(s.get('unique_visual_r2', np.nan)),
            'unique_semantic_r2': float(s.get('unique_semantic_r2', np.nan)),
            'shared_r2': float(s.get('shared_r2', np.nan)),
            'unique_delta_r': float(s.get('unique_semantic_r', np.nan) - s.get('unique_visual_r', np.nan)),
            'unique_delta_r2': float(s.get('unique_semantic_r2', np.nan) - s.get('unique_visual_r2', np.nan)),
        })

    rows.append(_summ('human_sem_minus_vis', 'human', d_h))
    rows.append(_summ('monkey_sem_minus_vis', 'monkey', d_m))
    rows.append(_summ('human_minus_monkey_delta', 'human_minus_monkey', inter))

    out_dir = out_fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'species_modality_stats.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"[species_modality] wrote {p}")


if __name__ == '__main__':
    rows = []
    per_comp = {}
    summaries = {}
    for fam in FAMILIES:
        res = _venn_for_family(fam, n_folds=N_FOLDS, seed=SEED)
        if res is None:
            continue
        vals, df = res
        rows.append(vals)
        per_comp[fam] = df
        summaries[fam] = vals
        _plot_venn(vals)

    if rows:
        res_dir = config.results_dir / 'feature' / 'modality_components' / DOMAIN
        res_dir.mkdir(parents=True, exist_ok=True)
        p = res_dir / 'venn_weighted_variance.csv'
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"[species_modality] wrote {p}")
        fig_dir = config.fig_dir / 'feature' / 'venn'
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fam, df in per_comp.items():
            if df is None or df.empty:
                continue
            _plot_component_lines(fam, df)
            fp = res_dir / f'{fam}_component_traces.csv'
            df.to_csv(fp, index=False)
            print(f"[species_modality] wrote {fp}")
        _component_stats(per_comp, summaries, fig_dir)
