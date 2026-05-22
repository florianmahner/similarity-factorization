#!/usr/bin/env python3
"""Build cross-/within-species CCA models for V1 and V4.

Outputs per-ROI pickle states to symbol/cca_state/ for downstream residualization.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up - 1]
    if (cand / 'config').exists():
        sys.path.append(str(cand))
        break

from config.paths import config
from functions.load import load_macq, load_mri, match_datasets
from functions.cca import fit_cca


PARAMS = config.analysis
CCA_CFG = config.hyperparameters['cca_families']
for fam in CCA_CFG.values():
    fam['reg'] = 10 ** fam['reg']

ROI_MAP = {
    'v1': {'monkey': 'v1', 'human': 'V1'},
    'v4': {'monkey': 'v4', 'human': 'hV4'},
}

OUT_DIR = Path(config.root_dir) / 'symbol' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_groups_for_human_view(view_name: str, stims: list[str]) -> np.ndarray | None:
    sid = view_name.split('_')[-1]
    meta_p = config.mri_dir / 'betas_csv' / f'sub-{sid}_StimulusMetadata.csv'
    if not meta_p.exists():
        return None
    df = pd.read_csv(meta_p)
    cols = {c.lower(): c for c in df.columns}
    if {'stimulus', 'session', 'run'} - set(cols):
        return None
    df['stim'] = df[cols['stimulus']].astype(str).str.replace('.jpg', '', regex=False)
    pos = {s: i for i, s in enumerate(df['stim'])}
    order = [pos.get(s, -1) for s in stims]
    labels = np.zeros(len(stims), dtype=int)
    ok = np.array(order) >= 0
    if ok.any():
        session = df[cols['session']].to_numpy()
        run = df[cols['run']].to_numpy()
        codes = (session * 100 + run).astype(int)
        labels[ok] = codes[np.array(order)[ok]]
    return labels


FAMILY_MAP = {
    'cross-species': 'cross-species',
    'human-only': 'human',
    'monkey-only': 'monkey',
}


def compute_families(data_all):
    matched, common = match_datasets(data_all)
    X_all = matched
    X_monkey = {k: matched[k] for k in matched if k.startswith('monkey_')}
    X_human = {k: matched[k] for k in matched if k.startswith('human_')}

    results = {}
    for name, views in (
        ('cross-species', X_all),
        ('human-only', X_human),
        ('monkey-only', X_monkey),
    ):
        if not views:
            continue
        cfg_key = FAMILY_MAP[name]
        cfg = CCA_CFG[cfg_key]
        demean = PARAMS.get('demean_groups', True)
        group_labels = {}
        if demean:
            for v in views:
                if v.startswith('human_'):
                    g = _build_groups_for_human_view(v, common)
                    if g is not None:
                        group_labels[v] = g
        if group_labels:
            res = fit_cca(views, reg=cfg['reg'], n_cc=cfg['n_cc'], group_labels=group_labels, demean_groups=True)
        else:
            res = fit_cca(views, reg=cfg['reg'], n_cc=cfg['n_cc'])
        results[name] = res

    return results, common


def build_roi_state(roi_key: str):
    roi_entry = ROI_MAP[roi_key]
    data_all = {}

    for m in PARAMS['monkey_ids']:
        X, st, _ = load_macq(
            monkey=m,
            roi=roi_entry['monkey'],
            min_reliab=PARAMS['default_reliability']
        )
        data_all[f'monkey_{m}'] = (X, st)

    for h in PARAMS['human_subjects']:
        X, st, _ = load_mri(
            sub=h,
            roi=roi_entry['human'],
            min_splithalf=PARAMS['default_split']
        )
        data_all[f'human_{h}'] = (X, st)

    results, stims = compute_families(data_all)
    state = {
        'cca_all': results.get('cross-species'),
        'cca_hum': results.get('human-only'),
        'cca_mon': results.get('monkey-only'),
        'all_stims': stims,
        'views': {k: v[0] for k, v in data_all.items()}
    }

    out_path = OUT_DIR / f'{roi_key}_cca_state.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(state, f)
    print(f'Saved {roi_key.upper()} CCA state → {out_path}')


def main():
    for roi_key in ('v1', 'v4'):
        print('\n' + '=' * 60)
        print(f'Building {roi_key.upper()} CCA state')
        print('=' * 60)
        build_roi_state(roi_key)


if __name__ == '__main__':
    main()
