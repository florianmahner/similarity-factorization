#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

_here = os.path.dirname(__file__)
for _u in (1,2,3,4):
    _cand = os.path.abspath(os.path.join(_here, *(['..']*_u)))
    if os.path.exists(os.path.join(_cand, 'config')):
        sys.path.append(_cand) if _cand not in sys.path else None
        break

from config.paths import config
from model.feature.core.feature_io import (
    load_vis, load_beh, load_sem,
    load_cca_components, load_admm_components
)
from model.feature.core.ridge_utils import ridge_cv_fold, ridge_univariate_vec_icv, get_ridge_params
from joblib import Parallel, delayed


# Select which feature kinds to include ('visual','behavioral','semantic')
FEATURE_KINDS = {'visual','behavioral','semantic'}

def _get_comps_and_stims(domain: str, family: str):
    if domain == 'cca':
        fam_map = {'cross-species': 'all', 'human': 'human', 'monkey': 'monkey'}
        key = fam_map.get(family, 'all')
        comps, stims = load_cca_components(key)
    else:
        # ADMM families are 'all' | 'human' | 'monkey'
        comps, stims = load_admm_components(family)
    return comps, stims


def compute_enc_scores(domain='cca', families=None):
    if families is None:
        families = ['cross-species', 'human', 'monkey'] if domain == 'cca' else ['all', 'human', 'monkey']

    for family in families:
        try:
            comps, stims = _get_comps_and_stims(domain, family)
        except Exception:
            continue

        # Load feature matrices aligned to stims order
        X_vis, names_vis, _ = load_vis(stims)
        X_beh, names_beh, _ = load_beh(stims)
        X_sem, names_sem, _ = load_sem(stims)

        n_comps = comps.shape[1]
        n_jobs = int(config.analysis.get('n_jobs', 4))
        n_vis, n_beh, n_sem = X_vis.shape[1], X_beh.shape[1], X_sem.shape[1]
        mats, nams, grps = [], [], []
        if 'visual' in FEATURE_KINDS and n_vis>0:
            mats.append(X_vis); nams += names_vis; grps += ['visual']*n_vis
        if 'behavioral' in FEATURE_KINDS and n_beh>0:
            mats.append(X_beh); nams += names_beh; grps += ['behavioral']*n_beh
        if 'semantic' in FEATURE_KINDS and n_sem>0:
            mats.append(X_sem); nams += names_sem; grps += ['semantic']*n_sem
        X_all = np.hstack(mats) if mats else np.zeros((len(stims),0))
        names_all = nams
        groups_all = grps
        col_names = [f"{g}::{n}" for g,n in zip(groups_all, names_all)]
        M = np.zeros((n_comps, len(col_names)), float)
        S = np.zeros((n_comps, len(col_names)), float)
        out_dir = config.results_dir / 'feature' / 'component' / domain / family
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ridge_component] {domain}/{family}: comps={n_comps} vis={n_vis} beh={n_beh} sem={n_sem}")

        # drop-out ΔR² helper (joint over all features)
        def drop_deltas(X, y, k=10):
            if X.shape[1] == 0:
                return np.zeros(0), np.zeros(0)
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge, RidgeCV
            alphas, _ = get_ridge_params()
            kf = KFold(n_splits=int(k), shuffle=True, random_state=42)
            p = X.shape[1]
            diffs = []
            for tr, te in kf.split(X):
                xs = StandardScaler().fit(X[tr])
                Xtr = xs.transform(X[tr]); Xte = xs.transform(X[te])
                ytr = y[tr]; yte = y[te]
                a = RidgeCV(alphas=alphas, cv=3).fit(Xtr, ytr).alpha_
                base = Ridge(alpha=a).fit(Xtr, ytr).score(Xte, yte)
                def _drop_j(j):
                    Xtr_r = np.delete(Xtr, j, axis=1)
                    Xte_r = np.delete(Xte, j, axis=1)
                    r = Ridge(alpha=a).fit(Xtr_r, ytr).score(Xte_r, yte)
                    return float(base - r)
                d = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_drop_j)(j) for j in range(p))
                diffs.append(d)
            D = np.asarray(diffs, float)  # (k, p)
            return D.mean(axis=0), D.std(axis=0, ddof=1)
        for comp_idx in range(n_comps):
            y = comps[:, comp_idx]

            res = {'visual': {}, 'behavioral': {}, 'semantic': {}}

            m_all, s_all = drop_deltas(X_all, y, k=10)
            M[comp_idx, :] = m_all
            S[comp_idx, :] = s_all

            # fill per-group dict for backward-compatible pkl
            idx = 0
            if 'visual' in FEATURE_KINDS and n_vis>0:
                for nm, m, s in zip(names_vis, m_all[idx:idx+n_vis], s_all[idx:idx+n_vis]):
                    res['visual'][nm] = {'mean': float(m), 'std': float(s)}
                idx += n_vis
            if 'behavioral' in FEATURE_KINDS and n_beh>0:
                for nm, m, s in zip(names_beh, m_all[idx:idx+n_beh], s_all[idx:idx+n_beh]):
                    res['behavioral'][nm] = {'mean': float(m), 'std': float(s)}
                idx += n_beh
            if 'semantic' in FEATURE_KINDS and n_sem>0:
                for nm, m, s in zip(names_sem, m_all[idx:idx+n_sem], s_all[idx:idx+n_sem]):
                    res['semantic'][nm] = {'mean': float(m), 'std': float(s)}

            with open(out_dir / f'comp_{comp_idx+1}.pkl', 'wb') as f:
                pickle.dump(res, f)
            print(f"[ridge_component] {domain}/{family}: comp {comp_idx+1}/{n_comps} done")

        # Write unified per-family matrices (drop-out ΔR²)
        pd.DataFrame(M, columns=col_names).to_csv(out_dir / 'r2_drop_mean.csv', index=False)
        pd.DataFrame(S, columns=col_names).to_csv(out_dir / 'r2_drop_std.csv', index=False)

if __name__ == '__main__': 
    compute_enc_scores('cca')
