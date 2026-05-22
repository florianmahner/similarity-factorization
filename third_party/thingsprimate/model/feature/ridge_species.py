#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys, pickle
import toml
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
from model.feature.core.ridge_utils import get_ridge_params
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

def load_dnn_features(stims):
    """Load per-layer DNN features via features_summary.tsv and collect depths from layer_seq.

    Only keep layers listed in config/dnn_layers.toml per model.
    """
    base = Path(config.dnn_dir) / 'features'
    tsv = base / 'features_summary.tsv'
    if not tsv.exists():
        return {}, {}
    n = len(stims)
    df = pd.read_csv(tsv, sep='\t')
    # filter by config/dnn_layers.toml
    sel_p = Path(config.root_dir) / 'config' / 'dnn_layers.toml'
    sel = toml.load(sel_p)
    allow = {}
    for ent in sel.get('model', []):
        name = str(ent.get('name','')).strip().lower()
        lays = {str(x) for x in ent.get('layers', [])}
        if name and lays:
            allow[name] = lays
    df = df[df.apply(lambda r: str(r['layer_name']) in allow.get(str(r['model']).strip().lower(), set()), axis=1)]
    out, depths = {}, {}
    for _, r in df.iterrows():
        model = str(r['model']).strip()
        layer = str(r['layer_name']).strip()
        p2 = Path(config.dnn_dir) / str(r['save_path']).strip()
        p2 = p2.parent / 'features' / 'features.npy'
        X = np.load(p2)
        k = f"{model}_{layer}"
        out[k] = X.astype(np.float32, copy=False)
        depths[k] = float(r['layer_seq'])
    return out, depths


def _get_comps_and_stims(domain: str, family: str):
    if domain == 'cca':
        fam_map = {'cross-species': 'all', 'human': 'human', 'monkey': 'monkey'}
        key = fam_map.get(family, 'all')
        comps, stims = load_cca_components(key)
    else:
        comps, stims = load_admm_components(family)
    return comps, stims


def compute_species_scores(domain='cca', families=None):
    if families is None:
        families = (['cross-species', 'human', 'monkey'] if domain == 'cca' else ['all', 'human', 'monkey'])

    for family in families:
        try:
            Y, stims = _get_comps_and_stims(domain, family)
        except Exception:
            continue

        X_vis, names_vis, _ = load_vis(stims)
        X_beh, names_beh, _ = load_beh(stims)
        X_sem, names_sem, _ = load_sem(stims)
        dnn_features, dnn_depths = load_dnn_features(stims)

        results = {}
        n_jobs = int(config.analysis.get('n_jobs', 4))
        print(f"[ridge_species] {domain}/{family}: Y={Y.shape} vis={X_vis.shape} beh={X_beh.shape} sem={X_sem.shape} dnn_layers={len(dnn_features)}")

        # ridge params
        alphas, _ = get_ridge_params()
        n_folds = 10
        seed = 42

        def ridge_cv(X, Y, pca_n=None):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            vals_r2, vals_r = [], []
            for tr, te in kf.split(X):
                Xtr, Xte = X[tr], X[te]
                Ytr, Yte = Y[tr], Y[te]
                xs = StandardScaler().fit(Xtr)
                Xtr = xs.transform(Xtr); Xte = xs.transform(Xte)
                if pca_n and Xtr.shape[1] > pca_n:
                    k = int(min(pca_n, Xtr.shape[1]))
                    pca = PCA(n_components=k, svd_solver='randomized', random_state=seed)
                    Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
                rc = RidgeCV(alphas=alphas, cv=3).fit(Xtr, Ytr)
                Yp = rc.predict(Xte)
                # per-output R² on the fold
                num = ((Yte - Yp)**2).sum(axis=0)
                den = ((Yte - Yte.mean(axis=0))**2).sum(axis=0) + 1e-12
                r2_vec = 1.0 - (num/den)
                fold_r2 = float(np.mean(r2_vec))  # may be negative
                fold_r = float(np.sqrt(max(fold_r2, 0.0)))  # on [0,1]
                vals_r2.append(fold_r2)
                vals_r.append(fold_r)
            vals_r2 = np.array(vals_r2, float)
            vals_r = np.array(vals_r, float)
            return (
                float(vals_r.mean()), float(vals_r.std()), [float(x) for x in vals_r],
                float(vals_r2.mean()), float(vals_r2.std()), [float(x) for x in vals_r2],
            )

        # Visual joint model
        if X_vis.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds = ridge_cv(X_vis, Y)
            results['visual'] = {'r_mean': mr, 'r_std': sr, 'r_folds': rfolds, 'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds, 'n_features': X_vis.shape[1]}

        # Behavioral joint model
        if X_beh.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds = ridge_cv(X_beh, Y)
            results['behavioral'] = {'r_mean': mr, 'r_std': sr, 'r_folds': rfolds, 'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds, 'n_features': X_beh.shape[1]}

        # Semantic joint model
        if X_sem.shape[1] > 0:
            mr, sr, rfolds, mr2, sr2, r2folds = ridge_cv(X_sem, Y)
            results['semantic'] = {'r_mean': mr, 'r_std': sr, 'r_folds': rfolds, 'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds, 'n_features': X_sem.shape[1]}

        # DNN layer models (with optional PCA)
        results['dnn'] = {}
        pca_n = int(config.hyperparameters.get('viz', {}).get('ridge_pca_n', 250))
        if dnn_features:
            items = list(dnn_features.items())
            completed = 0
            n_jobs_eff = max(1, min(n_jobs, len(items)))

            def _fit(pair):
                nonlocal completed
                k, Xd = pair
                if Xd.shape[0] != len(stims):
                    return k, None
                mr, sr, rfolds, mr2, sr2, r2folds = ridge_cv(Xd, Y, pca_n=pca_n)
                completed += 1
                if completed % 15 == 0 or completed == len(items):
                    print(f"[ridge_species] {domain}/{family}: dnn {completed}/{len(items)} layers")
                return k, (mr, sr, rfolds, mr2, sr2, r2folds, Xd.shape[1])
            outs = Parallel(n_jobs=n_jobs_eff)(delayed(_fit)(it) for it in items)
            for k, val in outs:
                if val is None: continue
                mr, sr, rfolds, mr2, sr2, r2folds, nf = val
                results['dnn'][k] = {'r_mean': mr, 'r_std': sr, 'r_folds': rfolds, 'r2_mean': mr2, 'r2_std': sr2, 'r2_folds': r2folds, 'n_features': nf}
            print(f"[ridge_species] {domain}/{family}: dnn {len(results['dnn'])}/{len(dnn_features)} layers done")

        # Save species results
        out_dir = config.results_dir / 'feature' / 'species' / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / f'{family}.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save legacy per-family CSV for backward compatibility
        rows = []
        for ftype in ['visual', 'behavioral', 'semantic']:
            if ftype in results:
                rows.append({'feature_type': ftype, 'r2_mean': results[ftype]['r2_mean'],
                           'r2_std': results[ftype]['r2_std'], 'n_features': results[ftype]['n_features']})
        for layer_key, layer_results in results.get('dnn', {}).items():
            rows.append({'feature_type': f'dnn_{layer_key}', 'r2_mean': layer_results['r2_mean'],
                       'r2_std': layer_results['r2_std'], 'n_features': layer_results['n_features']})
        pd.DataFrame(rows).to_csv(out_dir / f'{family}.csv', index=False)
        print(f"[ridge_species] {domain}/{family}: done")

        # Aggregate rows for new CSVs
        # Prepare per-fold wide columns
        def _fold_cols(folds, prefix='r2_f'):
            out = {}
            for i, v in enumerate(folds, 1):
                out[f'{prefix}{i}'] = v
            return out

        # Collect to buffers on function attribute to emit once after all families
        if not hasattr(compute_species_scores, '_buf_other'):
            compute_species_scores._buf_other = []
            compute_species_scores._buf_dnn = []

        for ftype in ['visual','behavioral','semantic']:
            if ftype in results:
                ent = results[ftype]
                row = {'family': family, 'feature_type': ftype, 'n_features': ent['n_features'],
                       'r_mean': ent['r_mean'], 'r_std': ent['r_std'], 'r2_mean': ent['r2_mean'], 'r2_std': ent['r2_std'], 'n_folds': n_folds}
                row.update(_fold_cols(ent.get('r_folds', []), 'r_f'))
                row.update(_fold_cols(ent.get('r2_folds', []), 'r2_f'))
                compute_species_scores._buf_other.append(row)

        for layer_key, ent in results.get('dnn', {}).items():
            model, layer = (str(layer_key).rsplit('_', 1) + [''])[:2]
            row = {'family': family, 'model': model, 'layer': layer, 'depth': float(dnn_depths.get(layer_key, np.nan)),
                   'n_features': ent['n_features'], 'r_mean': ent['r_mean'], 'r_std': ent['r_std'], 'r2_mean': ent['r2_mean'], 'r2_std': ent['r2_std'], 'n_folds': n_folds}
            row.update(_fold_cols(ent.get('r_folds', []), 'r_f'))
            row.update(_fold_cols(ent.get('r2_folds', []), 'r2_f'))
            compute_species_scores._buf_dnn.append(row)

if __name__ == '__main__':
    compute_species_scores('cca', families=['cross-species','human','monkey'])
    # Emit aggregated CSVs (CCA domain only)
    out_root = config.results_dir / 'feature' / 'species' / 'cca'
    out_root.mkdir(parents=True, exist_ok=True)
    if hasattr(compute_species_scores, '_buf_other') and compute_species_scores._buf_other:
        pd.DataFrame(compute_species_scores._buf_other).to_csv(out_root / 'other_features.csv', index=False)
        print(f"[ridge_species] wrote {out_root/'other_features.csv'}")
    if hasattr(compute_species_scores, '_buf_dnn') and compute_species_scores._buf_dnn:
        pd.DataFrame(compute_species_scores._buf_dnn).to_csv(out_root / 'dnn_features.csv', index=False)
        print(f"[ridge_species] wrote {out_root/'dnn_features.csv'}")
