#!/usr/bin/env python3
import os, sys, math, toml
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

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
from model.feature.core.feature_io import load_cca_components
from model.feature.core.ridge_utils import get_ridge_params


def _load_layer_entries(stims):
    base = Path(config.dnn_dir) / 'features'
    tsv = base / 'features_summary.tsv'
    if not tsv.exists():
        return []
    df = pd.read_csv(tsv, sep='\t')
    sel_p = Path(config.root_dir) / 'config' / 'dnn_layers.toml'
    sel = toml.load(sel_p) if sel_p.exists() else {}
    allow = {}
    for ent in sel.get('model', []):
        name = str(ent.get('name', '')).strip().lower()
        layers = {str(x) for x in ent.get('layers', [])}
        if name and layers:
            allow[name] = layers
    def _keep(row):
        m = str(row['model']).strip().lower()
        l = str(row['layer_name']).strip()
        return not allow or l in allow.get(m, set())
    df = df[df.apply(_keep, axis=1)]
    out = []
    for _, r in df.iterrows():
        model = str(r['model']).strip()
        layer = str(r['layer_name']).strip()
        save_path = Path(config.dnn_dir) / str(r['save_path']).strip()
        feat_path = save_path.parent / 'features' / 'features.npy'
        out.append((f"{model}_{layer}", feat_path))
    return out


def _ridge_layer(layer_key, feat_path, comps, alphas, pca_n, n_folds, seed):
    X = np.load(feat_path).astype(np.float32, copy=False)
    if X.shape[0] != comps.shape[0]:
        raise ValueError(f'{layer_key}: mismatched stimulus count')
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r_folds = []
    for tr, te in kf.split(X):
        xs = StandardScaler().fit(X[tr])
        Xtr = xs.transform(X[tr])
        Xte = xs.transform(X[te])
        if pca_n and Xtr.shape[1] > pca_n:
            k = min(int(pca_n), Xtr.shape[1])
            pca = PCA(n_components=k, svd_solver='randomized', random_state=seed)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        rc = RidgeCV(alphas=alphas, cv=3).fit(Xtr, comps[tr])
        Yp = rc.predict(Xte)
        vals = []
        for ci in range(comps.shape[1]):
            yt = comps[te, ci]
            yp = Yp[:, ci]
            yt_c = yt - yt.mean()
            yp_c = yp - yp.mean()
            num = float(np.dot(yt_c, yp_c))
            den = math.sqrt(float(np.dot(yt_c, yt_c)) * float(np.dot(yp_c, yp_c))) + 1e-12
            vals.append(abs(num / den) if den > 0 else 0.0)
        r_folds.append(vals)
    r_folds = np.asarray(r_folds, float)
    return layer_key, r_folds.mean(axis=0), r_folds.std(axis=0, ddof=1)


def compute_dnn_component_layers():
    out_dir = config.results_dir / 'feature' / 'species' / 'cca'
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas, _ = get_ridge_params()
    pca_n = int(config.hyperparameters.get('viz', {}).get('ridge_pca_n', 250))
    n_folds = 10
    seed = 42
    n_jobs = int(config.analysis.get('n_jobs', 4))

    all_rows = []
    best_rows = []

    fams = [('monkey', 'monkey'), ('human', 'human'), ('cross-species', 'all')]
    for fam_key, cca_key in fams:
        try:
            comps, stims = load_cca_components(cca_key)
        except Exception:
            continue
        comps = np.asarray(comps, dtype=np.float32)
        entries = _load_layer_entries(stims)
        if not entries:
            continue
        n_jobs_eff = max(1, min(n_jobs, len(entries)))
        jobs = Parallel(n_jobs=n_jobs_eff)(
            delayed(_ridge_layer)(key, path, comps, alphas, pca_n, n_folds, seed)
            for key, path in entries
        )
        for key, mu, sd in jobs:
            model = key.rsplit('_', 1)[0]
            for idx in range(comps.shape[1]):
                all_rows.append({
                    'family': fam_key,
                    'component': idx + 1,
                    'model': model,
                    'layer_key': key,
                    'metric_mean': float(mu[idx]),
                    'metric_std': float(sd[idx]),
                    'metric_type': 'r'
                })
        if all_rows:
            sub = [r for r in all_rows if r['family'] == fam_key]
            if sub:
                df_sub = pd.DataFrame(sub)
                df_sub = df_sub.sort_values('metric_mean', ascending=False).drop_duplicates(['component'])
                best_rows.extend(df_sub.to_dict('records'))

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / 'dnn_component_layer_scores.csv', index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_dir / 'dnn_component_layer_best.csv', index=False)


if __name__ == '__main__':
    compute_dnn_component_layers()
