import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import sys, pickle, math
import toml
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Robust project import
_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up-1]
    if (cand / 'config').exists():
        if str(cand) not in sys.path:
            sys.path.append(str(cand))
        break
from config.paths import config


# ──────────────────────────────────────────────────────────────────────
# User-tunable (script-local) params

METHOD = 'ridge'  # 'rsm' or 'ridge' (alias: 'rsa' treated as 'rsm')
N_FOLDS = 5
SEED = 42
DNN_PCA_DIMS = 250   # PCA downsampling for ridge; None disables
SUMMARY_TSV = config.dnn_dir / 'features' / 'features_summary.tsv'
RESULTS_PKL = config.results_dir / 'dnn_rsa.pkl'
RESULTS_CSV = config.results_dir / 'dnn_rsa.csv'
RESULTS_BEST_CSV = config.results_dir / 'dnn_rsa_best.csv'

# Temporarily skip problematic models (e.g., CLIP has non-standard dims)
# Include all models by default (CLIP, DINOv2/v3, ResNet, VGG, AlexNet, ...)
EXCLUDE_MODELS = set()


# ──────────────────────────────────────────────────────────────────────
# Helpers

def load_layer_selection(fp: Path | None = None) -> dict | None:
    """Load model→allowed-layers from config/dnn_layers.toml; None if missing."""
    fp = fp or (config.root_dir / 'config' / 'dnn_layers.toml')
    if not fp.exists():
        return None
    data = toml.load(fp)
    out = {}
    for ent in data.get('model', []):
        name = str(ent.get('name','')).lower()
        lays = {str(x) for x in ent.get('layers', [])}
        if name and lays:
            out[name] = lays
    return out or None


def rowcorr_rsm(X: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation matrix (rows=stimuli, cols=features).

    Uses mean-centering + L2 normalization for exact unit diagonal.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Xc, axis=1, keepdims=True)
    eps = 1e-12
    denom[denom <= eps] = np.inf  # avoid div-by-zero; yields 0 rows
    Z = Xc / denom
    G = Z @ Z.T
    np.fill_diagonal(G, 1.0)
    np.clip(G, -1.0, 1.0, out=G)
    return G


def rsa_vec(rsm_a: np.ndarray, rsm_b: np.ndarray) -> float:
    i, j = np.triu_indices(rsm_a.shape[0], 1)
    va = rsm_a[i, j]
    vb = rsm_b[i, j]
    if va.size == 0:
        return np.nan
    r = np.corrcoef(va, vb)[0, 1]
    return float(r)


def cosine_rsm(X: np.ndarray) -> np.ndarray:
    """Cosine-similarity RSM over stimuli (rows are stimuli).

    Matches the similarity metric used in the ADMM RSM pipeline.
    """
    X = np.asarray(X, dtype=float)
    return cosine_similarity(X)


def fisher_mean_rsm_for_group(cca_group: dict, idx: np.ndarray | list[int]) -> np.ndarray:
    """Build per-view cosine RSMs on a stimulus subset and Fisher-z average them.

    - cca_group: dict with key 'components' mapping view_name -> (n_stim, n_cc)
    - idx: indices selecting the held-out stimuli (same for rows/cols)
    """
    comps_by_view = cca_group.get('components', {})
    views = sorted(comps_by_view.keys())
    if not views:
        raise ValueError('CCA group has no component views to build RSMs from')
    sims = []
    for v in views:
        C = np.asarray(comps_by_view[v])
        if C.ndim != 2:
            raise ValueError(f'Components for view {v} are not 2D: {C.shape}')
        Ci = C[idx]
        sims.append(cosine_similarity(Ci))
    sims = np.stack(sims, axis=0)
    sims = np.clip(sims, -0.9999, 0.9999)
    z = np.arctanh(sims)
    mean_z = z.mean(axis=0)
    mean_sim = np.tanh(mean_z)
    return mean_sim


def load_state() -> dict:
    fp = config.cache_file
    if not fp.exists():
        raise FileNotFoundError(f"CCA state file not found: {fp}")
    with open(fp, 'rb') as f:
        st = pickle.load(f)
    return st


def group_repr(cca_res: dict, views_key: str) -> np.ndarray:
    """Average canonical components across views for a single group."""
    comps = cca_res['components']  # dict view -> (n_stim, n_cc)
    mats = []
    for v in sorted(comps.keys()):
        if views_key in v:  # keep safety flexible
            mats.append(comps[v])
    if not mats:  # fallback: use all views
        mats = [comps[v] for v in sorted(comps.keys())]
    X = np.stack(mats, 0).mean(0)
    return X


def load_dnn_layer_rel(save_path: str) -> np.ndarray:
    """Load feature matrix from a repo-committed path layout.

    Expects paths like 'features/<model>/<layer>/features.npy' relative to config.dnn_dir.
    """
    p = Path(config.dnn_dir) / save_path
    if not p.exists():
        alt = p.parent / 'features' / p.name
        if alt.exists():
            p = alt
    return np.load(p)


def fold_rsa(X_dnn: np.ndarray, X_grp: np.ndarray, n_folds: int, seed: int) -> tuple[float, float, list]:
    rng = np.random.RandomState(seed)
    n = X_dnn.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    vals = []
    for f in folds:
        Xi = X_dnn[f]
        Yi = X_grp[f]
        r = rsa_vec(rowcorr_rsm(Xi), rowcorr_rsm(Yi))
        vals.append(r)
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std()), [float(x) for x in vals]


def fold_rsm(X_dnn: np.ndarray, cca_group: dict, n_folds: int, seed: int) -> tuple[float, float, list]:
    """Cross-validated RSA using ADMM-style, view-wise Fisher-z RSMs.

    - DNN side: cosine RSM on the held-out stimuli.
    - CCA side: per-view cosine RSMs on the same held-out stimuli, Fisher-z averaged across views.
    - Score: Pearson correlation between upper triangles of the two RSMs.
    """
    rng = np.random.RandomState(seed)
    n = X_dnn.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    vals = []
    for f in folds:
        Xi = X_dnn[f]
        Rd = cosine_rsm(Xi)
        Rc = fisher_mean_rsm_for_group(cca_group, f)
        r = rsa_vec(Rd, Rc)
        vals.append(r)
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std()), [float(x) for x in vals]


def fold_ridge(X: np.ndarray, Y: np.ndarray, n_folds: int, seed: int, alphas: list[float]) -> tuple[float, float, list, list]:
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    vals = []
    a_sel = []
    for f in folds:
        tr = np.setdiff1d(idx, f, assume_unique=False)
        Xt, Xv = X[tr], X[f]
        Yt, Yv = Y[tr], Y[f]
        # Standardize X and center Y on train; apply to val
        xs = StandardScaler(with_mean=True, with_std=True).fit(Xt)
        Xt = xs.transform(Xt); Xv = xs.transform(Xv)
        ys = StandardScaler(with_mean=True, with_std=False).fit(Yt)
        Yt = ys.transform(Yt); Yv = ys.transform(Yv)
        # Optional PCA compression on X to stabilize
        if DNN_PCA_DIMS and DNN_PCA_DIMS > 0 and Xt.shape[1] > 1:
            k = int(min(DNN_PCA_DIMS, max(1, Xt.shape[1]-1)))
            pca = PCA(n_components=k, svd_solver='randomized', random_state=seed)
            Xt = pca.fit_transform(Xt)
            Xv = pca.transform(Xv)
        # Inner CV on training split
        rc = RidgeCV(alphas=alphas, cv=3, scoring='r2')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress ill-conditioned warnings
            rc.fit(Xt, Yt)
        Yp = rc.predict(Xv)
        r2 = r2_score(Yv, Yp, multioutput='uniform_average')
        vals.append(float(r2))
        a_sel.append(float(rc.alpha_))
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std()), [float(x) for x in vals], [float(x) for x in a_sel]


def _task(model: str, layer: str, layer_seq: int, feat_path_rel: str,
          cca_hum_X: np.ndarray, cca_mon_X: np.ndarray,
          cca_hum_group: dict, cca_mon_group: dict,
          cca_stims: list, alphas: list[float]):
    # Load features by committed layout under data/dnn/features
    X = load_dnn_layer_rel(feat_path_rel)
    # Ensure rows are stimuli
    n_stim = len(cca_stims)
    if X.shape[0] == n_stim:
        pass
    elif X.shape[1] == n_stim:
        X = X.T
    else:
        raise ValueError(f"Feature matrix dims do not match n_stim={n_stim}: {tuple(X.shape)} for save_path={feat_path_rel}")
    # Commit to shared ordering between DNN extraction and CCA stimuli
    # (no per-layer stimulus remapping)
    # Normalize METHOD alias: treat 'rsa' as 'rsm'
    method = METHOD.lower().strip()
    if method == 'rsa':
        method = 'rsm'

    if method == 'ridge':
        hum_mean, hum_std, hum_folds, hum_alphas = fold_ridge(X, cca_hum_X, N_FOLDS, SEED, alphas)
        mon_mean, mon_std, mon_folds, mon_alphas = fold_ridge(X, cca_mon_X, N_FOLDS, SEED, alphas)
        # R computed as sqrt(max(R²_fold, 0)) for consistency with ridge_species.py
        hum_r_folds = [math.sqrt(max(x, 0.0)) for x in hum_folds]
        mon_r_folds = [math.sqrt(max(x, 0.0)) for x in mon_folds]
        hum_r_mean = np.mean(hum_r_folds)
        hum_r_std = np.std(hum_r_folds)
        mon_r_mean = np.mean(mon_r_folds)
        mon_r_std = np.std(mon_r_folds)

        human = {'r_mean': hum_r_mean, 'r_std': hum_r_std, 'r_folds': hum_r_folds, 'r2_mean': hum_mean, 'r2_std': hum_std, 'r2_folds': hum_folds, 'alpha_folds': hum_alphas}
        macaque = {'r_mean': mon_r_mean, 'r_std': mon_r_std, 'r_folds': mon_r_folds, 'r2_mean': mon_mean, 'r2_std': mon_std, 'r2_folds': mon_folds, 'alpha_folds': mon_alphas}
    else:
        # RSM path: ADMM-style view-wise Fisher-z RSMs
        hum_mean, hum_std, hum_folds = fold_rsm(X, cca_hum_group, N_FOLDS, SEED)
        mon_mean, mon_std, mon_folds = fold_rsm(X, cca_mon_group, N_FOLDS, SEED)
        human = {'r_full': hum_mean, 'r_mean': hum_mean, 'r_std': hum_std, 'r_folds': hum_folds}
        macaque = {'r_full': mon_mean, 'r_mean': mon_mean, 'r_std': mon_std, 'r_folds': mon_folds}
    depth = float(layer_seq)
    return {
        'model': model,
        'layer': layer,
        'layer_seq': int(layer_seq),
        'n_stim': int(X.shape[0]),
        'human': human,
        'macaque': macaque,
        'depth': depth,
    }


def main():
    os.makedirs(config.results_dir, exist_ok=True)
    st = load_state()
    cca_stims = st['all_stims']

    # Build group representations (average across respective views)
    # We use flexible containment to select views; then fall back to all.
    cca_hum_X = group_repr(st['cca_hum'], views_key='human')
    cca_mon_X = group_repr(st['cca_mon'], views_key='monkey')

    # Summary table
    summ = pd.read_csv(SUMMARY_TSV, sep='\t') if SUMMARY_TSV.suffix == '.tsv' else pd.read_csv(SUMMARY_TSV)
    # Keep minimal columns
    req_cols = ['model', 'layer_name', 'layer_seq', 'save_path', 'n_images']
    for c in req_cols:
        if c not in summ.columns:
            raise ValueError(f"Missing column in summary TSV: {c}")

    # Drop excluded models (case-insensitive exact name match)
    if EXCLUDE_MODELS:
        excl = {m.lower() for m in EXCLUDE_MODELS}
        mask = ~summ['model'].astype(str).str.lower().isin(excl)
        summ = summ[mask]

    # Optional: restrict to configured layers per model
    layer_sel = load_layer_selection()
    if layer_sel:
        def _allow(row):
            m = str(row['model']).lower()
            return (m in layer_sel) and (str(row['layer_name']) in layer_sel[m])
        summ = summ[summ.apply(_allow, axis=1)]
    # Natural sort by model, then layer_seq
    summ = summ.sort_values(['model', 'layer_seq']).reset_index(drop=True)

    tasks = []
    for _, r in summ.iterrows():
        tasks.append((r['model'], r['layer_name'], int(r['layer_seq']), str(r['save_path'])))

    n_jobs = int(config.analysis.get('n_jobs', 1))
    n_jobs_eff = max(1, min(n_jobs, len(tasks)))
    par = Parallel(n_jobs=n_jobs_eff)
    # Ridge alpha grid from gridsearch settings (log10 range)
    gs = config.hyperparameters.get('gridsearch', {})
    if {'reg_start','reg_stop','reg_num'}.issubset(gs.keys()):
        start, stop, num = float(gs['reg_start']), float(gs['reg_stop']), int(gs['reg_num'])
        alpha_grid = list(np.logspace(start, stop, num, base=10.0))
    else:
        alpha_grid = config.hyperparameters.get('ridge', {}).get('alphas', [0.001, 0.01, 0.1, 1.0, 10.0])
    # Pass both group matrices (for ridge) and full CCA groups (for RSM)
    cca_hum_group = st['cca_hum']
    cca_mon_group = st['cca_mon']

    completed = 0
    def _task_with_progress(model, layer, layer_seq, feat_path_rel, cca_hum_X, cca_mon_X, cca_hum_group, cca_mon_group, cca_stims, alphas):
        nonlocal completed
        result = _task(model, layer, layer_seq, feat_path_rel, cca_hum_X, cca_mon_X, cca_hum_group, cca_mon_group, cca_stims, alphas)
        completed += 1
        if completed % 20 == 0 or completed == len(tasks):
            print(f"[dnn_feature] {completed}/{len(tasks)} layers")
        return result

    outs = par(delayed(_task_with_progress)(m, l, s, p, cca_hum_X, cca_mon_X, cca_hum_group, cca_mon_group, cca_stims, alpha_grid)
               for (m, l, s, p) in tasks)

    # Pack per-layer
    per_layer = {}
    for o in outs:
        per_layer.setdefault(o['model'], {})[o['layer']] = {
            'depth': o['depth'], 'n_stim': o['n_stim'],
            'human': o['human'], 'macaque': o['macaque']
        }

    # Helper to extract comparable score for best selection
    def best_score(v: dict, grp: str) -> float:
        g = v[grp]
        if METHOD.lower().strip() == 'ridge':
            return float(g.get('r2_mean', -np.inf))
        return float(g.get('r_mean', -np.inf))

    # Best per model per group
    best = {'human': {}, 'macaque': {}}
    for model, layers in per_layer.items():
        # Human
        items = [(lay, best_score(v, 'human'), v['human'].get('r2_std', v['human'].get('r_std', 0.0)), v['depth']) for lay, v in layers.items()]
        if items:
            lay, r, s, d = max(items, key=lambda t: t[1])
            best['human'][model] = {'layer': lay, 'r': float(r), 'std': float(s), 'depth': float(d)}
        # Macaque
        items = [(lay, best_score(v, 'macaque'), v['macaque'].get('r2_std', v['macaque'].get('r_std', 0.0)), v['depth']) for lay, v in layers.items()]
        if items:
            lay, r, s, d = max(items, key=lambda t: t[1])
            best['macaque'][model] = {'layer': lay, 'r': float(r), 'std': float(s), 'depth': float(d)}

    res = {
        'meta': {
            'n_jobs': n_jobs,
            'n_folds': N_FOLDS,
            'seed': SEED,
            'method': ('rsm' if METHOD.lower().strip() == 'rsa' else METHOD.lower().strip()),
        },
        'per_layer': per_layer,
        'best': best,
    }

    # Save pkl
    with open(RESULTS_PKL, 'wb') as f:
        pickle.dump(res, f)
    print(f"Saved: {RESULTS_PKL}")

    # Save long CSV (per-layer)
    rows = []
    for model, layers in per_layer.items():
        for layer, v in layers.items():
            for grp in ('macaque', 'human'):
                g = v[grp]
                if METHOD == 'ridge':
                    r_full = None
                    r_mean = None
                    r_std = None
                    r2_mean = g.get('r2_mean', np.nan)
                    r2_std = g.get('r2_std', np.nan)
                else:
                    r_full = g.get('r_full', np.nan)
                    r_mean = g.get('r_mean', np.nan)
                    r_std = g.get('r_std', np.nan)
                    r2_mean = r_mean*r_mean if np.isfinite(r_mean) else np.nan
                    r2_std = abs(2*r_mean)*r_std if (np.isfinite(r_mean) and np.isfinite(r_std)) else np.nan
                rows.append({
                    'model': model, 'layer': layer, 'group': grp,
                    'r_full': r_full, 'r_mean': r_mean, 'r_std': r_std, 'r2_mean': r2_mean, 'r2_std': r2_std,
                    'n_stim': v['n_stim'], 'depth': v['depth'], 'is_best': int(best[grp].get(model, {}).get('layer') == layer)
                })
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"Saved: {RESULTS_CSV}")

    # Save compact best CSV
    brow = []
    for grp in ('macaque', 'human'):
        for model, g in best[grp].items():
            brow.append({'group': grp, 'model': model, 'layer': g['layer'], 'r': g['r'], 'std': g['std'], 'depth': g['depth']})
    pd.DataFrame(brow).to_csv(RESULTS_BEST_CSV, index=False)
    print(f"Saved: {RESULTS_BEST_CSV}")


if __name__ == '__main__':
    main()
