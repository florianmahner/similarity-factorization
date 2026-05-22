#!/usr/bin/env python3
"""Decode IT components after residualizing early DNN features."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import pickle
import toml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
from scipy import stats

_here = Path(__file__).resolve()
for up in (1, 2, 3, 4):
    cand = _here.parents[up - 1]
    if (cand / 'config').exists():
        sys.path.append(str(cand))
        break

from config.paths import config
from functions.plotting import setup_style, format_axes, style_axes, wide_figure

# ======================================================================
# CONFIG
# ======================================================================
N_FOLDS = 10
N_PERMS = 100
RANDOM_STATE = 42
FEATURES_CORE = ['symbol', 'icon', 'numerosity', 'face', 'body_part']
RANDOM_FEATURE = 'random'
FEATURES_ALL = FEATURES_CORE + [RANDOM_FEATURE]
RANDOM_DRAWS = 100
RANDOM_POS_FRAC = 0.5

PCA_N = 'CCA'  # 'CCA' to match CCA dimensionality, or explicit int (e.g., 250)
RIDGE_ALPHA = 1.0
EARLY_DEPTH_THRESHOLD = 0.25

OUT_DIR = Path(config.root_dir) / 'symbol' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = config.cache_file
RESULT_FILE = 'decode_residualized_main.pkl'
SUPP_FILE = 'decode_residualized_nulls.pkl'

N_JOBS = min(6, max(1, int(config.analysis.get('n_jobs', 1))))  # Conservative: early DNN residualization is memory-heavy

# ======================================================================
# HELPERS
# ======================================================================

def load_state(state_path=STATE_PATH):
    with open(state_path, 'rb') as f:
        return pickle.load(f)

def load_annotations(stims):
    ann = pd.read_csv(config.things_dir / 'annotations.csv')
    ann['stim'] = ann['filename'].astype(str)
    ann = ann.set_index('stim').reindex(stims)
    return ann

def category_ids(stims):
    return np.array([str(s).split('_')[0] for s in stims])

def make_splits(y, groups, base_splits=N_FOLDS):
    y = np.asarray(y, dtype=int)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return []
    n_splits = min(base_splits, pos, neg)
    if n_splits < 2:
        return []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    idx = np.arange(len(y))
    return list(cv.split(idx, y, groups))

def avg_views(components, view_names, train_idx, test_idx):
    train_views, test_views = [], []
    for v in view_names:
        comp = components[v]
        scaler = StandardScaler().fit(comp[train_idx])
        train_views.append(scaler.transform(comp[train_idx]))
        test_views.append(scaler.transform(comp[test_idx]))
    train = np.mean(np.stack(train_views, axis=0), axis=0)
    test = np.mean(np.stack(test_views, axis=0), axis=0)
    return train, test

def apply_residualization(X_train, X_test, ctrl_train, ctrl_test):
    if ctrl_train.shape[1] == 0:
        return X_train, X_test
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(ctrl_train, X_train)
    return X_train - ridge.predict(ctrl_train), X_test - ridge.predict(ctrl_test)

def decode_species_cv(components, view_names, y, splits, residualizer=None, n_jobs=None):
    if not splits:
        return []
    y = np.asarray(y, dtype=int)
    max_jobs = min(N_JOBS, len(splits)) if n_jobs is None else min(max(1, n_jobs), len(splits))
    def run_fold(pair):
        train_idx, test_idx = pair
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        y_train, y_test = y[train_idx], y[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            return None
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return None

        X_train, X_test = avg_views(components, view_names, train_idx, test_idx)

        if residualizer is not None:
            ctrl_train, ctrl_test = residualizer(train_idx, test_idx)
            X_train, X_test = apply_residualization(X_train, X_test, ctrl_train, ctrl_test)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            cv=3,
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight='balanced',
            scoring='roc_auc'
        )
        clf.fit(X_train, y_train)

        y_pred = clf.decision_function(X_test)
        return roc_auc_score(y_test, y_pred)

    results = Parallel(n_jobs=max_jobs)(
        delayed(run_fold)(pair) for pair in splits
    )
    aucs = [r for r in results if r is not None]
    return aucs

def permutation_test(components, view_names, y, splits, residualizer=None, n_perms=N_PERMS, random_state=RANDOM_STATE):
    """Run permutation test for null distribution."""
    if not splits or n_perms <= 0:
        return np.array([])

    y = np.asarray(y, dtype=int)
    seeds = [random_state + i for i in range(n_perms)]
    n_jobs = min(max(1, int(config.analysis.get('n_jobs', 1))), n_perms)

    def run_perm(seed):
        rng = np.random.default_rng(seed)
        y_perm = rng.permutation(y)
        aucs = decode_species_cv(components, view_names, y_perm, splits, residualizer=residualizer, n_jobs=1)
        return np.mean(aucs) if aucs else np.nan

    perm_means = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_perm)(seed) for seed in seeds)
    perm_means = np.array([v for v in perm_means if np.isfinite(v)])
    return perm_means

def decode_all_features(components, view_names, labels, groups, residualizer=None, run_permutation=False, features=None):
    feats = FEATURES_CORE if features is None else list(features)
    stats = {}
    for feat in feats:
        y = labels[feat].values.astype(int)
        splits = make_splits(y, groups)
        if not splits:
            continue
        aucs = decode_species_cv(components, view_names, y, splits, residualizer=residualizer)
        if not aucs:
            continue
        mean_auc = float(np.mean(aucs))
        sem_auc = float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else 0.0
        
        # Permutation test if requested
        null_means = np.array([])
        p_val = np.nan
        if run_permutation:
            print(f"    Running permutation test for {feat}...")
            null_means = permutation_test(components, view_names, y, splits, residualizer=residualizer)
            if len(null_means) > 0:
                p_val = float((null_means >= mean_auc).sum() + 1) / (len(null_means) + 1)
        
        stats[feat] = {
            'fold_aucs': aucs,
            'mean_auc': mean_auc,
            'sem_auc': sem_auc,
            'n_folds': len(aucs),
            'null_means': null_means,
            'p_val': p_val
        }
    return {
        'features': stats,
        'n': len(labels)
    }


def generate_random_draws(n_samples, groups, n_draws=RANDOM_DRAWS, pos_frac=RANDOM_POS_FRAC, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    draws = []
    idx = np.arange(n_samples)
    pos_count = max(1, min(n_samples - 1, int(round(n_samples * pos_frac))))
    for _ in range(n_draws):
        y = np.zeros(n_samples, dtype=int)
        pos_idx = rng.choice(idx, size=pos_count, replace=False)
        y[pos_idx] = 1
        splits = make_splits(y, groups)
        if not splits:
            continue
        draws.append({
            'y': y,
            'splits': splits,
            'pos_idx': pos_idx
        })
    return draws


def random_baseline_species(components, view_names, random_draws, residualizer=None):
    if not random_draws:
        return {}
    fold_values = []
    draw_means = []
    draw_meta = []
    draw_fold_aucs = []
    first_draw = None

    for draw in random_draws:
        y = draw['y']
        splits = draw['splits']
        aucs = decode_species_cv(components, view_names, y, splits, residualizer=residualizer)
        if not aucs:
            continue
        if first_draw is None:
            first_draw = {'y': y, 'splits': splits, 'aucs': aucs}
        fold_values.extend(aucs)
        mean_auc = float(np.mean(aucs))
        draw_means.append(mean_auc)
        draw_meta.append({
            'mean_auc': mean_auc,
            'n_folds': len(aucs)
        })
        draw_fold_aucs.append(list(aucs))

    if not fold_values:
        return {}

    fold_values = np.asarray(fold_values, dtype=float)
    draw_means_arr = np.asarray(draw_means, dtype=float)
    mean_auc = float(np.mean(draw_means_arr))
    sem_auc = float(np.std(draw_means_arr, ddof=1) / np.sqrt(len(draw_means_arr))) if len(draw_means_arr) > 1 else 0.0

    null_means = np.array([])
    p_val = np.nan
    base_mean = mean_auc

    return {
        'fold_aucs': fold_values.tolist(),
        'mean_auc': mean_auc,
        'sem_auc': sem_auc,
        'null_means': null_means,
        'p_val': p_val,
        'n_draws': len(draw_meta),
        'draw_stats': draw_meta,
        'draw_means': draw_means_arr.tolist(),
        'draw_fold_aucs': draw_fold_aucs,
        'base_mean_for_perm': base_mean
    }

def mask_components(components, mask):
    if mask is None:
        return components
    return {v: comp[mask] for v, comp in components.items()}

# ======================================================================
# EARLY DNN LOADING
# ======================================================================

def load_early_dnn(stims):
    """Load early DNN layers (first 20%)."""
    base = Path(config.dnn_dir) / 'features'
    tsv = base / 'features_summary.tsv'
    if not tsv.exists():
        return None

    df = pd.read_csv(tsv, sep='\t')

    # Filter by config
    sel_p = Path(config.root_dir) / 'config' / 'dnn_layers.toml'
    sel = toml.load(sel_p)
    allow = {}
    for ent in sel.get('model', []):
        name = str(ent.get('name', '')).strip().lower()
        lays = {str(x) for x in ent.get('layers', [])}
        if name and lays:
            allow[name] = lays

    df = df[df.apply(lambda r: str(r['layer_name']) in allow.get(str(r['model']).strip().lower(), set()), axis=1)]

    # Normalize depth
    for model in df['model'].unique():
        mask = df['model'] == model
        max_seq = df.loc[mask, 'layer_seq'].max()
        df.loc[mask, 'depth_norm'] = df.loc[mask, 'layer_seq'] / max_seq

    df_early = df[df['depth_norm'] <= EARLY_DEPTH_THRESHOLD]

    print(f"  Early DNN layers:")
    for model in df_early['model'].unique():
        n_layers = len(df_early[df_early['model'] == model])
        print(f"    {model}: {n_layers} layers")

    # Load and concatenate
    all_feats = []
    for _, r in df_early.iterrows():
        p2 = Path(config.dnn_dir) / str(r['save_path']).strip()
        p2 = p2.parent / 'features' / 'features.npy'
        X = np.load(p2).astype(np.float32)
        all_feats.append(X)

    early_concat = np.hstack(all_feats)
    print(f"  Early DNN: {early_concat.shape}")
    return early_concat

def make_dnn_residualizer(X_ctrl_full, pca_n):
    def builder(train_idx, test_idx):
        ctrl_train = X_ctrl_full[train_idx]
        ctrl_test = X_ctrl_full[test_idx]

        scaler = StandardScaler().fit(ctrl_train)
        ctrl_train = scaler.transform(ctrl_train)
        ctrl_test = scaler.transform(ctrl_test)

        if ctrl_train.shape[1] > pca_n:
            n_pc = min(pca_n, ctrl_train.shape[0], ctrl_train.shape[1])
            pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
            ctrl_train = pca.fit_transform(ctrl_train)
            ctrl_test = pca.transform(ctrl_test)

        return ctrl_train, ctrl_test

    return builder

def resolve_pca_cap(pca_setting, components_matrix):
    if isinstance(pca_setting, str) and pca_setting.strip().upper() == 'CCA':
        return components_matrix.shape[1]
    try:
        val = int(pca_setting)
        return max(1, val)
    except Exception:
        return components_matrix.shape[1]

# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def run_residualized_analysis():
    """Run residualized decoding with per-species outputs."""
    print("Loading IT CCA state...")
    state = load_state()
    components_all = state['cca_all']['components']
    stims = np.array(state['all_stims'])

    print("\nLoading annotations...")
    labels_all = load_annotations(stims)
    groups_all = category_ids(stims)

    species_views = {
        'human': ['human_01', 'human_02', 'human_03'],
        'monkey': ['monkey_N', 'monkey_F']
    }

    per_species = {}

    # Early DNN residualization with permutation test
    print("\n" + "="*60)
    print("RESIDUALIZED DECODING (Early DNN removed) + NULL TEST")
    print("="*60)
    early_dnn = load_early_dnn(stims)
    if early_dnn is not None:
        mask_dnn = np.isfinite(early_dnn).all(axis=1)
        if not mask_dnn.all():
            print(f"  Early DNN usable stimuli: {mask_dnn.sum()} / {len(mask_dnn)}")
        early_use = early_dnn[mask_dnn]
        labels_dnn = labels_all.iloc[mask_dnn]
        groups_dnn = groups_all[mask_dnn]
        comps_dnn = mask_components(components_all, mask_dnn)
        first_view = next(iter(comps_dnn.values()))
        pca_cap = resolve_pca_cap(PCA_N, first_view)
        dnn_residualizer = make_dnn_residualizer(early_use, pca_cap)
        random_draws = generate_random_draws(len(labels_dnn), groups_dnn, RANDOM_DRAWS, RANDOM_POS_FRAC)

        for species, view_names in species_views.items():
            print(f"\n  {species.upper()}: residualized decoding with permutation test")
            comps = {v: comps_dnn[v] for v in view_names}
            residualized = decode_all_features(
                comps, view_names, labels_dnn, groups_dnn, 
                residualizer=dnn_residualizer, 
                run_permutation=True
            )
            if random_draws:
                print(f"    Random baseline ({len(random_draws)} draws)")
                rand_stats = random_baseline_species(
                    comps, view_names, random_draws, residualizer=dnn_residualizer
                )
                if rand_stats:
                    residualized['features'][RANDOM_FEATURE] = rand_stats
            per_species[species] = residualized
    else:
        print("  Early DNN features unavailable; skipping.")

    # Paired species comparison per feature
    paired = {}
    random_meta = []
    feats_common = sorted(set(per_species.get('human', {}).get('features', {})).intersection(
        per_species.get('monkey', {}).get('features', {})))
    for feat in feats_common:
        if feat == RANDOM_FEATURE:
            auc_h = per_species['human']['features'][feat].get('draw_means', [])
            auc_m = per_species['monkey']['features'][feat].get('draw_means', [])
        else:
            auc_h = per_species['human']['features'][feat]['fold_aucs']
            auc_m = per_species['monkey']['features'][feat]['fold_aucs']
        n_pairs = min(len(auc_h), len(auc_m))
        if n_pairs < 2:
            continue
        auc_h = np.asarray(auc_h[:n_pairs], dtype=float)
        auc_m = np.asarray(auc_m[:n_pairs], dtype=float)
        t_val, p_val = stats.ttest_rel(auc_m, auc_h, nan_policy='omit')
        mean_diff = float(np.nanmean(auc_m - auc_h))
        paired[feat] = {
            't_val': float(t_val),
            'p_val': float(p_val),
            'mean_diff': mean_diff,
            'n_pairs': int(np.isfinite(auc_h - auc_m).sum())
        }

    if 'random_draws' in locals() and random_draws:
        random_meta = [{'pos_idx': draw['pos_idx'].tolist()} for draw in random_draws]

    return {
        'per_species': per_species,
        'paired': paired,
        'random_draws': random_meta
    }

# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(results, out_dir):
    """Save residualization results to pickle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RESULT_FILE
    supp_path = out_dir / SUPP_FILE

    legacy_path = out_dir / 'decode_residualized_results.pkl'
    if legacy_path.exists():
        legacy_path.unlink()

    with open(result_path, 'wb') as f:
        pickle.dump(results, f)

    # Supplementary payload retains per-species nulls
    supp_payload = {
        'per_species': results['per_species'],
        'random_draws': results.get('random_draws', [])
    }
    with open(supp_path, 'wb') as f:
        pickle.dump(supp_payload, f)

    print(f"\nSaved results: {result_path}")
    print(f"Saved supplementary nulls: {supp_path}")
    return result_path

# ======================================================================
# PLOTTING (DEPRECATED - use viz_decoding.py)
# ======================================================================

def get_feature_colors():
    """Get colors for features: pink for math, bone grays for face/body."""
    try:
        pink_cmap = plt.colormaps.get_cmap('pink')
        bone_cmap = plt.colormaps.get_cmap('bone')
    except:
        pink_cmap = cm.get_cmap('pink')
        bone_cmap = cm.get_cmap('bone')

    return {
        'symbol': pink_cmap(0.20),
        'icon': pink_cmap(0.50),
        'numerosity': pink_cmap(0.80),
        'face': bone_cmap(0.60),
        'body_part': bone_cmap(0.40)
    }

def plot_residualized_results(results_species, species):
    """Plot per-species residualized decoding summary."""
    method_names = []
    method_data = []

    if 'Original' in results_species:
        method_names.append('Original')
        method_data.append(results_species['Original']['features'])

    for name in ('Early DNN',):
        entry = results_species.get(name)
        if entry and entry.get('residualized'):
            method_names.append(name)
            method_data.append(entry['residualized']['features'])

    if not method_names:
        print(f"No data to plot for {species}.")
        return

    feature_sets = [set(data.keys()) for data in method_data]
    features = [f for f in FEATURES_ALL if all(f in data for data in method_data)]
    if not features:
        print(f"No overlapping features for plotting ({species}).")
        return

    setup_style()
    fig = wide_figure()
    ax = fig.add_subplot(111)

    x_pos = np.arange(len(features))
    width = 0.8 / len(method_names)
    colors_feat = get_feature_colors()

    try:
        bone_cmap = plt.colormaps.get_cmap('bone')
    except:
        bone_cmap = cm.get_cmap('bone')

    gray_shades = [bone_cmap(0.35), bone_cmap(0.50), bone_cmap(0.65)]

    for i, (m_name, data) in enumerate(zip(method_names, method_data)):
        means = [data[f]['mean_auc'] for f in features]
        sems = [data[f]['sem_auc'] for f in features]

        if m_name == 'Original':
            bar_colors = [colors_feat[f] for f in features]
        else:
            bar_colors = [gray_shades[(i - 1) % len(gray_shades)]] * len(features)

        ax.bar(x_pos + i * width, means, width * 0.9,
               color=bar_colors, edgecolor='black', linewidth=1.2,
               label=m_name, zorder=2 if m_name == 'Original' else 1)
        ax.errorbar(x_pos + i * width, means, yerr=sems, fmt='none',
                    ecolor='black', elinewidth=2.0, capsize=0, zorder=3)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.set_xticks(x_pos + width * (len(method_names) - 1) / 2)
    ax.set_xticklabels([f.replace('_', '\n') for f in features])
    ax.set_title(f'{species.capitalize()} residualized decoding')
    ax.set_ylabel('AUC')
    ax.set_xlabel('Feature')
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    format_axes(ax, precision=3)

    fig.tight_layout(pad=0.4)
    out_path = OUT_DIR / f'decode_residualized_{species}.pdf'
    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    results = run_residualized_analysis()

    print("\n" + "="*60)
    print("SUMMARY: Residualized AUC vs Null")
    print("="*60)

    for species in ('human', 'monkey'):
        spec_res = results['per_species'].get(species, {})
        if not spec_res:
            continue
        print(f"\n[{species.upper()}]")
        features = spec_res.get('features', {})
        for feat in FEATURES_ALL:
            if feat in features:
                mean_auc = features[feat]['mean_auc']
                sem = features[feat]['sem_auc']
                p_val = features[feat]['p_val']
                if np.isfinite(p_val):
                    sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
                    print(f"  {feat:12s}: {mean_auc:.3f} ± {sem:.3f}  (p={p_val:.4f} {sig_str})")
                else:
                    print(f"  {feat:12s}: {mean_auc:.3f} ± {sem:.3f}  (p=nan)")

    if results['paired']:
        print("\n[MONKEY vs HUMAN]")
        for feat in FEATURES_ALL:
            comp = results['paired'].get(feat)
            if not comp:
                continue
            print(f"{feat:12s}: ΔAUC = {comp['mean_diff']:.3f}, t={comp['t_val']:.3f} (p={comp['p_val']:.4f})")

    if results.get('random_draws'):
        print(f"\nRandom baseline label sets: {len(results['random_draws'])}")

    save_results(results, OUT_DIR)
    print("\nDone. Use viz_decoding.py to visualize results.")

if __name__ == '__main__':
    main()
