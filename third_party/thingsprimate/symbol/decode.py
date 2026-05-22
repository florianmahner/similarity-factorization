#!/usr/bin/env python3
"""Decode symbol/icon/numerosity/face/body_part from cross-species CCA components.

Uses species-averaged projections (separate human/monkey averages) as predictors.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
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

STATE_PATH = config.cache_file
REGION_NAME = 'it'
OUT_DIR_BASE = Path(config.root_dir) / 'symbol' / 'results'
RESULT_FILE = 'decode_main.pkl'
SUPP_FILE = 'decode_nulls.pkl'


N_JOBS = max(1, int(config.analysis.get('n_jobs', 1)))

# ======================================================================
# HELPERS
# ======================================================================

def load_state(state_path):
    with open(state_path, 'rb') as f:
        return pickle.load(f)

def load_components(state_path):
    state = load_state(state_path)
    cca_cross = state['cca_all']
    stims = np.array(state['all_stims'])
    components = cca_cross['components']
    return stims, components

def load_annotations(stims):
    ann = pd.read_csv(config.things_dir / 'annotations.csv')
    ann['stim'] = ann['filename'].astype(str)
    ann = ann.set_index('stim').reindex(stims)
    return ann

def category_ids(stims):
    return np.array([s.split('_')[0] for s in stims])

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

def decode_species_cv(components, view_names, y, splits, n_jobs=None):
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

def permutation_test(components, view_names, y, splits, n_perms=N_PERMS, random_state=RANDOM_STATE):
    if not splits or n_perms <= 0:
        return np.array([])

    y = np.asarray(y, dtype=int)
    seeds = [random_state + i for i in range(n_perms)]
    n_jobs = min(max(1, int(config.analysis.get('n_jobs', 1))), n_perms)

    def run_perm(seed):
        rng = np.random.default_rng(seed)
        y_perm = rng.permutation(y)
        aucs = decode_species_cv(components, view_names, y_perm, splits, n_jobs=1)
        return np.mean(aucs) if aucs else np.nan

    perm_means = Parallel(n_jobs=n_jobs)(delayed(run_perm)(seed) for seed in seeds)
    perm_means = np.array([v for v in perm_means if np.isfinite(v)])
    return perm_means


def generate_random_draws(n_samples, groups, n_draws=RANDOM_DRAWS, pos_frac=RANDOM_POS_FRAC, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    draws = []
    idx = np.arange(n_samples)
    pos_count = max(1, min(n_samples - 1, int(round(n_samples * pos_frac))))
    for draw_idx in range(n_draws):
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


def random_baseline_species(components, view_names, random_draws):
    if not random_draws:
        return {}
    fold_values = []
    draw_means = []
    draw_data = []
    draw_fold_aucs = []
    first_draw = None
    for draw in random_draws:
        y = draw['y']
        splits = draw['splits']
        aucs = decode_species_cv(components, view_names, y, splits)
        if not aucs:
            continue
        if first_draw is None:
            first_draw = {'y': y, 'splits': splits, 'auc': aucs}
        fold_values.extend(aucs)
        mean_auc_draw = float(np.mean(aucs))
        draw_means.append(mean_auc_draw)
        draw_data.append({
            'mean_auc': mean_auc_draw,
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
        'n_draws': len(draw_data),
        'draw_stats': draw_data,
        'draw_means': draw_means_arr.tolist(),
        'draw_fold_aucs': draw_fold_aucs,
        'base_mean_for_perm': base_mean
    }

# ======================================================================
# ANALYSIS
# ======================================================================

def run_decoding_analysis(stims, components):
    """Run decoding separately for human and monkey species with paired comparisons."""
    print("\nLoading annotations...")
    labels = load_annotations(stims)
    groups = category_ids(stims)

    species_views = {
        'human': ['human_01', 'human_02', 'human_03'],
        'monkey': ['monkey_N', 'monkey_F']
    }

    per_species = {}
    paired = {}
    splits_cache = {}
    random_draws = None

    for species, view_names in species_views.items():
        print(f"\n=== {species.upper()} ===")
        species_res = {}
        for feat in FEATURES_CORE:
            y = labels[feat].values.astype(int)
            if feat not in splits_cache:
                splits_cache[feat] = make_splits(y, groups)
            splits = splits_cache[feat]
            if not splits:
                print(f"{feat}: skipping (insufficient positives)")
                continue

            n_pos = y.sum()
            print(f"{feat}: {n_pos} positives ({100*n_pos/len(y):.1f}%)")

            aucs_obs = decode_species_cv(components, view_names, y, splits)
            if not aucs_obs:
                print(f"  no valid folds")
                continue

            mean_auc = float(np.mean(aucs_obs))
            sem_auc = float(np.std(aucs_obs, ddof=1) / np.sqrt(len(aucs_obs))) if len(aucs_obs) > 1 else 0.0
            print(f"  observed: {mean_auc:.3f} ± {sem_auc:.3f}")

            null_means = permutation_test(components, view_names, y, splits)
            if null_means.size:
                p_val = (1 + np.sum(null_means >= mean_auc)) / (1 + len(null_means))
                print(f"  permutations: {len(null_means)} draws → p={p_val:.3f}")
            else:
                p_val = np.nan
                print("  permutations: skipped")

            species_res[feat] = {
                'fold_aucs': aucs_obs,
                'mean_auc': mean_auc,
                'sem_auc': sem_auc,
                'null_means': null_means,
                'p_val': p_val,
                'n_folds': len(aucs_obs)
            }

        if random_draws is None:
            random_draws = generate_random_draws(len(stims), groups, RANDOM_DRAWS, RANDOM_POS_FRAC)
        rand_stats = random_baseline_species(components, view_names, random_draws)
        if rand_stats:
            species_res[RANDOM_FEATURE] = rand_stats

        per_species[species] = species_res

    # Paired comparisons across species
    feats_common = sorted(set(per_species.get('human', {})).intersection(per_species.get('monkey', {})))
    for feat in feats_common:
        if feat == RANDOM_FEATURE:
            auc_h = per_species['human'][feat].get('draw_means', [])
            auc_m = per_species['monkey'][feat].get('draw_means', [])
        else:
            auc_h = per_species['human'][feat]['fold_aucs']
            auc_m = per_species['monkey'][feat]['fold_aucs']
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

    random_meta = [{'pos_idx': draw['pos_idx'].tolist()} for draw in (random_draws or [])]

    return {
        'per_species': per_species,
        'paired': paired,
        'random_draws': random_meta
    }

# ======================================================================
# MAIN
# ======================================================================

def run_region(state_path, region_name=REGION_NAME):
    """Compute decoding results and save to pickle."""
    print(f"Loading CCA state for {region_name.upper()}...")
    stims, components = load_components(state_path)
    results = run_decoding_analysis(stims, components)

    print("\n" + "="*60)
    print(f"SUMMARY · {region_name.upper()}")
    print("="*60)
    for species in ('human', 'monkey'):
        spec = results['per_species'].get(species)
        if not spec:
            continue
        print(f"\n[{species.upper()}]")
        for feat in FEATURES_ALL:
            r = spec.get(feat)
            if not r:
                continue
            p_val = r['p_val'] if np.isfinite(r['p_val']) else float('nan')
            print(f"{feat:12s}: AUC = {r['mean_auc']:.3f} ± {r['sem_auc']:.3f} (p={p_val:.3f})")

    if results['paired']:
        print("\n[MONKEY vs HUMAN]")
        for feat in FEATURES_ALL:
            comp = results['paired'].get(feat)
            if not comp:
                continue
            print(f"{feat:12s}: ΔAUC = {comp['mean_diff']:.3f}, t={comp['t_val']:.3f} (p={comp['p_val']:.3f})")

    if results.get('random_draws'):
        print(f"\nRandom baseline label sets: {len(results['random_draws'])}")

    # Save results
    out_dir = OUT_DIR_BASE / region_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RESULT_FILE
    supp_path = out_dir / SUPP_FILE

    save_data = {
        'region': region_name,
        'results': results,
        'state_path': str(state_path),
        'random_draws': results.get('random_draws', [])
    }

    # Remove legacy files
    legacy_path = out_dir / 'decode_results.pkl'
    if legacy_path.exists():
        legacy_path.unlink()

    with open(result_path, 'wb') as f:
        pickle.dump(save_data, f)

    # Supplementary nulls for archival
    supp_payload = {
        'region': region_name,
        'per_species': results['per_species'],
        'state_path': str(state_path),
        'random_draws': results.get('random_draws', [])
    }
    with open(supp_path, 'wb') as f:
        pickle.dump(supp_payload, f)

    print(f"\nSaved results: {result_path}")
    print(f"Saved supplementary nulls: {supp_path}")
    return results


def main():
    run_region(STATE_PATH, REGION_NAME)

if __name__ == '__main__':
    main()
