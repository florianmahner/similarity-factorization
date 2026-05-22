# Computation only - cross-view CCA analysis with shuffling
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_PROC_BIND'] = 'TRUE'

import numpy as np
import sys, pickle, time
from pathlib import Path

print("Starting cross-view CCA analysis with shuffling...")
print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# --- Clean, portable imports ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

print("Loading required modules...")
from functions.load import load_mri, load_macq, match_datasets
from functions.cca import cca_cv, fit_cca
import pandas as pd
from config.paths import config
print("Modules loaded successfully.")

# === Parameters Loaded from Config ===
HPARAMS = config.hyperparameters['crossview']
APARAMS = config.analysis
CCA_FAMILIES = config.hyperparameters['cca_families']

# Use default reliability and split from analysis section
RELIABILITY = APARAMS['default_reliability']
SPLIT = APARAMS['default_split']

# Family-specific regularization values from cca_families section
REG_VALUES = {
    'cross-species': 10**CCA_FAMILIES['cross-species']['reg'],
    'human-only': 10**CCA_FAMILIES['human']['reg'],
    'monkey-only': 10**CCA_FAMILIES['monkey']['reg']
}

NUM_CC = HPARAMS['n_cc']
NUM_FOLDS = HPARAMS['n_folds']

# Separate repetitions for empirical vs null distributions
NUM_REPS_EMPIRICAL = HPARAMS['n_reps_empirical']
NUM_REPS_NULL = HPARAMS['n_reps_null']

NUM_JOBS = APARAMS['n_jobs']

# Shuffling parameters from config
N_PERM = HPARAMS.get('n_perm', 50)  # Number of permutations for null distribution
PERM_JOBS = HPARAMS.get('perm_jobs', 0)  # Permutation parallelism (0 = use n_jobs)
if PERM_JOBS == 0:
    PERM_JOBS = min(NUM_JOBS, 20)  # Conservative limit for 80-core system

# View labels and styling info for visualization
view_label_map = {
   'human_01': 'Human 1',
   'human_02': 'Human 2', 
   'human_03': 'Human 3',
   'monkey_F': 'Monkey F',
   'monkey_N': 'Monkey N'
}

ALPHAS = {
   'monkey_N': 0.7,
   'monkey_F': 1.0,
   'human_01': 1.0,
   'human_02': 0.8,
   'human_03': 0.6
}

print(f"\n=== CONFIGURATION ===")
print(f"Using config from: {config.root_dir}")
print(f"Results will be saved to: {config.results_dir}")
print(f"Parameters: n_perm={N_PERM}, n_jobs={NUM_JOBS}, perm_jobs={PERM_JOBS}")
print(f"CCA settings: n_cc={NUM_CC}, n_folds={NUM_FOLDS}")
print(f"Regularization values from cca_families:")
for family, reg_val in REG_VALUES.items():
    print(f"  {family}: {reg_val:.0e}")
print(f"Repetitions: empirical={NUM_REPS_EMPIRICAL}, null={NUM_REPS_NULL}")
print(f"Thresholds: reliability={RELIABILITY}, split={SPLIT}")

# Load and prepare datasets for analysis
print(f"\n=== LOADING DATASETS ===")
print("Loading datasets - this may take a few minutes...")
raw_data = {}
for m in APARAMS['monkey_ids']:
    print(f"  Loading monkey {m} data...")
    arr, stim, _ = load_macq(
        monkey=m,
        min_reliab=RELIABILITY, roi=APARAMS['roi_monkey']
    )
    raw_data[f'monkey_{m}'] = (arr, stim)
    print(f"    Monkey {m}: {arr.shape[0]} stimuli, {arr.shape[1]} channels")

for h in APARAMS['human_subjects']:
    print(f"  Loading human subject {h} data...")
    arr, stim, _ = load_mri(
        sub=h, roi=APARAMS['roi_human'],
        min_splithalf=SPLIT
    )
    raw_data[f'human_{h}'] = (arr, stim)
    print(f"    Human {h}: {arr.shape[0]} stimuli, {arr.shape[1]} voxels")

print("  Matching datasets across subjects...")
data_all, stims = match_datasets(raw_data)
data_inter = data_all
data_hum = {k: v for k, v in data_all.items() if 'human' in k}
data_mon = {k: v for k, v in data_all.items() if 'monkey' in k}

print(f"\n=== DATASET SUMMARY ===")
print(f"Cross-species: {len(data_inter)} views, {list(data_inter.values())[0].shape[0]} stimuli")
print(f"Human-only: {len(data_hum)} views, {list(data_hum.values())[0].shape[0]} stimuli")  
print(f"Monkey-only: {len(data_mon)} views, {list(data_mon.values())[0].shape[0]} stimuli")
print("Data loading completed successfully!")

# Run CCA with permutation testing for all dataset combinations
print(f"\n=== RUNNING CCA WITH {N_PERM} PERMUTATIONS ===")

# ──────────────────────────────────────────────────────────────────────
# Significance helpers (BH/FDR and contiguous-from-start count)
def _bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    rej = out <= alpha
    return rej, out

def _sig_stats(obs_corrs, null_corrs, alpha=0.05):
    obs = np.asarray(obs_corrs, float)
    null = np.asarray(null_corrs, float)
    if null.ndim != 2 or null.shape[1] != obs.shape[0]:
        return {
            'p_values': [], 'q_values': [], 'sig_idx': [],
            'sig_mask_all': [], 'sig_mask_contig': [], 'contig_n': 0,
            'null95': [], 'alpha': float(alpha)
        }
    p = np.mean(null >= obs[None, :], axis=0)
    sig_mask_all, q = _bh_fdr(p, alpha=alpha)
    # contiguous from component 1 (index 0)
    contig = 0
    while contig < sig_mask_all.size and bool(sig_mask_all[contig]):
        contig += 1
    sig_mask_contig = np.zeros_like(sig_mask_all, dtype=bool)
    if contig > 0:
        sig_mask_contig[:contig] = True
    null95 = np.percentile(null, 95, axis=0)
    return {
        'p_values': p.astype(float).tolist(),
        'q_values': q.astype(float).tolist(),
        'sig_idx': np.where(sig_mask_all)[0].astype(int).tolist(),
        'sig_mask_all': sig_mask_all.astype(bool).tolist(),
        'sig_mask_contig': sig_mask_contig.astype(bool).tolist(),
        'contig_n': int(contig),
        'null95': null95.astype(float).tolist(),
        'alpha': float(alpha)
    }

def compute_family_cca(family_data):
    """Compute CCA for one family"""
    name, data, seed = family_data
    print(f"Computing {name} CCA with permutation testing...")
    start_time = time.time()
    
    results, null_dist = cca_cv(
        data, reg=REG_VALUES[name], n_cc=NUM_CC,
        n_folds=NUM_FOLDS, n_reps=NUM_REPS_EMPIRICAL, n_jobs=NUM_JOBS,
        random_state=seed, n_perm=N_PERM, n_reps_perm=NUM_REPS_NULL,
        resample_with_replacement=False, perm_jobs=PERM_JOBS, verbose=True
    )
    
    print(f"      {name} completed in {time.time() - start_time:.1f}s")
    return name, results, null_dist

# Run all families in parallel instead of sequentially
from joblib import Parallel, delayed

family_configs = [
    ("cross-species", data_inter, 42),
    ("human-only", data_hum, 43), 
    ("monkey-only", data_mon, 44)
]

family_results = Parallel(n_jobs=3)(
    delayed(compute_family_cca)(config) for config in family_configs
)

# Unpack results
for name, results, null_dist in family_results:
    if name == "cross-species":
        results_inter, null_inter = results, null_dist
    elif name == "human-only":
        results_hum, null_hum = results, null_dist
    elif name == "monkey-only":
        results_mon, null_mon = results, null_dist

# Add component scores from empirical CCA
print("Computing empirical CCA for component scores...")
empirical_components = {}
for name, data, seed in family_configs:
    if APARAMS.get('demean_groups', True):
        # Build per-view group labels for human views
        groups = {}
        for v in data:
            if v.startswith('human_'):
                sid = v.split('_')[-1]
                meta_p = os.path.join(config.mri_dir, 'betas_csv', f'sub-{sid}_StimulusMetadata.csv')
                if os.path.exists(meta_p):
                    df = pd.read_csv(meta_p)
                    if 'stimulus' in df.columns.str.lower().tolist():
                        stim_col = df.columns[df.columns.str.lower()=='stimulus'][0]
                        df['stim'] = df[stim_col].astype(str).str.replace('.jpg','', regex=False)
                        pos = {s:i for i,s in enumerate(df['stim'].tolist())}
                        order = [pos.get(s, -1) for s in stims]
                        if not any(o < 0 for o in order):
                            cols = {c.lower(): c for c in df.columns}
                            if 'session' in cols and 'run' in cols:
                                session = df[cols['session']].values[order].astype(int)
                                run = df[cols['run']].values[order].astype(int)
                                groups[v] = (session * 100 + run).astype(int)
        empirical_components[name] = fit_cca(data, reg=REG_VALUES[name], n_cc=NUM_CC, group_labels=groups if groups else None, demean_groups=bool(groups))
    else:
        empirical_components[name] = fit_cca(data, reg=REG_VALUES[name], n_cc=NUM_CC)
    print(f"  {name}: components shape {list(empirical_components[name]['components'].values())[0].shape}")

# Save all results for visualization
alpha = HPARAMS.get('alpha', 0.05)
significance = {
    'cross-species': _sig_stats(results_inter['comp_corrs'], null_inter, alpha=alpha),
    'human_only':    _sig_stats(results_hum['comp_corrs'],   null_hum,   alpha=alpha),
    'monkey_only':   _sig_stats(results_mon['comp_corrs'],   null_mon,   alpha=alpha),
}

results_to_save = {
    'cross-species': results_inter,
    'human_only': results_hum,
    'monkey_only': results_mon,
    'null_distributions': {
        'cross-species': null_inter,
        'human_only': null_hum,
        'monkey_only': null_mon
    },
    'significance': significance,
    'components': {
        'cross-species': empirical_components['cross-species']['components'],
        'human_only': empirical_components['human-only']['components'],
        'monkey_only': empirical_components['monkey-only']['components']
    },
    'view_label_map': view_label_map,
    'alphas': ALPHAS,
    'num_cc': NUM_CC,
    'config': {
        'reliability': RELIABILITY,
        'split': SPLIT,
        'reg_values': REG_VALUES,
        'num_folds': NUM_FOLDS,
        'num_reps_empirical': NUM_REPS_EMPIRICAL,
        'num_reps_null': NUM_REPS_NULL,
        'n_perm': N_PERM
    }
}

output_path = config.results_dir / 'crossview_results.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_to_save, f)

print(f"\nCross-view analysis with shuffling complete. Results saved to {output_path}")
