#!/usr/bin/env python3
"""ADMM Analysis over CCA Space"""

import os
import sys
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

admm_path = os.path.join(module_path, 'functions', 'admm')
if admm_path not in sys.path:
    sys.path.append(admm_path)

from config.paths import config
from functions.admm.cross_validation import cross_val_score

def compute_mean_rsm(cca_results, fisher=True, view_weights=None):
    """Return Fisher-z averaged RSM across views."""
    views = sorted(cca_results['components'])
    sims = []
    
    print(f"Computing RSMs for {len(views)} views: {views}")
    
    for v in views:
        comps = cca_results['components'][v]
        sim = cosine_similarity(comps)
        sims.append(sim)

    sims = np.stack(sims)
    print(f"RSM shape: {sims[0].shape}")

    if view_weights is None:
        view_weights = np.ones(len(views)) / len(views)
    else:
        view_weights = np.asarray(view_weights) / view_weights.sum()
        print(f"View weights: {dict(zip(views, view_weights))}")

    if fisher:
        sims_clipped = np.clip(sims, -0.9999, 0.9999)
        sims_z = np.arctanh(sims_clipped)
        mean_z = (view_weights[:,None,None] * sims_z).sum(axis=0)
        mean_sim = np.tanh(mean_z)
        print("Used Fisher-z transform")
    else:
        mean_sim = (view_weights[:,None,None] * sims).sum(axis=0)
        print("Used simple averaging")

    # Rescale to [0,1] for SymNMF
    mean_sim = (mean_sim - mean_sim.min()) / (mean_sim.max() - mean_sim.min())
    print(f"Final RSM range: [{mean_sim.min():.3f}, {mean_sim.max():.3f}]")
    
    return mean_sim, sims

def compute_residuals(W_final, sims, views, store_matrices=False):
    """Compute per-view residuals."""
    residuals = {}
    recon = W_final @ W_final.T
    
    for i, view in enumerate(views):
        resid = sims[i] - recon
        rms_err = np.sqrt(np.mean(resid**2))
        max_err = np.abs(resid).max()
        
        residuals[view] = {
            'rms_error': rms_err,
            'max_error': max_err,
            'mean_error': np.mean(np.abs(resid))
        }
        
        if store_matrices:
            residuals[view]['residual_matrix'] = resid
    
    return residuals

def check_admm_status():
    """Check which ADMM families have been completed."""
    output_dir = os.path.join(config.results_dir, 'admm_results')
    results_file = os.path.join(output_dir, 'admm_all_results.pkl')
    
    all_families = ['all', 'human', 'monkey']
    
    if not os.path.exists(results_file):
        print("No ADMM results found. All families need to be processed.")
        return set(), all_families
    
    try:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        completed = set(data['results'].keys())
        pending = [f for f in all_families if f not in completed]
        
        print(f"ADMM Status:")
        print(f"  Completed: {sorted(completed) if completed else 'None'}")
        print(f"  Pending: {pending if pending else 'None'}")
        
        return completed, pending
    except Exception as e:
        print(f"Error reading results file: {e}")
        return set(), all_families

def main():
    """Main ADMM analysis."""
    
    # Check Cython implementation status
    from functions.admm.admm import ADMM
    ADMM.check_cython_status()
    
    # Check current status
    completed_families, pending_families = check_admm_status()
    
    # Load CCA state
    state_fp = config.cache_file
    if not os.path.exists(state_fp):
        print(f"ERROR: CCA state file not found at {state_fp}")
        print("Please run CCA_families.py first.")
        return
    
    print(f"Loading CCA state from {state_fp}...")
    with open(state_fp, 'rb') as f:
        state = pickle.load(f)
    
    cca_all = state['cca_all']
    cca_hum = state['cca_hum'] 
    cca_mon = state['cca_mon']
    all_stims = state['all_stims']
    
    print(f"CCA state loaded. Common stimuli: {len(all_stims)}")
    
    # Output directory
    output_dir = os.path.join(config.results_dir, 'admm_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # ADMM parameters
    admm_config = config.hyperparameters.get('admm', {})
    n_repeats = admm_config.get('n_repeats', 5)
    obs_frac = admm_config.get('observed_fraction', 0.6)
    n_jobs = admm_config.get('n_jobs', -1)
    fisher_z = admm_config.get('use_fisher_z', True)
    
    print(f"ADMM params: n_repeats={n_repeats}, obs_frac={obs_frac}, n_jobs={n_jobs}")
    
    # Get rank ranges from config
    rank_ranges = config.hyperparameters.get('admm', {}).get('rank_ranges', {})
    
    families = {
        'all': (cca_all, rank_ranges.get('all')),
        'human': (cca_hum, rank_ranges.get('human')),
        'monkey': (cca_mon, rank_ranges.get('monkey'))
    }

    print(f"Rank ranges:")
    for fam, (_, ranks) in families.items():
        print(f"  {fam}: {ranks}")
    
    # Load existing results if available
    results_file = os.path.join(output_dir, 'admm_all_results.pkl')
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}...")
        with open(results_file, 'rb') as f:
            existing_data = pickle.load(f)
        results = existing_data.get('results', {})
        print(f"Found existing results for: {list(results.keys())}")
    else:
        results = {}

    # Process families and save results incrementally
    for family_name, (cca_results, rank_range) in tqdm(families.items(), desc="Processing families"):
        # Skip if already processed
        if family_name in results:
            print(f"\n{family_name.upper()} already processed, skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing {family_name.upper()}")
        print(f"{'='*50}")
        
        views = sorted(cca_results['components'].keys())
        rsm, individual_rsms = compute_mean_rsm(cca_results, fisher=fisher_z)
        
        # Check individual cache
        cache_file = os.path.join(output_dir, f"admm_{family_name}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached CV results...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            scorer = cached['scorer']
        else:
            print(f"Running ADMM cross-validation...")
            param_grid = {"rank": rank_range}
            scorer = cross_val_score(
                rsm, param_grid=param_grid, n_repeats=n_repeats,
                observed_fraction=obs_frac, random_state=42,
                verbose=1, n_jobs=n_jobs, fit_final_estimator=True
            )
            
            # Save CV cache (without RSM to save space)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'scorer': scorer, 'views': views,
                    'param_grid': param_grid
                }, f)
        
        print(f"Best rank={scorer.best_params_['rank']}, Best score={scorer.best_score_:.4f}")
        
        # Fit all rank models for visualization flexibility
        print(f"Generating components for all ranks...")
        from functions.admm.admm import ADMM
        components = {}
        for rank in tqdm(rank_range, desc=f"{family_name} ranks", leave=False):
            if hasattr(scorer, 'best_estimator_') and scorer.best_params_['rank'] == rank:
                components[rank] = scorer.best_estimator_.transform(rsm)
            else:
                model = ADMM(rank=rank, random_state=42)
                model.fit(rsm)
                components[rank] = model.transform(rsm)
        
        # Store results for this family
        results[family_name] = {
            'best_rank': scorer.best_params_['rank'],
            'best_score': scorer.best_score_,
            'cv_results': scorer.cv_results_,
            'components': components
        }
        
        # Save incremental results immediately
        print(f"Saving incremental results...")
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'all_stims': all_stims,
                'config': {'n_repeats': n_repeats, 'observed_fraction': obs_frac}
            }, f)
        print(f"Results saved with {len(results)} families complete")
    
    print(f"\n{'='*50}")
    print("ADMM ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Results: {output_dir}")
    for name, result in results.items():
        print(f"  {name}: best_rank={result['best_rank']}, best_score={result['best_score']:.4f}")
    print(f"All results: {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ADMM Analysis over CCA Space')
    parser.add_argument('--status-only', action='store_true', 
                       help='Only check status of existing results, do not run analysis')
    args = parser.parse_args()
    
    if args.status_only:
        check_admm_status()
    else:
        main() 