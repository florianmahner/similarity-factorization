"""Bounds estimation with missing values via coherence iteration.

Key insight: When running CV on data with missing values, the effective validation
set is smaller because some validation entries are already NaN. We compensate by
adjusting the sampling fraction: sf_adjusted = sf_bounds / (1 - missing_frac).

Usage:
    poetry run python sandbox/missing_bounds/run.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.cross_validation import cross_val_score

from datasets import load_dataset
from src.colors import TEAL, ROSE, CYAN, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()


def introduce_missing_values(S: np.ndarray, missing_fraction: float, random_state: int = 42) -> np.ndarray:
    """Introduce symmetric missing values (NaN) into a similarity matrix."""
    rng = np.random.RandomState(random_state)
    n = S.shape[0]
    S_missing = S.copy()
    
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)
    n_to_mask = int(missing_fraction * n_pairs)
    mask_indices = rng.choice(n_pairs, size=n_to_mask, replace=False)
    
    for idx in mask_indices:
        i, j = triu_i[idx], triu_j[idx]
        S_missing[i, j] = np.nan
        S_missing[j, i] = np.nan
    
    return S_missing


def get_missing_fraction(S: np.ndarray) -> float:
    """Compute the fraction of off-diagonal entries that are missing (NaN)."""
    n = S.shape[0]
    n_pairs = n * (n - 1) // 2
    n_missing = np.isnan(S).sum() // 2  # Symmetric
    return n_missing / n_pairs


def coherence_rank_estimation(
    S_missing: np.ndarray,
    candidate_ranks: list[int],
    cv_ranks: list[int] | None = None,
    n_cv_repeats: int = 5,
    random_state: int = 42,
    adjust_sf: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimate rank via coherence iteration for missing data.
    
    If adjust_sf=True, compensates for missing data by scaling up the sampling
    fraction used in CV: sf_cv = sf_bounds / (1 - missing_frac).
    """
    if cv_ranks is None:
        cv_ranks = candidate_ranks
    
    missing_frac = get_missing_fraction(S_missing)
    observed_frac = 1 - missing_frac
    
    results = []
    
    for k in candidate_ranks:
        if verbose:
            log.info(f"  k={k}: factorizing...")
        
        # Step 1: Factorize S_missing with rank k
        model = SRF(rank=k, missing_values=np.nan, random_state=random_state, max_outer=100, tol=1e-5)
        model.fit(S_missing)
        W = model.w_
        
        # Step 2: Impute
        S_imputed = W @ W.T
        obs_min, obs_max = np.nanmin(S_missing), np.nanmax(S_missing)
        S_imputed = np.clip(S_imputed, obs_min, obs_max)
        
        # Step 3: Estimate bounds on imputed matrix
        pmin, pmax, _ = estimate_sampling_bounds_ultra(S_imputed, random_state=random_state, verbose=False)
        sf_bounds = 0.5 * (pmin + pmax)
        
        # Step 4: Adjust sampling fraction for CV on missing data
        if adjust_sf:
            sf_cv = min(0.9, sf_bounds / observed_frac)
        else:
            sf_cv = sf_bounds
        
        # Step 5: Run CV on S_missing
        cv_result = cross_val_score(
            S_missing,
            estimator=SRF(random_state=random_state, max_outer=50),
            param_grid={"rank": cv_ranks},
            n_repeats=n_cv_repeats,
            sampling_fraction=sf_cv,
            random_state=random_state,
            verbose=0,
            missing_values=np.nan,
        )
        
        k_predicted = cv_result.best_params_["rank"]
        best_score = cv_result.best_score_
        
        if verbose:
            log.info(f"       bounds=[{pmin:.3f}, {pmax:.3f}], sf_bounds={sf_bounds:.3f}, sf_cv={sf_cv:.3f}, k'={k_predicted}")
        
        results.append({
            "k_candidate": k,
            "k_predicted": k_predicted,
            "pmin": pmin,
            "pmax": pmax,
            "sf_bounds": sf_bounds,
            "sf_cv": sf_cv,
            "cv_best_score": best_score,
            "is_fixed_point": k == k_predicted,
        })
    
    return pd.DataFrame(results)


def get_baseline_rank(S_complete: np.ndarray, cv_ranks: list[int], random_state: int = 42) -> dict:
    """Get optimal rank from complete matrix as baseline."""
    pmin, pmax, _ = estimate_sampling_bounds_ultra(S_complete, random_state=random_state, verbose=False)
    sf = 0.5 * (pmin + pmax)
    
    cv_result = cross_val_score(
        S_complete,
        estimator=SRF(random_state=random_state, max_outer=50),
        param_grid={"rank": cv_ranks},
        n_repeats=5,
        sampling_fraction=sf,
        random_state=random_state,
        verbose=0,
    )
    
    return {
        "optimal_rank": int(cv_result.best_params_["rank"]),
        "pmin": float(pmin),
        "pmax": float(pmax),
        "sf": float(sf),
    }


def plot_results(results_adj: dict, results_unadj: dict, baseline: dict, output_dir: Path):
    """Compare adjusted vs unadjusted coherence results."""
    
    n_fracs = len(results_adj)
    fig, axes = plt.subplots(2, n_fracs, figsize=(4 * n_fracs, 6), sharex=True)
    
    for col, frac in enumerate(results_adj.keys()):
        df_adj = results_adj[frac]
        df_unadj = results_unadj[frac]
        
        # Top row: Adjusted
        ax = axes[0, col]
        ax.plot(df_adj["k_candidate"], df_adj["k_predicted"], "o-", color=TEAL, lw=2, ms=8)
        k_range = [df_adj["k_candidate"].min(), df_adj["k_candidate"].max()]
        ax.plot(k_range, k_range, "--", color=GRAY, lw=1.5)
        ax.axhline(baseline["optimal_rank"], color=ROSE, ls=":", lw=2)
        fixed = df_adj[df_adj["is_fixed_point"]]
        if not fixed.empty:
            ax.scatter(fixed["k_candidate"], fixed["k_predicted"], s=200, c="gold", marker="*", zorder=5, edgecolors="k")
        ax.set_title(f"{int(frac*100)}% missing\n(adjusted SF)")
        despine(ax)
        
        # Bottom row: Unadjusted
        ax = axes[1, col]
        ax.plot(df_unadj["k_candidate"], df_unadj["k_predicted"], "o-", color=CYAN, lw=2, ms=8)
        ax.plot(k_range, k_range, "--", color=GRAY, lw=1.5)
        ax.axhline(baseline["optimal_rank"], color=ROSE, ls=":", lw=2)
        fixed = df_unadj[df_unadj["is_fixed_point"]]
        if not fixed.empty:
            ax.scatter(fixed["k_candidate"], fixed["k_predicted"], s=200, c="gold", marker="*", zorder=5, edgecolors="k")
        ax.set_title(f"(unadjusted SF)")
        ax.set_xlabel("Candidate rank k")
        despine(ax)
    
    axes[0, 0].set_ylabel("CV-predicted rank k'")
    axes[1, 0].set_ylabel("CV-predicted rank k'")
    
    fig.suptitle(f"Coherence iteration: Adjusted vs Unadjusted SF (baseline k*={baseline['optimal_rank']})", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "coherence_adj_vs_unadj.pdf", tight=False)
    plt.close(fig)


def main():
    log.info("=" * 70)
    log.info("Bounds Estimation with Missing Values - Adjusted vs Unadjusted SF")
    log.info("=" * 70)
    
    ds = load_dataset("peterson-animals", root="/SSD/datasets/similarity_datasets/peterson")
    S_complete = ds.rsm
    n = S_complete.shape[0]
    log.info(f"Loaded {n}x{n} Peterson animals matrix")
    
    candidate_ranks = list(range(2, 14, 2))
    cv_ranks = list(range(1, 16))
    missing_fractions = [0.1, 0.3, 0.5]
    
    # Baseline
    baseline = get_baseline_rank(S_complete, cv_ranks)
    log.info(f"Baseline (complete): optimal rank = {baseline['optimal_rank']}, SF = {baseline['sf']:.3f}")
    
    results_adj = {}
    results_unadj = {}
    all_results = []
    
    for frac in missing_fractions:
        log.info(f"\n{'='*60}")
        log.info(f"Missing fraction: {frac*100:.0f}%")
        log.info(f"{'='*60}")
        
        S_missing = introduce_missing_values(S_complete, frac, random_state=42)
        
        # With adjustment
        log.info("\nWith SF adjustment:")
        df_adj = coherence_rank_estimation(S_missing, candidate_ranks, cv_ranks, adjust_sf=True)
        df_adj["missing_fraction"] = frac
        df_adj["adjusted"] = True
        results_adj[frac] = df_adj
        
        # Without adjustment
        log.info("\nWithout SF adjustment:")
        df_unadj = coherence_rank_estimation(S_missing, candidate_ranks, cv_ranks, adjust_sf=False)
        df_unadj["missing_fraction"] = frac
        df_unadj["adjusted"] = False
        results_unadj[frac] = df_unadj
        
        all_results.extend([df_adj, df_unadj])
        
        # Summary
        fixed_adj = df_adj[df_adj["is_fixed_point"]]["k_candidate"].tolist()
        fixed_unadj = df_unadj[df_unadj["is_fixed_point"]]["k_candidate"].tolist()
        log.info(f"\nFixed points - Adjusted: {fixed_adj}, Unadjusted: {fixed_unadj}")
    
    # Save
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    
    # Summary table
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Baseline rank: {baseline['optimal_rank']}")
    print(f"\n{'Missing':>10} {'Adjusted FP':>15} {'Unadjusted FP':>15}")
    print("-" * 45)
    for frac in missing_fractions:
        fp_adj = results_adj[frac][results_adj[frac]["is_fixed_point"]]["k_candidate"].tolist()
        fp_unadj = results_unadj[frac][results_unadj[frac]["is_fixed_point"]]["k_candidate"].tolist()
        print(f"{frac*100:>10.0f}% {str(fp_adj):>15} {str(fp_unadj):>15}")
    
    # Plot
    plot_results(results_adj, results_unadj, baseline, OUTPUT_DIR)
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump({
            "dataset": "peterson-animals",
            "n": n,
            "candidate_ranks": candidate_ranks,
            "cv_ranks": cv_ranks,
            "missing_fractions": missing_fractions,
            "baseline": baseline,
        }, f, indent=2)
    
    log.info(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
