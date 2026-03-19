"""
Consensus matrix analysis with dimension visualization.

Implements Brunet et al. (2004) consensus clustering for SRF stability assessment:
1. Run SRF multiple times with different seeds
2. Compute consensus matrix (co-clustering frequency)
3. Hierarchical clustering to reveal structure
4. Identify boundary samples and sub-clusters
5. Visualize the actual dimensions

Usage:
    ./scripts/submit sandbox/consensus/run.py --bg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from similarity import build_similarity
from src.utils import get_output_dir

from core import (
    run_consensus_analysis,
    extract_cluster_assignments,
    compute_soft_memberships,
)
from plotting import (
    plot_consensus_heatmap,
    plot_consensus_histogram,
    plot_dendrogram,
    plot_soft_memberships,
    plot_cluster_splits,
    plot_granularity_analysis,
)
from visualize_dims import visualize_dimensions


OUTPUT_DIR = get_output_dir()


# === Configuration ===
DATASET = "mur92"  # Dataset to analyze
CV_RANK = 2        # CV-selected optimal rank
N_RUNS = 50        # Number of SRF runs
N_JOBS = 40        # Parallel jobs


def load_dataset(name: str) -> tuple[np.ndarray, dict]:
    """Load dataset by name, return similarity matrix and config."""
    configs = {
        'mur92': {
            'name': 'mur92',
            'type': 'neural_rsm',
            'path': '/SSD/datasets/similarity_datasets/mur92',
            'cv_rank': 2,
        },
        'peterson_various': {
            'name': 'peterson_various',
            'type': 'behavioral',
            'path': '/SSD/datasets/similarity_datasets/peterson',
            'subset': 'various',
            'cv_rank': 4,
        },
        'peterson_animals': {
            'name': 'peterson_animals',
            'type': 'behavioral',
            'path': '/SSD/datasets/similarity_datasets/peterson',
            'subset': 'animals',
            'cv_rank': 5,
        },
    }

    if name not in configs:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(configs.keys())}")

    cfg = configs[name]
    dataset_cfg = OmegaConf.create(cfg)
    similarity = build_similarity(dataset_cfg)

    return similarity, cfg


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    # Load data
    print(f"Loading {DATASET}...")
    similarity, cfg = load_dataset(DATASET)
    cv_rank = cfg.get('cv_rank', CV_RANK)
    n_samples = similarity.shape[0]
    print(f"  Shape: {similarity.shape}")
    print(f"  CV-selected rank: {cv_rank}")

    # === Run consensus analysis ===
    print(f"\n=== Consensus Analysis (k={cv_rank}, n_runs={N_RUNS}) ===")
    result = run_consensus_analysis(similarity, cv_rank, n_runs=N_RUNS, n_jobs=N_JOBS)

    consensus = result['consensus']
    linkage_matrix = result['linkage']
    embeddings = result['embeddings']

    print(f"\n  Cophenetic correlation: {result['cophenetic_corr']:.3f}")
    print(f"  Dispersion: {result['dispersion']:.3f}")
    print(f"  Reconstruction error: {result['mean_recon_error']:.4f} +/- {result['std_recon_error']:.4f}")

    # Interpret stability
    ccc = result['cophenetic_corr']
    if ccc > 0.9:
        stability = "STABLE"
    elif ccc > 0.7:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"
    print(f"\n  Stability: {stability} (CCC={ccc:.3f})")

    # === Consensus visualizations ===
    print("\n=== Consensus Visualizations ===")

    print("  Plotting consensus heatmap...")
    plot_consensus_heatmap(
        consensus, linkage_matrix,
        OUTPUT_DIR / "consensus_heatmap.png",
        title=f"Consensus Matrix ({DATASET}, k={cv_rank}, CCC={ccc:.3f})"
    )

    print("  Plotting histogram...")
    plot_consensus_histogram(consensus, OUTPUT_DIR / "consensus_histogram.png")

    print("  Plotting dendrogram...")
    plot_dendrogram(
        linkage_matrix,
        OUTPUT_DIR / "dendrogram.png",
        k=cv_rank,
        title=f"Hierarchical Clustering (cut at k={cv_rank})"
    )

    print("  Plotting soft memberships...")
    soft_memberships = compute_soft_memberships(consensus, linkage_matrix, cv_rank)
    plot_soft_memberships(soft_memberships, linkage_matrix, OUTPUT_DIR / "soft_memberships.png")

    # === Granularity analysis ===
    print("\n=== Granularity Analysis ===")
    print("  Analyzing sub-structure beyond CV rank...")

    granularity_df = plot_granularity_analysis(
        consensus, linkage_matrix, cv_rank,
        OUTPUT_DIR / "granularity_analysis.png"
    )
    granularity_df.to_csv(OUTPUT_DIR / "granularity_metrics.csv", index=False)

    print("  Plotting cluster splits...")
    plot_cluster_splits(linkage_matrix, cv_rank, OUTPUT_DIR / "cluster_splits.png")

    # === Cluster assignments ===
    print("\n=== Cluster Assignments ===")
    for k in range(cv_rank, min(cv_rank + 3, n_samples)):
        assignments = extract_cluster_assignments(linkage_matrix, k)
        sizes = [np.sum(assignments == c) for c in range(1, k + 1)]
        print(f"  k={k}: {sizes}")

        # Save
        df = pd.DataFrame({'sample_idx': range(n_samples), 'cluster': assignments})
        df.to_csv(OUTPUT_DIR / f"assignments_k{k}.csv", index=False)

    # === Boundary samples ===
    print("\n=== Boundary Samples (high entropy) ===")
    entropy = -np.sum(soft_memberships * np.log(soft_memberships + 1e-10), axis=1)
    top_boundary = np.argsort(entropy)[-10:][::-1]
    print("  Sample  Entropy  Memberships")
    for idx in top_boundary:
        memb_str = "  ".join([f"{m:.2f}" for m in soft_memberships[idx]])
        print(f"  {idx:5d}   {entropy[idx]:.3f}    [{memb_str}]")

    # Save soft memberships
    membership_df = pd.DataFrame(soft_memberships, columns=[f'cluster_{i+1}' for i in range(cv_rank)])
    membership_df['sample_idx'] = range(n_samples)
    membership_df['entropy'] = entropy
    membership_df = membership_df[['sample_idx', 'entropy'] + [f'cluster_{i+1}' for i in range(cv_rank)]]
    membership_df.to_csv(OUTPUT_DIR / "soft_memberships.csv", index=False)

    # === Dimension visualizations ===
    # Use the most central embedding (lowest reconstruction error)
    recon_errors = [np.linalg.norm(similarity - e @ e.T, 'fro') for e in embeddings]
    best_idx = np.argmin(recon_errors)
    best_embedding = embeddings[best_idx]
    print(f"\n  Using embedding from run {best_idx} (lowest recon error)")

    visualize_dimensions(
        best_embedding,
        embeddings,
        linkage_matrix,
        OUTPUT_DIR,
        labels=None,  # Could add object names
    )

    # === Summary ===
    summary = {
        'dataset': DATASET,
        'n_samples': n_samples,
        'cv_rank': cv_rank,
        'n_runs': N_RUNS,
        'cophenetic_corr': result['cophenetic_corr'],
        'dispersion': result['dispersion'],
        'mean_recon_error': result['mean_recon_error'],
        'stability': stability,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(f"\n=== Done ===")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
