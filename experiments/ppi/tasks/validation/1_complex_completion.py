"""Complex completion validation: leave-one-out prediction of complex members.

This script tests whether the SRF embedding can predict missing members of known
CORUM complexes, going beyond simple "clustering matches complexes" validation.

For each CORUM complex with ≥5 members:
1. Hold out one protein at a time
2. Compute similarity of all proteins to remaining complex members
3. Rank the held-out protein among all candidates
4. Measure AUROC, AUPR, and top-k hit rates

Baselines:
- Common Neighbors (CN): topological similarity in STRING network
- Node2Vec: graph embedding baseline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from analyses.ppi.corum import load_corum, map_string_ids_to_genes
from analyses.ppi.graph_utils import load_network, build_sparse_adjacency
from analyses.ppi.baselines import evaluate_baseline_fast


def compute_similarity_to_complex(
    embedding: np.ndarray,
    complex_members_idx: list[int],
    held_out_idx: int,
    method: str = "cosine_mean",
) -> np.ndarray:
    """Compute similarity of all proteins to a complex (excluding held-out member).

    Parameters
    ----------
    embedding : np.ndarray
        Protein embeddings (n_proteins, n_dims)
    complex_members_idx : list[int]
        Indices of complex members
    held_out_idx : int
        Index of held-out protein
    method : str
        Similarity method: "cosine_mean" or "cosine_median"

    Returns
    -------
    np.ndarray
        Similarity scores for all proteins (n_proteins,)
    """
    # Remove held-out protein from complex members
    remaining_idx = [idx for idx in complex_members_idx if idx != held_out_idx]

    if len(remaining_idx) == 0:
        return np.zeros(len(embedding))

    # Compute centroid of remaining complex members
    if method == "cosine_mean":
        centroid = embedding[remaining_idx].mean(axis=0, keepdims=True)
    elif method == "cosine_median":
        centroid = np.median(embedding[remaining_idx], axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute cosine similarity of all proteins to centroid
    similarities = cosine_similarity(embedding, centroid).flatten()

    return similarities


def evaluate_complex_completion(
    embedding: np.ndarray,
    proteins: np.ndarray,
    corum_complexes: dict,
    min_complex_size: int = 5,
    max_complex_size: int = 200,
) -> pd.DataFrame:
    """Evaluate complex completion using leave-one-out validation.

    Parameters
    ----------
    embedding : np.ndarray
        Protein embeddings (n_proteins, n_dims)
    proteins : np.ndarray
        Protein names aligned with embedding rows
    corum_complexes : dict
        Dictionary mapping complex names to sets of protein names
    min_complex_size : int
        Minimum complex size to include
    max_complex_size : int
        Maximum complex size to include (to avoid computational burden)

    Returns
    -------
    pd.DataFrame
        Results with columns: complex_name, complex_size, auroc, aupr, top10_hit, top50_hit, top100_hit
    """
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    results = []

    # Filter complexes by size and coverage
    valid_complexes = {}
    for name, members in corum_complexes.items():
        # Find members present in embedding
        present_members = [m for m in members if m in protein_to_idx]
        if min_complex_size <= len(present_members) <= max_complex_size:
            valid_complexes[name] = present_members

    print(
        f"Evaluating {len(valid_complexes)} complexes (size {min_complex_size}-{max_complex_size})"
    )

    for complex_name, complex_members in tqdm(
        valid_complexes.items(), desc="Complexes"
    ):
        complex_idx = [protein_to_idx[m] for m in complex_members]
        complex_size = len(complex_idx)

        # Leave-one-out for each member
        aurocs = []
        auprs = []
        top10_hits = []
        top50_hits = []
        top100_hits = []

        for held_out_idx in complex_idx:
            # Compute similarity scores
            scores = compute_similarity_to_complex(
                embedding, complex_idx, held_out_idx, method="cosine_mean"
            )

            # Create labels: 1 for held-out protein, 0 for all others
            labels = np.zeros(len(proteins), dtype=int)
            labels[held_out_idx] = 1

            # Exclude other complex members from evaluation (only predict held-out)
            mask = np.ones(len(proteins), dtype=bool)
            mask[complex_idx] = False
            mask[held_out_idx] = True  # Include held-out in evaluation

            scores_eval = scores[mask]
            labels_eval = labels[mask]

            # Compute metrics
            if len(np.unique(labels_eval)) == 2:  # Need both classes
                auroc = roc_auc_score(labels_eval, scores_eval)
                aupr = average_precision_score(labels_eval, scores_eval)
            else:
                auroc = np.nan
                aupr = np.nan

            # Compute top-k hit rates
            top_k_idx = np.argsort(scores)[-100:][::-1]  # Top 100
            top10_hit = 1 if held_out_idx in top_k_idx[:10] else 0
            top50_hit = 1 if held_out_idx in top_k_idx[:50] else 0
            top100_hit = 1 if held_out_idx in top_k_idx else 0

            aurocs.append(auroc)
            auprs.append(aupr)
            top10_hits.append(top10_hit)
            top50_hits.append(top50_hit)
            top100_hits.append(top100_hit)

        # Aggregate metrics across all leave-one-out trials
        results.append(
            {
                "complex_name": complex_name,
                "complex_size": complex_size,
                "auroc_mean": np.nanmean(aurocs),
                "auroc_std": np.nanstd(aurocs),
                "aupr_mean": np.nanmean(auprs),
                "aupr_std": np.nanstd(auprs),
                "top10_hit_rate": np.mean(top10_hits),
                "top50_hit_rate": np.mean(top50_hits),
                "top100_hit_rate": np.mean(top100_hits),
                "n_trials": complex_size,
            }
        )

    return pd.DataFrame(results)


def evaluate_cn_baseline(
    adj_sparse: csr_matrix,
    proteins: np.ndarray,
    corum_complexes: dict,
    min_complex_size: int = 5,
    max_complex_size: int = 200,
) -> pd.DataFrame:
    """Evaluate common neighbors baseline for complex completion.

    Parameters
    ----------
    adj_sparse : csr_matrix
        Sparse adjacency matrix of STRING network
    proteins : np.ndarray
        Protein names
    corum_complexes : dict
        CORUM complexes
    min_complex_size : int
        Minimum complex size
    max_complex_size : int
        Maximum complex size

    Returns
    -------
    pd.DataFrame
        Results with same structure as evaluate_complex_completion
    """
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Compute CN similarity matrix: A @ A
    cn_matrix = adj_sparse @ adj_sparse

    results = []
    valid_complexes = {}
    for name, members in corum_complexes.items():
        present_members = [m for m in members if m in protein_to_idx]
        if min_complex_size <= len(present_members) <= max_complex_size:
            valid_complexes[name] = present_members

    print(f"Evaluating {len(valid_complexes)} complexes with CN baseline")

    for complex_name, complex_members in tqdm(
        valid_complexes.items(), desc="CN baseline"
    ):
        complex_idx = [protein_to_idx[m] for m in complex_members]
        complex_size = len(complex_idx)

        aurocs = []
        auprs = []
        top10_hits = []
        top50_hits = []
        top100_hits = []

        for held_out_idx in complex_idx:
            # Compute CN scores: sum of CN with remaining complex members
            remaining_idx = [idx for idx in complex_idx if idx != held_out_idx]

            # Sum CN scores across remaining members
            scores = np.array(cn_matrix[:, remaining_idx].sum(axis=1)).flatten()

            # Create labels
            labels = np.zeros(len(proteins), dtype=int)
            labels[held_out_idx] = 1

            # Exclude other complex members
            mask = np.ones(len(proteins), dtype=bool)
            mask[complex_idx] = False
            mask[held_out_idx] = True

            scores_eval = scores[mask]
            labels_eval = labels[mask]

            if len(np.unique(labels_eval)) == 2:
                auroc = roc_auc_score(labels_eval, scores_eval)
                aupr = average_precision_score(labels_eval, scores_eval)
            else:
                auroc = np.nan
                aupr = np.nan

            top_k_idx = np.argsort(scores)[-100:][::-1]
            top10_hit = 1 if held_out_idx in top_k_idx[:10] else 0
            top50_hit = 1 if held_out_idx in top_k_idx[:50] else 0
            top100_hit = 1 if held_out_idx in top_k_idx else 0

            aurocs.append(auroc)
            auprs.append(aupr)
            top10_hits.append(top10_hit)
            top50_hits.append(top50_hit)
            top100_hits.append(top100_hit)

        results.append(
            {
                "complex_name": complex_name,
                "complex_size": complex_size,
                "auroc_mean": np.nanmean(aurocs),
                "auroc_std": np.nanstd(aurocs),
                "aupr_mean": np.nanmean(auprs),
                "aupr_std": np.nanstd(auprs),
                "top10_hit_rate": np.mean(top10_hits),
                "top50_hit_rate": np.mean(top50_hits),
                "top100_hit_rate": np.mean(top100_hits),
                "n_trials": complex_size,
            }
        )

    return pd.DataFrame(results)


def plot_results(srf_results: pd.DataFrame, cn_results: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. AUROC comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUROC
    ax = axes[0]
    methods = ["SRF (Rank-50)", "CN Baseline"]
    aurocs = [srf_results["auroc_mean"].mean(), cn_results["auroc_mean"].mean()]
    auroc_stds = [srf_results["auroc_mean"].std(), cn_results["auroc_mean"].std()]

    ax.bar(
        methods,
        aurocs,
        yerr=auroc_stds,
        capsize=5,
        alpha=0.7,
        color=["#2E86AB", "#A23B72"],
    )
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Complex Completion: AUROC")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    # AUPR
    ax = axes[1]
    auprs = [srf_results["aupr_mean"].mean(), cn_results["aupr_mean"].mean()]
    aupr_stds = [srf_results["aupr_mean"].std(), cn_results["aupr_mean"].std()]

    ax.bar(
        methods,
        auprs,
        yerr=aupr_stds,
        capsize=5,
        alpha=0.7,
        color=["#2E86AB", "#A23B72"],
    )
    ax.set_ylabel("Mean AUPR")
    ax.set_title("Complex Completion: AUPR")
    ax.set_ylim([0, max(auprs) * 1.2])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "auroc_aupr_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Top-k hit rates
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(3)
    width = 0.35

    srf_hits = [
        srf_results["top10_hit_rate"].mean(),
        srf_results["top50_hit_rate"].mean(),
        srf_results["top100_hit_rate"].mean(),
    ]
    cn_hits = [
        cn_results["top10_hit_rate"].mean(),
        cn_results["top50_hit_rate"].mean(),
        cn_results["top100_hit_rate"].mean(),
    ]

    ax.bar(x - width / 2, srf_hits, width, label="SRF", alpha=0.7, color="#2E86AB")
    ax.bar(x + width / 2, cn_hits, width, label="CN", alpha=0.7, color="#A23B72")

    ax.set_ylabel("Hit Rate")
    ax.set_title("Top-k Hit Rates for Held-Out Proteins")
    ax.set_xticks(x)
    ax.set_xticklabels(["Top-10", "Top-50", "Top-100"])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "topk_hit_rates.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. AUROC vs complex size
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        srf_results["complex_size"],
        srf_results["auroc_mean"],
        alpha=0.5,
        label="SRF",
        s=30,
        color="#2E86AB",
    )
    ax.scatter(
        cn_results["complex_size"],
        cn_results["auroc_mean"],
        alpha=0.5,
        label="CN",
        s=30,
        color="#A23B72",
    )

    ax.set_xlabel("Complex Size")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("AUROC vs Complex Size")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "auroc_vs_size.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Paths
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data" / "ppi"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    validation_output_dir = (
        base_dir
        / "experiments"
        / "ppi"
        / "validation"
        / "outputs"
        / "complex_completion"
    )

    # Load embedding
    print("Loading rank-50 SRF embedding...")
    embedding = np.load(output_dir / "embedding.npy")
    print(f"Embedding shape: {embedding.shape}")

    # Load protein names
    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])
    print(f"Number of proteins: {len(proteins)}")

    # Load and map CORUM complexes
    print("Loading CORUM complexes...")
    corum_path = data_dir / "corum_complexes.txt"
    corum_complexes = load_corum(corum_path)
    print(f"Total CORUM complexes: {len(corum_complexes)}")

    # Load STRING network for baseline
    print("Loading STRING network...")
    string_path = data_dir / "STRING_human_min900_v12.csv"
    g, has_weights = load_network(string_path)
    all_nodes = sorted(g.nodes())

    # Map proteins if needed
    mapping_file = data_dir / "protein_info_900.csv"
    if mapping_file.exists():
        all_nodes_mapped, n_mapped = map_string_ids_to_genes(all_nodes, mapping_file)
        print(f"Mapped {n_mapped} ENSP IDs to gene names")
        all_nodes = all_nodes_mapped

    # Build adjacency for baseline
    adj_sparse = build_sparse_adjacency(g, all_nodes)

    # Evaluate SRF
    print("\n" + "=" * 50)
    print("Evaluating SRF (Rank-50)")
    print("=" * 50)
    srf_results = evaluate_complex_completion(
        embedding, proteins, corum_complexes, min_complex_size=5, max_complex_size=50
    )

    # Evaluate CN baseline
    print("\n" + "=" * 50)
    print("Evaluating Common Neighbors Baseline")
    print("=" * 50)
    cn_results = evaluate_cn_baseline(
        adj_sparse, proteins, corum_complexes, min_complex_size=5, max_complex_size=50
    )

    # Save results
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    srf_results.to_csv(validation_output_dir / "srf_results.csv", index=False)
    cn_results.to_csv(validation_output_dir / "cn_results.csv", index=False)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    print(f"\nSRF (Rank-50):")
    print(
        f"  Mean AUROC: {srf_results['auroc_mean'].mean():.3f} ± {srf_results['auroc_mean'].std():.3f}"
    )
    print(
        f"  Mean AUPR:  {srf_results['aupr_mean'].mean():.4f} ± {srf_results['aupr_mean'].std():.4f}"
    )
    print(f"  Top-10 hit rate:  {srf_results['top10_hit_rate'].mean():.3f}")
    print(f"  Top-50 hit rate:  {srf_results['top50_hit_rate'].mean():.3f}")
    print(f"  Top-100 hit rate: {srf_results['top100_hit_rate'].mean():.3f}")

    print(f"\nCommon Neighbors Baseline:")
    print(
        f"  Mean AUROC: {cn_results['auroc_mean'].mean():.3f} ± {cn_results['auroc_mean'].std():.3f}"
    )
    print(
        f"  Mean AUPR:  {cn_results['aupr_mean'].mean():.4f} ± {cn_results['aupr_mean'].std():.4f}"
    )
    print(f"  Top-10 hit rate:  {cn_results['top10_hit_rate'].mean():.3f}")
    print(f"  Top-50 hit rate:  {cn_results['top50_hit_rate'].mean():.3f}")
    print(f"  Top-100 hit rate: {cn_results['top100_hit_rate'].mean():.3f}")

    # Compute improvement
    auroc_improvement = (
        (srf_results["auroc_mean"].mean() - cn_results["auroc_mean"].mean())
        / cn_results["auroc_mean"].mean()
        * 100
    )
    aupr_improvement = (
        (srf_results["aupr_mean"].mean() - cn_results["aupr_mean"].mean())
        / cn_results["aupr_mean"].mean()
        * 100
    )

    print(f"\nImprovement over CN:")
    print(f"  AUROC: +{auroc_improvement:.1f}%")
    print(f"  AUPR:  +{aupr_improvement:.1f}%")

    # Create plots
    print("\nGenerating plots...")
    plot_results(srf_results, cn_results, validation_output_dir)

    print(f"\nResults saved to: {validation_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
