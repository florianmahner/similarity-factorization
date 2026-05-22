"""
Consensus Comparison for Symmetric NMF

Compare different consensus approaches for combining multiple SRF runs:
1. K-means clustering (cNMF approach, Kotliar et al. 2019)
2. Hungarian alignment (exploits permutation-only ambiguity)
3. Selection-based (pick most central run)

Background:
-----------
Standard NMF (V ≈ WH) has both permutation AND rotation ambiguity, so clustering
on the factor space is appropriate - factors from different runs may not correspond
1-to-1.

Symmetric NMF (S ≈ WW^T) has ONLY permutation ambiguity - the same k factors exist
in each run, just in different order. Hungarian matching should be more principled.

Literature:
-----------
- Brunet et al. 2004: Introduced consensus clustering for NMF using connectivity
  matrices and cophenetic correlation for rank selection.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC384712/

- Kotliar et al. 2019 (cNMF): L2-normalize factors, filter outliers via KNN,
  k-means clustering, then median within clusters.
  https://elifesciences.org/articles/43803

Key Finding:
------------
K-means clustering FAILS for symmetric NMF (2.3-3.7x worse than Hungarian):
- All runs find nearly identical factors (just permuted)
- K-means clusters corresponding factors together but loses proper magnitude
- Hungarian alignment preserves 1-to-1 correspondence and scale

For symmetric NMF, use Hungarian alignment. K-means is only appropriate for
standard NMF where W and H differ and rotation ambiguity exists.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import median_abs_deviation
from sklearn.cluster import KMeans

from pysrf import SRF, EnsembleEmbedding
from src.datasets.loaders import load_mur92
from src.colors import TEAL, ROSE, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def l2_norm_columns(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def recon_error(emb: np.ndarray, rsm: np.ndarray) -> float:
    recon = emb @ emb.T
    return np.linalg.norm(rsm - recon, "fro") / np.linalg.norm(rsm, "fro")


def hungarian_align(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target to reference using Hungarian algorithm on cosine similarity."""
    sim = ref.T @ target
    _, col_ind = linear_sum_assignment(-sim)
    return col_ind


def consensus_kmeans(
    embeddings: np.ndarray, rank: int, outlier_quantile: float = 0.9
) -> np.ndarray:
    """
    cNMF-style consensus: k-means clustering on L2-normalized factors.

    Parameters
    ----------
    embeddings : ndarray of shape (n_runs, n_samples, rank)
        Embeddings from multiple runs
    rank : int
        Number of factors
    outlier_quantile : float
        Quantile threshold for outlier removal based on KNN distance

    Returns
    -------
    consensus : ndarray of shape (n_samples, rank)
    """
    n_runs, n_samples, _ = embeddings.shape

    # L2-normalize columns and stack all factors
    all_factors = []
    for emb in embeddings:
        emb_norm = l2_norm_columns(emb)
        all_factors.append(emb_norm.T)  # (rank, n_samples)
    all_factors = np.vstack(all_factors)  # (n_runs * rank, n_samples)

    # Outlier removal based on local density (simplified KNN approach)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(10, len(all_factors) - 1), metric="euclidean")
    nn.fit(all_factors)
    distances, _ = nn.kneighbors(all_factors)
    mean_dist = distances.mean(axis=1)
    threshold = np.quantile(mean_dist, outlier_quantile)
    good_mask = mean_dist <= threshold

    good_factors = all_factors[good_mask]

    # K-means clustering
    kmeans = KMeans(n_clusters=rank, random_state=42, n_init=10)
    labels = kmeans.fit_predict(good_factors)

    # Median within each cluster
    consensus = np.zeros((rank, n_samples))
    for k in range(rank):
        cluster_factors = good_factors[labels == k]
        if len(cluster_factors) > 0:
            consensus[k] = np.median(cluster_factors, axis=0)

    return consensus.T  # (n_samples, rank)


def consensus_hungarian(
    embeddings: np.ndarray, rank: int, aggregation: str = "median"
) -> np.ndarray:
    """
    Hungarian alignment consensus: align factors 1-to-1 across runs.

    Parameters
    ----------
    embeddings : ndarray of shape (n_runs, n_samples, rank)
    rank : int
    aggregation : str, "median" or "mean"

    Returns
    -------
    consensus : ndarray of shape (n_samples, rank)
    """
    n_runs = embeddings.shape[0]

    # L2-normalize for alignment
    embeddings_norm = np.array([l2_norm_columns(e) for e in embeddings])

    # Align all to first run
    ref = embeddings_norm[0]
    aligned = np.zeros_like(embeddings)
    aligned[0] = embeddings[0]

    for i in range(1, n_runs):
        perm = hungarian_align(ref, embeddings_norm[i])
        aligned[i] = embeddings[i][:, perm]

    # Aggregate
    if aggregation == "median":
        return np.median(aligned, axis=0)
    else:
        return np.mean(aligned, axis=0)


def consensus_select(embeddings: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    """
    Selection-based: return the most central run (highest agreement with others).

    Returns
    -------
    consensus : ndarray of shape (n_samples, rank)
    selected_idx : int
    """
    n_runs = embeddings.shape[0]

    # L2-normalize and align
    embeddings_norm = np.array([l2_norm_columns(e) for e in embeddings])
    ref = embeddings_norm[0]
    aligned = np.zeros_like(embeddings)
    aligned[0] = embeddings[0]

    for i in range(1, n_runs):
        perm = hungarian_align(ref, embeddings_norm[i])
        aligned[i] = embeddings[i][:, perm]

    # Compute agreement scores
    aligned_norm = np.array([l2_norm_columns(e) for e in aligned])
    scores = np.zeros(n_runs)

    for i in range(n_runs):
        total_sim = 0.0
        count = 0
        for j in range(rank):
            for k in range(n_runs):
                if k != i:
                    sim = np.dot(aligned_norm[i, :, j], aligned_norm[k, :, j])
                    total_sim += sim
                    count += 1
        scores[i] = total_sim / count if count > 0 else 0

    best_idx = int(np.argmax(scores))
    return aligned[best_idx], best_idx


def run_experiment(
    rsm: np.ndarray,
    rank: int,
    n_runs: int = 100,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run consensus comparison experiment."""

    log.info(f"Fitting {n_runs} SRF runs with rank={rank}...")
    model = SRF(rank=rank, loss="frobenius")
    ensemble = EnsembleEmbedding(model, n_runs=n_runs, random_state=42, n_jobs=-1)
    stacked = ensemble.fit_transform(rsm)

    # Reshape to (n_runs, n_samples, rank)
    n_samples = rsm.shape[0]
    embeddings = stacked.reshape(n_samples, n_runs, rank).transpose(1, 0, 2)

    # Individual run statistics
    individual_errors = [recon_error(e, rsm) for e in embeddings]
    log.info(
        f"Individual runs: error={np.mean(individual_errors):.4f} "
        f"(std={np.std(individual_errors):.6f})"
    )

    results = []

    # 1. K-means clustering (cNMF style)
    for outlier_q in [0.9, 0.95, 1.0]:
        emb_kmeans = consensus_kmeans(embeddings, rank, outlier_quantile=outlier_q)
        err = recon_error(emb_kmeans, rsm)
        results.append(
            {
                "method": f"kmeans_q{outlier_q}",
                "error": err,
                "description": f"K-means (outlier q={outlier_q})",
            }
        )
        log.info(f"K-means (q={outlier_q}): error={err:.4f}")

    # 2. Hungarian alignment
    for agg in ["median", "mean"]:
        emb_hung = consensus_hungarian(embeddings, rank, aggregation=agg)
        err = recon_error(emb_hung, rsm)
        results.append(
            {
                "method": f"hungarian_{agg}",
                "error": err,
                "description": f"Hungarian + {agg}",
            }
        )
        log.info(f"Hungarian ({agg}): error={err:.4f}")

    # 3. Selection-based
    emb_select, selected_idx = consensus_select(embeddings, rank)
    err = recon_error(emb_select, rsm)
    results.append(
        {
            "method": "select",
            "error": err,
            "description": f"Select (run {selected_idx})",
        }
    )
    log.info(f"Selection (run {selected_idx}): error={err:.4f}")

    # 4. Single best run (baseline)
    best_idx = np.argmin(individual_errors)
    results.append(
        {
            "method": "single_best",
            "error": individual_errors[best_idx],
            "description": f"Single best (run {best_idx})",
        }
    )
    log.info(f"Single best (run {best_idx}): error={individual_errors[best_idx]:.4f}")

    df = pd.DataFrame(results)

    if output_dir is not None:
        # Save results
        df.to_csv(output_dir / "results.csv", index=False)

        # Create comparison plot
        fig, ax = create_figure("single")

        df_sorted = df.sort_values("error")
        colors = [TEAL if "hungarian" in m or m == "select" else GRAY
                  for m in df_sorted["method"]]

        bars = ax.barh(range(len(df_sorted)), df_sorted["error"], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["description"], fontsize=8)
        ax.set_xlabel("Reconstruction Error")
        ax.set_title(f"Consensus Methods (rank={rank})")

        # Add value labels
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(row["error"] + 0.001, i, f"{row['error']:.4f}",
                   va="center", fontsize=7)

        despine(ax)
        save_figure(fig, output_dir / "comparison.pdf")
        plt.close(fig)

        log.info(f"Saved results to {output_dir}")

    return df


def run_multirank_comparison(
    rsm: np.ndarray,
    ranks: list[int],
    n_runs: int = 50,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare methods across multiple ranks."""
    results = []

    for rank in ranks:
        log.info(f"\nRank {rank}...")
        model = SRF(rank=rank, loss="frobenius")
        ensemble = EnsembleEmbedding(model, n_runs=n_runs, random_state=42, n_jobs=-1)
        stacked = ensemble.fit_transform(rsm)

        n_samples = rsm.shape[0]
        embeddings = stacked.reshape(n_samples, n_runs, rank).transpose(1, 0, 2)

        # Individual best
        errors = [recon_error(e, rsm) for e in embeddings]
        best_err = min(errors)

        # Hungarian
        hung_emb = consensus_hungarian(embeddings, rank, aggregation="median")
        hung_err = recon_error(hung_emb, rsm)

        # K-means
        km_emb = consensus_kmeans(embeddings, rank)
        km_err = recon_error(km_emb, rsm)

        results.append({
            "rank": rank,
            "single_best": best_err,
            "hungarian": hung_err,
            "kmeans": km_err,
            "km_vs_hung_ratio": km_err / hung_err,
        })

        log.info(f"  Single: {best_err:.4f}, Hungarian: {hung_err:.4f}, "
                f"K-means: {km_err:.4f} ({km_err/hung_err:.1f}x worse)")

    df = pd.DataFrame(results)

    if output_dir is not None:
        df.to_csv(output_dir / "multirank_results.csv", index=False)

        # Plot
        fig, ax = create_figure("wide")
        x = np.arange(len(ranks))
        width = 0.25

        ax.bar(x - width, df["single_best"], width, label="Single best", color=GRAY)
        ax.bar(x, df["hungarian"], width, label="Hungarian", color=TEAL)
        ax.bar(x + width, df["kmeans"], width, label="K-means", color=ROSE)

        ax.set_xlabel("Rank")
        ax.set_ylabel("Reconstruction Error")
        ax.set_xticks(x)
        ax.set_xticklabels(ranks)
        ax.legend(fontsize=8)
        ax.set_title("Consensus Methods vs Rank")
        despine(ax)

        save_figure(fig, output_dir / "multirank_comparison.pdf")
        plt.close(fig)

    return df


def main():
    parser = argparse.ArgumentParser(description="Consensus comparison experiment")
    parser.add_argument("--rank", type=int, default=3, help="Embedding rank")
    parser.add_argument("--n-runs", type=int, default=100, help="Number of SRF runs")
    parser.add_argument("--multirank", action="store_true", help="Run multi-rank comparison")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/SSD/datasets/similarity_datasets/mur92",
        help="Path to mur92 dataset",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = get_output_dir()

    # Load data
    log.info(f"Loading mur92 from {args.data_path}...")
    dataset = load_mur92(args.data_path)
    rsm = dataset.rsm
    log.info(f"RSM shape: {rsm.shape}")

    if args.multirank:
        # Multi-rank comparison
        df = run_multirank_comparison(
            rsm=rsm,
            ranks=[3, 5, 10, 15, 20],
            n_runs=50,
            output_dir=output_dir,
        )
        log.info("\n" + "=" * 60)
        log.info("MULTI-RANK SUMMARY")
        log.info("=" * 60)
        log.info(f"\nK-means vs Hungarian ratio: {df['km_vs_hung_ratio'].mean():.1f}x worse on average")
    else:
        # Single rank experiment
        df = run_experiment(
            rsm=rsm,
            rank=args.rank,
            n_runs=args.n_runs,
            output_dir=output_dir,
        )
        log.info("\n" + "=" * 60)
        log.info("SUMMARY")
        log.info("=" * 60)
        log.info("\nMethods ranked by reconstruction error:")
        for i, row in df.sort_values("error").iterrows():
            log.info(f"  {row['description']}: {row['error']:.4f}")

    log.info("\n" + "=" * 60)
    log.info("CONCLUSION")
    log.info("=" * 60)
    log.info("""
For symmetric NMF (SRF):
- Hungarian alignment exploits the permutation-only ambiguity
- K-means clustering is 2-4x WORSE (designed for standard NMF with rotation ambiguity)
- Selection-based approaches give best reconstruction since no averaging
- Consensus provides noise reduction but slightly hurts reconstruction
""")


if __name__ == "__main__":
    main()
