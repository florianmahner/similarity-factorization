#!/usr/bin/env python
"""Debug script for testing embedding generation pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from datasets import load_dataset
from pysrf import SRF, cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast
from pysrf.consensus import ClusterEmbedding, EnsembleEmbedding

matplotlib.use("Agg")


def main():
    print("loading peterson-animals dataset")
    ds = load_dataset("peterson-animals")
    similarity = ds.rsm

    print(f"similarity shape: {similarity.shape}")

    rank_grid = list(range(1, 11))
    print(f"rank grid: {rank_grid}")

    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity, random_state=0, n_jobs=2, verbose=False
    )
    sampling_fraction = 0.5 * (pmin + pmax)
    print(f"pmin={pmin:.4f}, pmax={pmax:.4f}, fraction={sampling_fraction:.4f}")

    print("running cross-validation")
    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=2,
        sampling_fraction=sampling_fraction,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=0,
        n_jobs=2,
        verbose=0,
    )

    optimal_rank = cv_result.best_params_["rank"]
    print(f"optimal rank: {optimal_rank}")

    min_k = max(2, optimal_rank - 1)
    max_k = optimal_rank + 2

    print(f"building consensus pipeline (rank={optimal_rank}, n_runs=3)")
    pipeline = Pipeline(
        [
            (
                "ensemble",
                EnsembleEmbedding(
                    SRF(rank=optimal_rank, random_state=0),
                    n_runs=3,
                    random_state=0,
                    n_jobs=2,
                ),
            ),
            (
                "cluster",
                ClusterEmbedding(
                    min_clusters=min_k,
                    max_clusters=max_k,
                    step=1,
                    random_state=0,
                    n_jobs=2,
                ),
            ),
        ]
    )

    print("fitting pipeline")
    pipeline.fit(similarity)
    final_embedding = pipeline.transform(similarity)

    stacked_embeddings = pipeline.named_steps["ensemble"].embeddings_
    cluster_results = pipeline.named_steps["cluster"].cluster_results_
    best_k = pipeline.named_steps["cluster"].best_k_

    print(f"optimal clusters: {best_k}")
    print(f"stacked shape: {stacked_embeddings.shape}")
    print(f"final shape: {final_embedding.shape}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = Path(__file__).parent.parent.parent
    output_dir = repo_root / "experiments" / "embedding_generation" / "outputs" / "debug"
    log_dir = repo_root / "experiments" / "embedding_generation" / "logs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cv_results_clean = {
        "cv_results_": cv_result.cv_results_,
        "best_params_": cv_result.best_params_,
        "best_score_": cv_result.best_score_,
    }
    joblib.dump(cv_results_clean, output_dir / "cv_results.joblib")
    joblib.dump(cluster_results, output_dir / "clustering_results.joblib")
    np.save(output_dir / "stacked_embeddings.npy", stacked_embeddings)
    np.save(output_dir / "consensus_embedding.npy", final_embedding)

    meta = {
        "dataset": "peterson-animals",
        "n_samples": int(final_embedding.shape[0]),
        "optimal_rank": int(optimal_rank),
        "optimal_clusters": int(best_k),
        "best_cv_score": float(cv_result.best_score_),
        "n_stable_runs": 3,
        "n_cv_repeats": 2,
        "random_state": 0,
        "p_min": float(pmin),
        "p_max": float(pmax),
        "observed_fraction": float(sampling_fraction),
    }

    (output_dir / "summary.json").write_text(json.dumps(meta, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    grouped = cv_result.cv_results_.groupby("rank")["score"].agg(["mean", "std"])
    ranks = grouped.index.values
    means = grouped["mean"].values
    stds = grouped["std"].values

    axes[0].plot(ranks, means, "o-", linewidth=2, markersize=4)
    axes[0].fill_between(ranks, means - stds, means + stds, alpha=0.3)
    axes[0].axvline(
        optimal_rank,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"optimal rank: {optimal_rank}",
    )
    axes[0].set_xlabel("rank", fontsize=7)
    axes[0].set_ylabel("cv score (mse)", fontsize=7)
    axes[0].set_title("Cross-validation results", fontsize=7)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        cluster_results["n_clusters"],
        cluster_results["silhouette_score"],
        "o-",
        linewidth=2,
        markersize=4,
    )
    axes[1].axvline(
        best_k,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"optimal k: {best_k}",
    )
    axes[1].set_xlabel("number of clusters", fontsize=7)
    axes[1].set_ylabel("silhouette score", fontsize=7)
    axes[1].set_title("Clustering results", fontsize=7)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "analysis_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nresults saved to {output_dir}")
    print(f"logs directory: {log_dir}")
    print("\nverify outputs match expected structure:")
    print(f"  - cv_results.joblib")
    print(f"  - clustering_results.joblib")
    print(f"  - stacked_embeddings.npy")
    print(f"  - consensus_embedding.npy")
    print(f"  - summary.json")
    print(f"  - analysis_summary.png")


if __name__ == "__main__":
    main()

