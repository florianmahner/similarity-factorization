from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pysrf import SRF, cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast
from pysrf.consensus import ClusterEmbedding, EnsembleEmbedding
from sklearn.pipeline import Pipeline

from similarity import build_similarity, get_rank_grid


def _plot_results(
    output_dir: Path,
    cv_results: pd.DataFrame,
    cluster_results,
    best_rank: int,
    best_k: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    grouped = cv_results.groupby("rank")["score"].agg(["mean", "std"])
    ranks = grouped.index.values
    means = grouped["mean"].values
    stds = grouped["std"].values

    axes[0].plot(ranks, means, "o-", linewidth=2, markersize=4)
    axes[0].fill_between(ranks, means - stds, means + stds, alpha=0.3)
    axes[0].axvline(best_rank, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("rank", fontsize=7)
    axes[0].set_ylabel("cv score (mse)", fontsize=7)
    axes[0].set_title("Cross-validation results", fontsize=7)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        cluster_results["n_clusters"],
        cluster_results["silhouette_score"],
        "o-",
        linewidth=2,
        markersize=4,
    )
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("number of clusters", fontsize=7)
    axes[1].set_ylabel("silhouette score", fontsize=7)
    axes[1].set_title("Clustering results", fontsize=7)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "analysis_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def run(cfg: DictConfig) -> None:
    similarity = build_similarity(cfg.dataset, subject_id=cfg.generate.subject_id)
    ds_cfg = cfg.dataset
    rank_grid = get_rank_grid(ds_cfg)

    # Try to find bounds from the 'bounds' experiment output
    # This logic assumes a specific directory structure that might need adjustment
    # For now, we'll recalculate if not found or just assume defaults if we want to decouple
    
    # Recalculate bounds for self-contained run (safer than hardcoded path assumptions)
    pmin, pmax, _ = estimate_sampling_bounds_fast(
        similarity,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=False,
    )
    sampling_fraction = 0.5 * (pmin + pmax)

    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=cfg.generate.n_cv_repeats,
        sampling_fraction=sampling_fraction,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=0,
    )
    optimal_rank = cv_result.best_params_["rank"]

    min_k = max(2, optimal_rank - 3)
    max_k = optimal_rank + 4
    pipeline = Pipeline(
        [
            (
                "ensemble",
                EnsembleEmbedding(
                    SRF(rank=optimal_rank, random_state=cfg.common.random_state),
                    n_runs=cfg.generate.n_stable_runs,
                    random_state=cfg.common.random_state,
                    n_jobs=cfg.common.n_jobs,
                ),
            ),
            (
                "cluster",
                ClusterEmbedding(
                    min_clusters=min_k,
                    max_clusters=max_k,
                    step=1,
                    random_state=cfg.common.random_state,
                    n_jobs=cfg.common.n_jobs,
                ),
            ),
        ]
    )
    pipeline.fit(similarity)
    final_embedding = pipeline.transform(similarity)

    stacked_embeddings = pipeline.named_steps["ensemble"].embeddings_
    cluster_results = pipeline.named_steps["cluster"].cluster_results_
    best_k = pipeline.named_steps["cluster"].best_k_

    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "dataset": cfg.dataset.get("name", "unknown"),
        "n_samples": int(final_embedding.shape[0]),
        "optimal_rank": int(optimal_rank),
        "optimal_clusters": int(best_k),
        "best_cv_score": float(cv_result.best_score_),
        "n_stable_runs": int(cfg.generate.n_stable_runs),
        "n_cv_repeats": int(cfg.generate.n_cv_repeats),
        "random_state": int(cfg.common.random_state),
        "p_min": float(pmin),
        "p_max": float(pmax),
        "observed_fraction": float(sampling_fraction),
    }
    if cfg.generate.subject_id is not None:
        meta["subject_id"] = int(cfg.generate.subject_id)
    (output_dir / "summary.json").write_text(json.dumps(meta, indent=2))

    _plot_results(output_dir, cv_result.cv_results_, cluster_results, optimal_rank, best_k)
