#!/usr/bin/env python
"""Generate consensus embeddings using similarity representation factorization."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from config import get_dataset_path
from datasets import load_dataset
from pysrf import SRF, cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast
from pysrf.consensus import ClusterEmbedding, EnsembleEmbedding
from tools.rsa import compute_similarity

matplotlib.use("Agg")

SMALL_DATASETS = {"mur92", "cichy118", "peterson-animals", "peterson-various"}
LARGE_DATASETS = {"nsd", "things-monkey-22k", "vit", "things-ooo"}


def get_rank_grid(dataset_slug: str) -> list[int]:
    if dataset_slug in SMALL_DATASETS:
        return list(range(1, 31))
    if dataset_slug in LARGE_DATASETS:
        return list(range(5, 101, 5))
    return list(range(5, 101, 5))


def build_similarity(dataset_slug: str, subject_id: int | None) -> np.ndarray:
    if dataset_slug == "nsd":
        ds = load_dataset(
            "nsd", subject_id=subject_id, roi_name="streams", space="func1pt8mm"
        )
        return compute_similarity(ds.data, ds.data, "gaussian_kernel")

    if dataset_slug == "things-monkey-22k":
        ds = load_dataset(
            "things-monkey-22k", min_reliab=0.3, monkey_type="F", roi="it"
        )
        if hasattr(ds, "rsm"):
            return ds.rsm
        return compute_similarity(ds.it, ds.it, "gaussian_kernel")

    if dataset_slug in SMALL_DATASETS:
        return load_dataset(dataset_slug).rsm

    if dataset_slug == "vit":
        vit_path = get_dataset_path("vit")
        features = np.load(f"{vit_path}/features.npy")
        import pandas as pd

        image_info = pd.read_csv("/SSD/projects/deepsim/raw/features/image_info.csv")
        # find where filename contains _plus
        plus_indices = image_info[image_info["filename"].str.contains("_plus")].index

        features = features[plus_indices, :]
        return compute_similarity(features, features, "gaussian_kernel")

    if dataset_slug == "things-ooo":
        from utils.io import load_triplets
        from analysis.things.common import compute_similarity_matrix_from_triplets

        train_triplets, validation_triplets = load_triplets(
            Path(get_dataset_path("things-ooo")), number="4.7mio"
        )
        sim = compute_similarity_matrix_from_triplets(1854, train_triplets)
        return sim

    raise ValueError(f"unknown dataset: {dataset_slug}")


def plot_results(output_dir: Path, cv_results, cluster_results, best_rank, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    grouped = cv_results.groupby("rank")["score"].agg(["mean", "std"])
    ranks = grouped.index.values
    means = grouped["mean"].values
    stds = grouped["std"].values

    axes[0].plot(ranks, means, "o-", linewidth=2, markersize=4)
    axes[0].fill_between(ranks, means - stds, means + stds, alpha=0.3)
    axes[0].axvline(
        best_rank,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"optimal rank: {best_rank}",
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


def run_analysis(
    dataset_slug: str,
    output_dir: Path,
    subject_id: int | None = None,
    n_jobs: int = -1,
    random_state: int = 0,
    n_cv_repeats: int = 5,
    n_stable_runs: int = 50,
) -> None:
    print(f"running analysis for {dataset_slug}", flush=True)
    if subject_id is not None:
        print(f"subject {subject_id}", flush=True)

    similarity = build_similarity(dataset_slug, subject_id)
    print(f"similarity matrix shape: {similarity.shape}", flush=True)

    rank_grid = get_rank_grid(dataset_slug)
    print(f"rank grid: {rank_grid[0]} to {rank_grid[-1]}", flush=True)

    # check if in experiments/bounds/outputs/dataset_slug/bounds.json exists
    bounds_file = Path(f"experiments/bounds/outputs/{dataset_slug}/bounds.json")
    if bounds_file.exists():
        with open(bounds_file, "r") as f:
            bounds = json.load(f)
        pmin = bounds["pmin"]
        pmax = bounds["pmax"]
        sampling_fraction = bounds["mean_sampling_fraction"]
    else:
        print(f"sampling bounds not found, computing them", flush=True)
        pmin, pmax, _ = estimate_sampling_bounds_fast(
            similarity, random_state=random_state, n_jobs=n_jobs, verbose=True
        )
        sampling_fraction = 0.5 * (pmin + pmax)
        print(f"sampling bounds: pmin={pmin:.4f}, pmax={pmax:.4f}", flush=True)
        print(f"using sampling_fraction={sampling_fraction:.4f}", flush=True)

    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=n_cv_repeats,
        sampling_fraction=sampling_fraction,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    optimal_rank = cv_result.best_params_["rank"]
    print(f"optimal rank: {optimal_rank}", flush=True)

    min_k = max(2, optimal_rank - 3)
    max_k = optimal_rank + 4
    print(f"clustering range: {min_k} to {max_k}", flush=True)

    pipeline = Pipeline(
        [
            (
                "ensemble",
                EnsembleEmbedding(
                    SRF(rank=optimal_rank, random_state=random_state),
                    n_runs=n_stable_runs,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "cluster",
                ClusterEmbedding(
                    min_clusters=min_k,
                    max_clusters=max_k,
                    step=1,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    print(f"fitting consensus pipeline with {n_stable_runs} runs", flush=True)
    pipeline.fit(similarity)
    final_embedding = pipeline.transform(similarity)

    stacked_embeddings = pipeline.named_steps["ensemble"].embeddings_
    cluster_results = pipeline.named_steps["cluster"].cluster_results_
    best_k = pipeline.named_steps["cluster"].best_k_

    print(f"optimal clusters: {best_k}", flush=True)

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
        "dataset": dataset_slug,
        "n_samples": int(final_embedding.shape[0]),
        "optimal_rank": int(optimal_rank),
        "optimal_clusters": int(best_k),
        "best_cv_score": float(cv_result.best_score_),
        "n_stable_runs": int(n_stable_runs),
        "n_cv_repeats": int(n_cv_repeats),
        "random_state": int(random_state),
        "p_min": float(pmin),
        "p_max": float(pmax),
        "observed_fraction": float(sampling_fraction),
    }

    if subject_id is not None:
        meta["subject_id"] = int(subject_id)

    (output_dir / "summary.json").write_text(json.dumps(meta, indent=2))

    plot_results(
        output_dir, cv_result.cv_results_, cluster_results, optimal_rank, best_k
    )

    print(f"results saved to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="embedding generation pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(SMALL_DATASETS | LARGE_DATASETS),
        help="dataset to analyze",
    )
    parser.add_argument(
        "--subject-id", type=int, default=None, help="subject id for nsd dataset"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="output directory for results"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="number of parallel jobs"
    )
    parser.add_argument("--random-state", type=int, default=0, help="random seed")
    parser.add_argument(
        "--n-cv-repeats", type=int, default=10, help="number of cv repeats"
    )
    parser.add_argument(
        "--n-stable-runs", type=int, default=30, help="number of stable runs"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.output_dir is None:
        args.output_dir = script_dir / "outputs"

    args.output_dir = args.output_dir.expanduser().resolve()

    if args.dataset == "nsd":
        if args.subject_id is None:
            raise ValueError("subject_id required for nsd dataset")
        dataset_output = args.output_dir / "nsd" / f"subj{args.subject_id:02d}"
    else:
        dataset_output = args.output_dir / args.dataset

    run_analysis(
        args.dataset,
        dataset_output,
        args.subject_id,
        args.n_jobs,
        args.random_state,
        args.n_cv_repeats,
        args.n_stable_runs,
    )


if __name__ == "__main__":
    main()
