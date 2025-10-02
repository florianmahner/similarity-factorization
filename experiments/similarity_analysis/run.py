import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from config import get_dataset_path
from datasets.nsd_utils import get_available_subjects
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_fast
from tools.rsa import compute_similarity

SMALL_DATASETS = {"mur92", "cichy118", "peterson-animals", "peterson-various"}
LARGE_DATASETS = {"nsd", "things-monkey-22k", "vit"}


def get_rank_grid(dataset_slug: str) -> list[int]:
    if dataset_slug in SMALL_DATASETS:
        return list(range(1, 31))
    if dataset_slug in LARGE_DATASETS:
        return list(range(5, 151, 5))
    return list(range(5, 151, 5))


def build_similarity(dataset_slug: str, subject_id: int | None) -> np.ndarray:
    if dataset_slug == "nsd":
        ds = load_dataset(
            "nsd", subject_id=subject_id, roi_name="streams", space="func1pt8mm"
        )
        return compute_similarity(ds.betas, ds.betas, "gaussian_kernel")
    
    if dataset_slug == "things-monkey-22k":
        ds = load_dataset(
            "things-monkey-22k", min_reliab=0.3, monkey_type="F", roi="it"
        )
        if hasattr(ds, "rsm"):
            return ds.rsm
        return compute_similarity(ds.it, ds.it, "gaussian_kernel")
    
    if dataset_slug == "mur92":
        ds = load_dataset("mur92")
        return ds.group_rsm
    
    if dataset_slug == "cichy118":
        ds = load_dataset("cichy118")
        return ds.group_rsm
    
    if dataset_slug == "peterson-animals":
        ds = load_dataset("peterson-animals")
        return ds.rsm
    
    if dataset_slug == "peterson-various":
        ds = load_dataset("peterson-various")
        return ds.rsm
    
    if dataset_slug == "vit":
        vit_path = get_dataset_path("vit")
        features = np.load(f"{vit_path}/features.npy")
        return compute_similarity(features, features, "cosine")
    
    raise ValueError(f"Unknown dataset slug: {dataset_slug}")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def plot_results(output_dir: Path, cv_result) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    
    cv_df = cv_result.cv_results_
    grouped = cv_df.groupby('rank')['score'].agg(['mean', 'std'])
    ranks = grouped.index.values
    means = grouped['mean'].values
    stds = grouped['std'].values
    
    axes[0].plot(ranks, means, 'o-', linewidth=2, markersize=4)
    axes[0].fill_between(ranks, means - stds, means + stds, alpha=0.3)
    axes[0].axvline(cv_result.best_params_['rank'], color='red', linestyle='--', 
                    linewidth=1, label=f"optimal rank: {cv_result.best_params_['rank']}")
    axes[0].set_xlabel('rank', fontsize=7)
    axes[0].set_ylabel('cv score (mse)', fontsize=7)
    axes[0].set_title('cross-validation results', fontsize=7)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)
    
    cluster_df = cv_result.cluster_results_
    axes[1].plot(cluster_df['n_clusters'], cluster_df['silhouette_score'], 
                 'o-', linewidth=2, markersize=4)
    axes[1].axvline(cv_result.best_k_, color='red', linestyle='--', 
                    linewidth=1, label=f"optimal k: {cv_result.best_k_}")
    axes[1].set_xlabel('number of clusters', fontsize=7)
    axes[1].set_ylabel('silhouette score', fontsize=7)
    axes[1].set_title('clustering results', fontsize=7)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_results(
    output_dir: Path,
    similarity: np.ndarray,
    cv_result,
    dataset_slug: str,
    subject_id: int | None,
    random_state: int,
    n_stable_runs: int,
    n_cv_repeats: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    

    cv_result.cv_results_.to_csv(output_dir / "cv_results.csv", index=False)
    
    cv_results_clean = {
        'cv_results_': cv_result.cv_results_,
        'best_params_': cv_result.best_params_,
        'best_score_': cv_result.best_score_,
    }
    joblib.dump(cv_results_clean, output_dir / "cv_results.joblib")
    
    cv_result.cluster_results_.to_csv(
        output_dir / "clustering_results.csv", index=False
    )

    np.save(output_dir / "stacked_embeddings.npy", cv_result.stable_embeddings_)
    
    np.save(output_dir / "consensus_embedding.npy", cv_result.clustered_embedding_)
    
    meta = {
        "dataset": dataset_slug,
        "n_samples": int(cv_result.clustered_embedding_.shape[0]),
        "optimal_rank": int(cv_result.best_params_["rank"]),
        "optimal_clusters": int(cv_result.best_k_),
        "best_cv_score": float(cv_result.best_score_),
        "n_stable_runs": int(n_stable_runs),
        "n_cv_repeats": int(n_cv_repeats),
        "random_state": int(random_state),
        "p_min": float(cv_result.p_min_),
        "p_max": float(cv_result.p_max_),
        "observed_fraction": float(cv_result.sampling_fraction_),
    }
    
    if subject_id is not None:
        meta["subject_id"] = int(subject_id)
    
    save_json(output_dir / "summary.json", meta)
    
    plot_results(output_dir, cv_result)


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
    
    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=n_cv_repeats,
        estimate_sampling_fraction=True,
        fit_final_estimator=True,
        n_stable_runs=n_stable_runs,
        cluster_stable=False,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    
    print(f"optimal rank: {cv_result.best_params_['rank']}", flush=True)
    
    from pysrf.stability import cluster_stable
    optimal_rank = cv_result.best_params_["rank"]
    min_k = max(2, optimal_rank - 5)
    max_k = optimal_rank + 5
    
    print(f"clustering range: {min_k} to {max_k}", flush=True)
    
    cv_result.clustered_embedding_, cv_result.best_k_, cv_result.cluster_results_ = cluster_stable(
        cv_result.stable_embeddings_,
        min_clusters=min_k,
        max_clusters=max_k,
        step=1,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    
    print(f"optimal clusters: {cv_result.best_k_}", flush=True)
    
    save_results(
        output_dir,
        similarity,
        cv_result,
        dataset_slug,
        subject_id,
        random_state,
        n_stable_runs,
        n_cv_repeats,
    )
    
    print(f"results saved to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="similarity representation analysis pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(SMALL_DATASETS | LARGE_DATASETS),
        help="dataset to analyze",
    )
    parser.add_argument(
        "--subject_id",
        type=int,
        default=None,
        help="subject id for nsd dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="output directory for results",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument("--random_state", type=int, default=0, help="random seed")
    parser.add_argument(
        "--n_cv_repeats", type=int, default=5, help="number of cv repeats for rank selection"
    )
    parser.add_argument(
        "--n_stable_runs",
        type=int,
        default=50,
        help="number of stable runs for consensus embedding",
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        script_dir = Path(__file__).parent
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
