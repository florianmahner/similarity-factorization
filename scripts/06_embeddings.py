""" Here we want to actually find the optimal dimensionality for the embeddings.
We will use the cross validation score to find the optimal dimensionality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from datasets import load_dataset
from models.admm import ADMM
from cross_validation import cross_validate_admm
from experiments.cluster_embeddings import cluster_stacked_embeddings

num_runs = 100


class DimensionalityEstimator(BaseEstimator, TransformerMixin):
    """Sklearn-compatible estimator for finding optimal dimensionality via cross-validation."""

    def __init__(
        self,
        rank_range: tuple[int, int] = (5, 50),
        n_repeats: int = 5,
        train_ratio: float = 0.8,
        random_state: int = 0,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        self.rank_range = rank_range
        self.n_repeats = n_repeats
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x, y=None):
        """Find optimal dimensionality using cross-validation."""
        min_rank, max_rank = self.rank_range
        param_grid = {"rank": list(range(min_rank, max_rank + 1, 2))}

        if self.verbose:
            print(f"Testing ranks: {param_grid['rank']}")

        results = cross_validate_admm(
            similarity_matrix=x,
            param_grid=param_grid,
            n_repeats=self.n_repeats,
            train_ratio=self.train_ratio,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.best_rank_ = results.best_params_["rank"]
        self.best_score_ = results.best_score_
        self.cv_results_ = results.cv_results_

        if self.verbose:
            print(f"Best rank: {self.best_rank_} (CV score: {self.best_score_:.4f})")

        return self

    def transform(self, x):
        """Return the similarity matrix unchanged - this is for pipeline compatibility."""
        return x

    def get_optimal_rank(self):
        """Get the optimal rank found during fitting."""
        if not hasattr(self, "best_rank_"):
            raise ValueError("Must call fit() first")
        return self.best_rank_


class EmbeddingFactorizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for multiple ADMM factorizations."""

    def __init__(
        self,
        rank: int = None,
        n_runs: int = 100,
        rho: float = 1.0,
        max_outer: int = 15,
        max_inner: int = 40,
        tol: float = 1e-4,
        init: str = "random_sqrt",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.rank = rank
        self.n_runs = n_runs
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.init = init
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x, y=None):
        """Fit multiple ADMM factorizations."""
        if self.rank is None:
            raise ValueError("rank must be specified")

        def fit_single_run(seed):
            """Fit a single ADMM model with given seed."""
            model = ADMM(
                rank=self.rank,
                rho=self.rho,
                max_outer=self.max_outer,
                max_inner=self.max_inner,
                tol=self.tol,
                init=self.init,
                random_state=seed,
                verbose=False,
            )
            w = model.fit_transform(x)
            return w

        if self.verbose:
            print(f"Running {self.n_runs} factorizations with rank {self.rank}")

        # Run factorizations in parallel
        embeddings = Parallel(n_jobs=self.n_jobs, verbose=1 if self.verbose else 0)(
            delayed(fit_single_run)(seed) for seed in range(self.n_runs)
        )

        # Stack embeddings horizontally
        self.stacked_embeddings_ = np.hstack(embeddings)

        return self

    def transform(self, x):
        """Return the stacked embeddings."""
        if not hasattr(self, "stacked_embeddings_"):
            raise ValueError("Must call fit() first")
        return self.stacked_embeddings_


def estimate_dimensionality(dataset, rank_range=(5, 50), **kwargs):
    """Estimate optimal dimensionality for a dataset using cross-validation."""
    estimator = DimensionalityEstimator(rank_range=rank_range, **kwargs)

    # Get similarity matrix from dataset
    if hasattr(dataset, "similarity_matrix"):
        sim_matrix = dataset.similarity_matrix
    elif hasattr(dataset, "x"):
        # If we have raw data, we need to compute similarity
        from tools.rsa import compute_similarity

        sim_matrix = compute_similarity(dataset.x, dataset.x, "linear")
    else:
        raise ValueError("Dataset must have 'similarity_matrix' or 'x' attribute")

    estimator.fit(sim_matrix)
    return estimator


def factorize_dataset(dataset, rank=None, n_runs=100, **kwargs):
    """Run multiple factorizations on a dataset."""
    factorizer = EmbeddingFactorizer(rank=rank, n_runs=n_runs, **kwargs)

    # Get similarity matrix from dataset
    if hasattr(dataset, "similarity_matrix"):
        sim_matrix = dataset.similarity_matrix
    elif hasattr(dataset, "x"):
        from tools.rsa import compute_similarity

        sim_matrix = compute_similarity(dataset.x, dataset.x, "linear")
    else:
        raise ValueError("Dataset must have 'similarity_matrix' or 'x' attribute")

    factorizer.fit(sim_matrix)
    return factorizer.stacked_embeddings_


def create_embedding_pipeline(rank_range=(5, 50), n_runs=100, **kwargs):
    """Create a sklearn pipeline for dimensionality estimation -> factorization -> clustering."""
    pipeline = Pipeline(
        [
            (
                "dimensionality",
                DimensionalityEstimator(rank_range=rank_range, **kwargs),
            ),
            ("factorizer", EmbeddingFactorizer(n_runs=n_runs, **kwargs)),
        ]
    )

    return pipeline


def run_full_analysis(
    datasets=None,
    rank_range=(5, 50),
    n_runs=100,
    output_dir="./results/embeddings",
    verbose=True,
    **kwargs,
):
    """Run complete analysis: dimensionality estimation, factorization, and clustering."""

    if datasets is None:
        # Load default datasets
        datasets = {
            "peterson-animals": load_dataset("peterson-animals"),
            "peterson-various": load_dataset("peterson-various"),
            "mur92": load_dataset("mur92"),
            "cichy118": load_dataset("cichy118"),
            "nsd": load_dataset("nsd"),
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_embeddings = {}

    for dataset_name, dataset in datasets.items():
        if verbose:
            print(f"\n=== Processing {dataset_name} ===")

        # Step 1: Estimate dimensionality
        dim_estimator = estimate_dimensionality(
            dataset, rank_range=rank_range, verbose=verbose, **kwargs
        )
        optimal_rank = dim_estimator.get_optimal_rank()

        # Step 2: Run multiple factorizations with optimal rank
        stacked_embeddings = factorize_dataset(
            dataset, rank=optimal_rank, n_runs=n_runs, verbose=verbose, **kwargs
        )

        # Step 3: Cluster the embeddings
        embeddings_dict = {dataset_name: stacked_embeddings}
        cluster_results, clustered_embeddings = cluster_stacked_embeddings(
            embeddings_dict,
            min_clusters=max(2, optimal_rank // 2),
            max_clusters=min(optimal_rank * 2, 100),
            step=2,
        )

        # Store results
        dataset_results = {
            "dataset": dataset_name,
            "optimal_rank": optimal_rank,
            "cv_score": dim_estimator.best_score_,
            "n_runs": n_runs,
            "clustering_results": cluster_results,
        }
        all_results.append(dataset_results)
        all_embeddings[dataset_name] = clustered_embeddings[dataset_name]

        # Save individual results
        cluster_results.to_csv(
            output_path / f"{dataset_name}_clustering.csv", index=False
        )
        np.save(
            output_path / f"{dataset_name}_embeddings.npy",
            clustered_embeddings[dataset_name],
        )

        if verbose:
            print(f"Optimal rank: {optimal_rank}")
            print(f"CV score: {dim_estimator.best_score_:.4f}")
            print(f"Best clustering k: {cluster_results['best_k'].iloc[0]}")

    # Save summary results
    summary_df = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "optimal_rank": r["optimal_rank"],
                "cv_score": r["cv_score"],
                "n_runs": r["n_runs"],
            }
            for r in all_results
        ]
    )
    summary_df.to_csv(output_path / "dimensionality_summary.csv", index=False)

    if verbose:
        print(f"\nResults saved to {output_path}")
        print("\nSummary:")
        print(summary_df)

    return all_results, all_embeddings


if __name__ == "__main__":
    # Example usage
    results, embeddings = run_full_analysis(
        rank_range=(5, 30), n_runs=50, verbose=True, n_jobs=-1  # Reduced for testing
    )
