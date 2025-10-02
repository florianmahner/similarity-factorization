"""Simplified embedding pipeline using pysrf."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pysrf import cross_val_score, fit_stable, cluster_stable

ndarray = np.ndarray


def run_embedding_pipeline(
    similarity_matrix: ndarray,
    rank_grid: list[int] | None = None,
    n_runs: int = 50,
    n_repeats: int = 5,
    random_state: int = 0,
    n_jobs: int = -1,
    estimate_sampling_fraction: bool = True,
    cluster_min: int | None = None,
    cluster_max: int | None = None,
    cluster_step: int = 2,
    rho: float = 3.0,
    max_outer: int = 15,
    max_inner: int = 40,
    tol: float = 1e-4,
    init: str = "random_sqrt",
    output_dir: str | Path | None = None,
    name: str = "model",
    verbose: int = 0,
) -> dict:
    """
    Complete pipeline: rank selection, stable ensemble fitting, and clustering.

    This function provides a simplified interface to the full SRF pipeline using
    the new pysrf API with automatic sampling fraction estimation and stability.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix
    rank_grid : list[int], optional
        Ranks to search over. If None, uses range(5, 50)
    n_runs : int, default=50
        Number of stable runs with different initializations
    n_repeats : int, default=5
        Number of CV repeats for rank selection
    random_state : int, default=0
        Random seed
    n_jobs : int, default=-1
        Number of parallel jobs
    estimate_sampling_fraction : bool, default=True
        Whether to auto-estimate optimal sampling fraction
    cluster_min : int, optional
        Minimum clusters to try. If None, uses best_rank // 2
    cluster_max : int, optional
        Maximum clusters to try. If None, uses min(best_rank * 2, 100)
    cluster_step : int, default=2
        Step size for cluster search
    rho : float, default=3.0
        ADMM penalty parameter
    max_outer : int, default=15
        Maximum outer ADMM iterations
    max_inner : int, default=40
        Maximum inner iterations
    tol : float, default=1e-4
        Convergence tolerance
    init : str, default='random_sqrt'
        Initialization method
    output_dir : Path, optional
        Directory to save results
    name : str, default='model'
        Name prefix for saved files
    verbose : int, default=0
        Verbosity level

    Returns
    -------
    results : dict
        Dictionary containing:
            - sampling_fraction : estimated sampling fraction
            - optimal_rank : best rank from CV
            - best_k : optimal number of clusters
            - final_embedding : clustered embedding (n_samples, best_k)
            - stacked_embeddings : all embeddings (n_samples, rank * n_runs)
            - cv_results : CV results DataFrame
            - cluster_results : clustering results DataFrame
            - cv_object : fitted ADMMGridSearchCV object

    Examples
    --------
    >>> from src.embedding_pipeline import run_embedding_pipeline
    >>> results = run_embedding_pipeline(
    ...     similarity_matrix,
    ...     rank_grid=list(range(5, 50)),
    ...     n_runs=50
    ... )
    >>> embedding = results['final_embedding']
    """
    if rank_grid is None:
        rank_grid = list(range(5, 50))

    cluster_kwargs = {"step": cluster_step}
    if cluster_min is not None:
        cluster_kwargs["min_clusters"] = cluster_min
    if cluster_max is not None:
        cluster_kwargs["max_clusters"] = cluster_max

    result = cross_val_score(
        similarity_matrix,
        param_grid={
            "rank": rank_grid,
            "rho": [rho],
            "max_outer": [max_outer],
            "max_inner": [max_inner],
            "tol": [tol],
            "init": [init],
        },
        n_repeats=n_repeats,
        estimate_sampling_fraction=estimate_sampling_fraction,
        fit_final_estimator=True,
        n_stable_runs=n_runs,
        cluster_stable=cluster_kwargs,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    best_rank = result.best_params_["rank"]
    if cluster_min is None:
        cluster_kwargs["min_clusters"] = max(2, best_rank // 2)
    if cluster_max is None:
        cluster_kwargs["max_clusters"] = min(best_rank * 2, 100)

    if hasattr(result, "clustered_embedding_"):
        final = result.clustered_embedding_
        best_k = result.best_k_
        cluster_df = result.cluster_results_
    else:
        final, best_k, cluster_df = cluster_stable(
            result.stable_embeddings_,
            random_state=random_state,
            n_jobs=n_jobs,
            **cluster_kwargs,
        )

    out = {
        "sampling_fraction": result.best_score_,
        "optimal_rank": best_rank,
        "best_k": best_k,
        "final_embedding": final,
        "stacked_embeddings": result.stable_embeddings_,
        "cv_results": pd.DataFrame(result.cv_results_),
        "cluster_results": cluster_df,
        "cv_object": result,
    }

    if output_dir is not None:
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / f"{name}_stacked.npy", result.stable_embeddings_)
        np.save(path / f"{name}_embedding.npy", final)
        cluster_df.to_csv(path / f"{name}_clustering.csv", index=False)
        pd.DataFrame(result.cv_results_).to_csv(path / f"{name}_cv.csv", index=False)

        metadata = {
            "optimal_rank": int(best_rank),
            "best_k": int(best_k),
            "n_runs": int(n_runs),
        }
        with open(path / f"{name}_meta.json", "w") as f:
            json.dump(metadata, f)

    return out
