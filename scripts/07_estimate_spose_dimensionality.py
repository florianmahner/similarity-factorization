"""
Dimension estimation experiment for SPoSE data using cross-validation on observed entries.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from cross_validation import cross_val_score
from experiments.things.things import compute_similarity_matrix_from_triplets


def estimate_dimensions_from_triplets(
    triplets: np.ndarray,
    n_items: int,
    rank_range: range | None = None,
    n_repeats: int = 3,
    observed_fraction: float = 0.8,
    n_jobs: int = -1,
    verbose: int = 0,
    random_state: int = 42,
) -> dict:

    if rank_range is None:
        rank_range = range(2, min(51, n_items // 2))

    similarity = compute_similarity_matrix_from_triplets(n_items, triplets)

    param_grid = {
        "rank": list(rank_range),
        "max_outer": [10],
        "max_inner": [100],
        "tol": [0.0],
        "rho": [1.0],
    }
    cv_results = cross_val_score(
        similarity,
        param_grid=param_grid,
        n_repeats=n_repeats,
        observed_fraction=observed_fraction,
        random_state=random_state,
        verbose=verbose,
        n_jobs=-1,
        missing_values=np.nan,
        fit_final_estimator=False,
    )

    best_rank = cv_results.best_params_["rank"]
    best_rmse = cv_results.best_score_

    return {
        "best_rank": best_rank,
        "best_rmse": best_rmse,
        "cv_results": cv_results.cv_results_,
        "rank_range": rank_range,
    }


def dimension_estimation_experiment(
    triplets_path: Path,
    n_items: int = 1854,
    rank_range: range = None,
    output_path: Path = None,
    **cv_kwargs,
) -> Dict:
    """Run dimension estimation experiment on SPoSE triplet data."""
    if output_path is None:
        output_path = Path("results/spose/dimension_estimation.csv")

    if rank_range is None:
        rank_range = range(2, 51)

    triplets = np.loadtxt(triplets_path).astype(int)

    results = estimate_dimensions_from_triplets(
        triplets=triplets,
        n_items=n_items,
        rank_range=rank_range,
        **cv_kwargs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["cv_results"].to_csv(output_path, index=False)

    summary = {
        "dataset": triplets_path.name,
        "n_triplets": results["n_triplets"],
        "n_items": results["n_items"],
        "best_rank": results["best_rank"],
        "best_rmse": results["best_rmse"],
        "rank_range_min": min(rank_range),
        "rank_range_max": max(rank_range),
    }

    summary_path = (
        output_path.parent / f"dimension_estimation_summary_{triplets_path.stem}.csv"
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return results


def run_spose_dimension_analysis():
    """Run SPoSE dimension estimation analysis on the training dataset only."""
    data_dir = Path("data/things/triplets")
    output_dir = Path("./results/spose/dimension_estimation")
    triplet_path = data_dir / "trainset.txt"

    if not triplet_path.exists():
        raise FileNotFoundError(f"Training set not found: {triplet_path}")

    results = dimension_estimation_experiment(
        triplets_path=triplet_path,
        n_items=1854,
        # rank_range=range(5, 60, 5),
        rank_range=[10],
        output_path=output_dir / f"cv_results_{triplet_path.stem}.csv",
        n_repeats=1,
        observed_fraction=0.8,
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    return results


if __name__ == "__main__":
    results = run_spose_dimension_analysis()
