from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from .common import compute_similarity_matrix_from_triplets
from pysrf import cross_val_score

from .resources import load_resources


def run_spose_dimensionality_analysis(
    train_triplets,
    rank_range=range(5, 90, 5),
    n_repeats: int = 5,
    n_jobs: int = -1,
):
    similarity = compute_similarity_matrix_from_triplets(1854, train_triplets)
    param_grid = {
        "rank": list(rank_range),
        "max_outer": [10],
        "max_inner": [50],
        "tol": [0.0],
        "rho": [1.0],
    }

    scorer = cross_val_score(
        similarity,
        param_grid=param_grid,
        n_repeats=n_repeats,
        estimate_sampling_fraction=True,
        random_state=0,
        verbose=0,
        n_jobs=n_jobs,
        fit_final_estimator=False,
    )
    return pd.DataFrame(scorer.cv_results_)


def run(cfg: DictConfig) -> None:
    resources = load_resources(cfg)
    ranks = range(cfg.rank_min, cfg.rank_max, cfg.rank_step)

    df = run_spose_dimensionality_analysis(
        resources.train_triplets,
        rank_range=ranks,
        n_repeats=cfg.n_repeats,
        n_jobs=cfg.n_jobs,
    )

    df.to_csv(Path.cwd() / "results.csv", index=False)
