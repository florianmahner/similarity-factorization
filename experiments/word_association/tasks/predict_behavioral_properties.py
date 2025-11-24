from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..lib.plotting import (
    plot_correlation_bars,
    plot_projection_scatter_grid,
)
from ..lib.semantic_helpers import (
    evaluate_gland_projection_method,
    evaluate_ridge_encoding,
    evaluate_lasso_encoding,
)
from ..lib.validation import load_behavioral_ratings


def run(cfg: DictConfig) -> None:
    if not cfg.embedding_dir:
        raise ValueError("predict_behavioral_properties.embedding_dir is required")

    ratings_dir = Path(cfg.ratings_data_dir)
    out_dir = Path.cwd() / "behavioral_prediction"
    out_dir.mkdir(exist_ok=True)

    embeddings = np.load(Path(cfg.embedding_dir) / "word_embedding.npy")
    with open(Path(cfg.embedding_dir) / "metadata.json") as f:
        metadata = json.load(f)

    vocabulary = metadata["vocabulary"]
    ratings = load_behavioral_ratings(ratings_dir)

    projection_results, projections = evaluate_gland_projection_method(
        embeddings, vocabulary, ratings, pole_size=cfg.get("pole_size", 20)
    )

    supervised_results, predictions_ridge = evaluate_ridge_encoding(
        embeddings,
        vocabulary,
        ratings,
        n_outer_folds=cfg.get("n_outer_folds", 5),
        n_inner_folds=cfg.get("n_inner_folds", 3),
    )

    lasso_results, predictions_lasso = evaluate_lasso_encoding(
        embeddings,
        vocabulary,
        ratings,
        n_outer_folds=cfg.get("n_outer_folds", 5),
    )

    lasso_results.to_csv(out_dir / "metrics_lasso.csv", index=False)

    all_results = pd.concat(
        [projection_results, supervised_results, lasso_results], ignore_index=True
    )
    all_results.to_csv(out_dir / "results.csv", index=False)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_projection_scatter_grid(
        predictions_ridge, ratings, plots_dir / "projection_scatter_grid.png"
    )
    plot_correlation_bars(all_results, plots_dir / "method_comparison.png")
