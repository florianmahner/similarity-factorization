"""Predict behavioral properties from SWOW consensus embedding.

Reads the consensus embedding from experiments/datasets/consensus/outputs/swow/
and evaluates how well embedding dimensions predict Glasgow Norms ratings
(arousal, valence, concreteness, etc.) using three methods:
  - Gland projection (unsupervised, semantic axis)
  - Ridge regression (supervised, nested CV)
  - Lasso regression (sparse, nested CV)

Also generates word clouds for the top dimensions by variance.

Usage:
    ./scripts/submit experiments/analyses/swow/run.py --bg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from wordcloud import WordCloud

from datasets import load_swow_ppmi
from .plot import plot_correlation_bars, plot_projection_scatter_grid
from .behavioral_prediction import (
    evaluate_gland_projection_method,
    evaluate_lasso_encoding,
    evaluate_ridge_encoding,
)
from .norms import load_behavioral_ratings

CONSENSUS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "consensus" / "outputs"


def _load_consensus_embedding(dataset_name: str) -> np.ndarray:
    path = CONSENSUS_DIR / dataset_name / "embedding.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Consensus embedding not found: {path}\n"
            f"Run: ./scripts/submit experiments/datasets/consensus/run.py dataset={dataset_name}"
        )
    return np.load(path)


def _make_wordclouds(embedding: np.ndarray, vocabulary: list[str],
                      out_dir: Path, n_dims: int = 9) -> None:
    """Render word clouds for the top-`n_dims` dimensions by variance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    var_per_dim = np.var(embedding, axis=0)
    dims = np.argsort(var_per_dim)[::-1][:n_dims]

    for dim_idx in dims:
        order = np.argsort(embedding[:, dim_idx])[::-1][:40]
        freqs = {vocabulary[i]: max(float(embedding[i, dim_idx]), 1e-6)
                 for i in order}
        wc = WordCloud(
            width=800, height=800, background_color="white",
            relative_scaling=0.4, min_font_size=8, max_font_size=200,
            prefer_horizontal=0.7, margin=5,
        )
        wc.generate_from_frequencies(freqs)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"Dim {dim_idx}", fontsize=10, pad=4)
        fig.savefig(out_dir / f"dim_{dim_idx:03d}.png", dpi=150,
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)


def run(cfg: DictConfig) -> None:
    out_dir = Path.cwd()

    embeddings = _load_consensus_embedding("swow")

    _, vocabulary, _ = load_swow_ppmi(
        Path(cfg.data_dir),
        use_all_responses=cfg.use_all_responses,
        top_n_words=cfg.top_n_words,
        min_word_length=cfg.min_word_length,
        symmetrization=cfg.symmetrization,
        bidirectional_only=cfg.bidirectional_only,
    )

    ratings = load_behavioral_ratings(Path(cfg.ratings_data_dir))

    projection_results, projections = evaluate_gland_projection_method(
        embeddings, vocabulary, ratings, pole_size=cfg.get("pole_size", 20),
    )

    supervised_results, predictions_ridge = evaluate_ridge_encoding(
        embeddings, vocabulary, ratings,
        n_outer_folds=cfg.get("n_outer_folds", 5),
        n_inner_folds=cfg.get("n_inner_folds", 3),
    )

    lasso_results, predictions_lasso = evaluate_lasso_encoding(
        embeddings, vocabulary, ratings,
        n_outer_folds=cfg.get("n_outer_folds", 5),
    )

    lasso_results.to_csv(out_dir / "metrics_lasso.csv", index=False)

    all_results = pd.concat(
        [projection_results, supervised_results, lasso_results], ignore_index=True,
    )
    all_results.to_csv(out_dir / "results.csv", index=False)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_projection_scatter_grid(predictions_ridge, ratings, plots_dir / "projection_scatter_grid.pdf")
    plot_correlation_bars(all_results, plots_dir / "method_comparison.pdf")

    _make_wordclouds(embeddings, vocabulary, out_dir / "wordclouds")


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=".", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run(cfg)

    _main()


if __name__ == "__main__":
    main()
