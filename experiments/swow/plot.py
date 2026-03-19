"""
Create all word_association experiment plots.

Usage:
    poetry run python experiments/word_association/plot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT, CYCLE, setup_style
from src.utils.figure_theme import (
    create_figure,
    despine,
    save_figure,
)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).parents[2]
WORD_ASSOC_DIR = PROJECT_ROOT / "outputs/experiments/word_association"
DATA_DIR = WORD_ASSOC_DIR / "behavioral_prediction"


def _plot_method_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot comparing correlation across methods and dimensions."""
    fig, ax = create_figure("single")

    dimensions = ["concreteness", "valence", "animacy", "size", "heaviness"]
    methods = ["Ridge_CV", "Lasso_CV", "Gland_Projection"]
    method_labels = ["Ridge", "Lasso", "Projection"]
    colors = [TEAL, CYAN, ROSE]

    x = np.arange(len(dimensions))
    width = 0.25

    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        method_df = df[df["method"] == method]
        correlations = []
        for dim in dimensions:
            row = method_df[method_df["dimension"] == dim]
            if len(row) > 0:
                correlations.append(row["correlation"].values[0])
            else:
                correlations.append(0)

        offset = (i - 1) * width
        ax.bar(
            x + offset,
            correlations,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(y=0, color=GRAY_LIGHT, linestyle="-", linewidth=0.5)
    ax.set_ylabel("Spearman correlation")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], fontsize=8)
    ax.set_ylim(0, 0.85)

    despine(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    save_figure(fig, output_path)


def _plot_dimension_bars(df: pd.DataFrame, method: str, output_path: Path) -> None:
    """Horizontal bar plot of correlations for one method."""
    fig, ax = create_figure("single", pad_left=0.7)

    method_df = df[df["method"] == method].copy()
    method_df = method_df.sort_values("correlation", ascending=True)

    y = np.arange(len(method_df))
    colors = [CYCLE[i % len(CYCLE)] for i in range(len(method_df))]

    ax.barh(
        y,
        method_df["correlation"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([d.capitalize() for d in method_df["dimension"]], fontsize=8)
    ax.set_xlabel("Spearman correlation")
    ax.set_xlim(0, 0.85)

    despine(ax)
    save_figure(fig, output_path)


def _load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load behavioral ratings."""
    size_df = pd.read_csv(data_dir / "things/size_ratings.csv")
    size_df = size_df.rename(columns={"Word": "word", "Size_mean": "size"})
    size_df["word"] = size_df["word"].str.lower()
    size_df = size_df[["word", "size"]]

    conc_df = pd.read_excel(data_dir / "concreteness_ratings_brysbaert.xlsx")
    conc_df = conc_df.rename(columns={"Word": "word", "Conc.M": "concreteness"})
    conc_df["word"] = conc_df["word"].str.lower()
    conc_df = conc_df[["word", "concreteness"]]

    props_df = pd.read_csv(data_dir / "things/things_property_ratings.csv")
    props_df = props_df.rename(columns={
        "Word": "word",
        "pleasant_mean": "valence",
        "heavy_mean": "heaviness",
        "lives_mean": "animacy",
    })
    props_df["word"] = props_df["word"].str.lower()
    props_df = props_df[["word", "valence", "heaviness", "animacy"]]

    merged = size_df.merge(conc_df, on="word", how="outer")
    merged = merged.merge(props_df, on="word", how="outer")
    return merged


def _get_ridge_predictions(
    embeddings: np.ndarray,
    word_to_idx: dict[str, int],
    ratings: pd.DataFrame,
    dimension: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get cross-validated Ridge predictions for a dimension."""
    valid = ratings[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]

    word_indices = np.array([word_to_idx[w] for w in valid["word"]])
    X = embeddings[word_indices]
    y = valid[dimension].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X_scaled, y, cv=cv)

    return y, predictions


def _plot_prediction_scatter(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    dimension: str,
    output_path: Path,
) -> None:
    """Scatter plot of ground truth vs predicted with regression line."""
    import seaborn as sns

    # Normalize to [0, 1]
    true_norm = (true_values - true_values.min()) / (true_values.max() - true_values.min())
    pred_norm = (pred_values - pred_values.min()) / (pred_values.max() - pred_values.min())

    fig, ax = create_figure("single", pad_top=0.3)

    # Muted steel blue for points, darker for line
    point_color = "#5a7d9a"
    line_color = "#2c3e50"

    sns.regplot(
        x=true_norm,
        y=pred_norm,
        ax=ax,
        scatter_kws={"alpha": 0.3, "s": 8, "color": point_color, "edgecolor": "none"},
        line_kws={"color": line_color, "linewidth": 1.2},
        ci=None,
    )

    r, _ = spearmanr(true_values, pred_values)
    ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom")

    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.set_title(dimension.capitalize(), fontsize=10)

    despine(ax)
    save_figure(fig, output_path)


def _get_dimension_words(top_words: dict, dim: int, n_words: int = 20) -> dict[str, float]:
    """Extract word frequencies for a single dimension."""
    word_freqs = {}
    for i, d in enumerate(top_words["dimension"]):
        if d == dim and top_words["rank"][i] < n_words:
            word = top_words["word"][i]
            loading = top_words["loading"][i]
            word_freqs[word] = max(loading, 0.01)
    return word_freqs


DIMENSION_COLORMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples", "YlOrBr"]


def _create_wordcloud(
    word_freqs: dict[str, float],
    dim: int = 0,
) -> WordCloud:
    """Create a WordCloud object from word frequencies."""
    wc = WordCloud(
        width=600,
        height=400,
        background_color="white",
        colormap=DIMENSION_COLORMAPS[dim % len(DIMENSION_COLORMAPS)],
        relative_scaling=0.4,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(word_freqs)
    return wc


def _plot_single_wordcloud(
    word_freqs: dict[str, float],
    dim: int,
    output_path: Path,
) -> None:
    """Create and save a high-DPI PDF word cloud for a single dimension."""
    if not word_freqs:
        return

    wc = _create_wordcloud(word_freqs, dim=dim)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def _plot_wordcloud_grid(
    top_words: dict,
    n_dims: int,
    output_path: Path,
    n_words: int = 100,
) -> None:
    """Create a grid of all word clouds as high-DPI PDF."""
    n_cols = 5
    n_rows = int(np.ceil(n_dims / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 8, n_rows * 5),
        dpi=300,
    )
    axes = axes.flatten()

    for dim in range(n_dims):
        word_freqs = _get_dimension_words(top_words, dim, n_words)
        ax = axes[dim]

        if word_freqs:
            top_5 = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
            top_words_str = ", ".join([w for w, _ in top_5])

            wc = _create_wordcloud(word_freqs, dim=dim)
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"Dimension {dim + 1}\n{top_words_str}", fontsize=11, fontweight="bold", pad=10)
        else:
            ax.set_title(f"Dimension {dim + 1}", fontsize=11, fontweight="bold", pad=10)

        ax.axis("off")

    for i in range(n_dims, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Semantic dimensions discovered by SRF", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Created: {output_path.name}")


def _compute_top_words(embedding: np.ndarray, vocabulary: list[str], n_words: int = 50) -> pd.DataFrame:
    """Compute top words per dimension from embedding matrix."""
    n_items, n_dims = embedding.shape
    records = []

    for dim in range(n_dims):
        loadings = embedding[:, dim]
        top_indices = np.argsort(loadings)[::-1][:n_words]

        for rank, idx in enumerate(top_indices):
            records.append({
                "dimension": dim,
                "rank": rank,
                "word": vocabulary[idx],
                "loading": loadings[idx],
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Create word_association plots")
    parser.add_argument("--rank", type=int, default=50, help="Embedding rank (50 or 100)")
    parser.add_argument("--wordclouds-only", action="store_true", help="Only generate word clouds")
    parser.add_argument("--n-words", type=int, default=50, help="Number of words per dimension")
    args = parser.parse_args()

    embedding_dir = WORD_ASSOC_DIR / "generate_embedding" / str(args.rank)
    output_dir = WORD_ASSOC_DIR / "plots" / f"dim{args.rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding and vocabulary
    embedding = np.load(embedding_dir / "word_embedding.npy")
    with open(embedding_dir / "metadata.json") as f:
        metadata = json.load(f)
    vocabulary = metadata["vocabulary"]

    # Compute top words from embedding (more words than stored in metadata)
    top_words_df = _compute_top_words(embedding, vocabulary, n_words=args.n_words)
    print(f"Computed top {args.n_words} words per dimension from embedding")

    # Generate grid AND individual PDFs using the SAME word cloud objects
    from experiments.word_association.plotting import plot_word_clouds_grid
    plot_word_clouds_grid(
        top_words_df,
        output_dir / "wordcloud_grid.pdf",
        n_words=args.n_words,
        save_individual=True,
    )
    print("Created: wordcloud_grid.pdf")

    if args.wordclouds_only:
        print(f"\nSaved to {output_dir}")
        return

    # Load results
    results_csv = DATA_DIR / "results.csv"
    if not results_csv.exists():
        print(f"Results not found: {results_csv}")
        return

    results = pd.read_csv(results_csv)

    # Method comparison bar chart
    _plot_method_comparison(results, output_dir / "method_comparison.pdf")
    print("Created: method_comparison.pdf")

    # Ridge-only horizontal bars
    _plot_dimension_bars(results, "Ridge_CV", output_dir / "ridge_correlations.pdf")
    print("Created: ridge_correlations.pdf")

    # Load embeddings and ratings for scatter plots
    embeddings = np.load(embedding_dir / "word_embedding.npy")
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    ratings = _load_ratings(PROJECT_ROOT / "data")

    # Generate scatter plot for each dimension
    dimensions = ["size", "concreteness", "valence", "heaviness", "animacy"]
    for dimension in dimensions:
        true_vals, pred_vals = _get_ridge_predictions(
            embeddings, word_to_idx, ratings, dimension
        )
        _plot_prediction_scatter(
            true_vals, pred_vals, dimension,
            output_dir / f"{dimension}_prediction.pdf"
        )
        print(f"Created: {dimension}_prediction.pdf")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
