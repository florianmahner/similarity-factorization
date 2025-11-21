from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_data(data: pd.DataFrame, axis_col: str, value_col: str) -> pd.DataFrame:
    subset = data[[axis_col, value_col]].dropna().copy()
    subset = subset.sort_values(value_col, ascending=False)
    subset[value_col] = subset[value_col].astype(float)
    return subset


def plot_bars(
    data: pd.DataFrame,
    axis_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_data = _prepare_data(data, axis_col, value_col)
    if plot_data.empty:
        return

    sns.set_theme(style="ticks")
    plt.figure(figsize=(9, 4))
    colors = sns.color_palette("viridis", len(plot_data))
    ax = sns.barplot(
        data=plot_data,
        x=value_col,
        y=axis_col,
        hue=axis_col,
        palette=colors,
        dodge=False,
        legend=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, max(plot_data[value_col].max() * 1.1, 0.1))
    sns.despine(left=True, bottom=True)

    for idx, val in enumerate(plot_data[value_col]):
        ax.text(
            val + 0.01,
            idx,
            f"{val:.2f}",
            va="center",
            fontsize=9,
            color="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_grouped_bars(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot grouped bar chart comparing multiple groups."""
    if data.empty:
        return

    categories = sorted(data[category_col].dropna().unique())
    groups = sorted(data[group_col].dropna().unique())
    if not categories or not groups:
        return

    complete_index = pd.MultiIndex.from_product(
        [categories, groups], names=[category_col, group_col]
    )
    plot_data = (
        data[[category_col, group_col, value_col]]
        .dropna(subset=[category_col, group_col])
        .set_index([category_col, group_col])
        .reindex(complete_index, fill_value=0.0)
        .reset_index()
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("crest", len(groups))
    sns.barplot(
        data=plot_data,
        x=category_col,
        y=value_col,
        hue=group_col,
        order=categories,
        hue_order=groups,
        palette=palette,
        ax=ax,
    )

    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Semantic Axis", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.legend(title=group_col, loc="upper right", frameon=False)
    sns.despine()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_dimension_correlations(
    embedding: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot heatmap showing which dimensions correlate with which properties."""
    from scipy.stats import spearmanr

    property_cols = {
        "concreteness": "concreteness_rating",
        "size": "size_rating",
        "valence": "valence_rating",
        "heaviness": "heaviness_rating",
        "animacy": "animacy_rating",
    }

    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}
    rated_words = ratings[ratings["word"].isin(word_to_idx)]

    corr_matrix = []
    prop_names = []
    for prop_name, col in property_cols.items():
        if col not in rated_words.columns:
            continue

        valid = rated_words[["word", col]].dropna()
        if len(valid) < 10:
            continue

        dim_corrs = []
        for dim_idx in range(embedding.shape[1]):
            dim_values = [embedding[word_to_idx[w], dim_idx] for w in valid["word"]]
            r, _ = spearmanr(dim_values, valid[col])
            dim_corrs.append(r)

        corr_matrix.append(dim_corrs)
        prop_names.append(prop_name)

    if not corr_matrix:
        return

    corr_matrix = np.array(corr_matrix)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        xticklabels=range(embedding.shape[1]),
        yticklabels=prop_names,
        cbar_kws={"label": "Spearman r"},
    )
    ax.set_xlabel("Embedding Dimension", fontsize=11, fontweight="bold")
    ax.set_ylabel("Behavioral Property", fontsize=11, fontweight="bold")
    ax.set_title(
        "Which Dimensions Capture Behavioral Properties?",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_top_words_per_dimension(
    embedding: np.ndarray,
    vocabulary: list[str],
    ratings: pd.DataFrame,
    output_path: Path,
    n_words: int = 10,
) -> None:
    """Show top words for best dimension per property."""
    from scipy.stats import spearmanr

    property_cols = {
        "concreteness": "concreteness_rating",
        "size": "size_rating",
        "valence": "valence_rating",
        "heaviness": "heaviness_rating",
        "animacy": "animacy_rating",
    }

    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}
    rated_words = ratings[ratings["word"].isin(word_to_idx)]

    fig, axes = plt.subplots(len(property_cols), 1, figsize=(10, 12))

    for ax_idx, (prop_name, col) in enumerate(property_cols.items()):
        if col not in rated_words.columns:
            continue

        valid = rated_words[["word", col]].dropna()
        if len(valid) < 10:
            continue

        best_dim = None
        best_corr = 0
        for dim_idx in range(embedding.shape[1]):
            dim_values = [embedding[word_to_idx[w], dim_idx] for w in valid["word"]]
            r, _ = spearmanr(dim_values, valid[col])
            if abs(r) > abs(best_corr):
                best_corr = r
                best_dim = dim_idx

        if best_dim is None:
            continue

        vocab_lower = [w.lower() for w in vocabulary]
        word_scores = [
            (w, embedding[word_to_idx[w], best_dim])
            for w in vocab_lower
            if w in word_to_idx
        ]
        word_scores.sort(key=lambda x: x[1])

        top_words = [w for w, _ in word_scores[-n_words:]]
        bottom_words = [w for w, _ in word_scores[:n_words]]

        ax = axes[ax_idx]
        y_pos = np.arange(n_words * 2)
        words = bottom_words + top_words
        colors = ["#d62728"] * n_words + ["#2ca02c"] * n_words

        ax.barh(y_pos, [1] * len(words), color=colors, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.set_title(
            f"{prop_name.title()} (Dim {best_dim}, r={best_corr:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.suptitle(
        "Top Words for Best-Predicting Dimensions",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_prediction_scatter(
    projections: pd.DataFrame,
    ratings: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plots: predicted vs true ratings for each property."""
    from scipy.stats import spearmanr

    property_cols = {
        "concreteness": "concreteness_rating",
        "size": "size_rating",
        "valence": "valence_rating",
        "heaviness": "heaviness_rating",
        "animacy": "animacy_rating",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (prop_name, col) in enumerate(property_cols.items()):
        if prop_name not in projections.columns or col not in ratings.columns:
            axes[ax_idx].axis("off")
            continue

        merged = projections.merge(ratings[["word", col]], on="word", how="inner")
        valid = merged[[prop_name, col]].dropna()

        if len(valid) < 10:
            axes[ax_idx].axis("off")
            continue

        ax = axes[ax_idx]
        ax.scatter(
            valid[col],
            valid[prop_name],
            alpha=0.3,
            s=20,
            color="#1f77b4",
            edgecolors="none",
        )

        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        X_vals = valid[col].values.reshape(-1, 1)
        y_vals = valid[prop_name].values
        lr.fit(X_vals, y_vals)

        x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
        y_line = lr.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, "r--", alpha=0.5, lw=2, label="Linear fit")

        r, p = spearmanr(valid[prop_name], valid[col])
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np = {p:.2e}\nn = {len(valid)}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("True Rating", fontsize=10)
        ax.set_ylabel("Predicted (Projection)", fontsize=10)
        ax.set_title(prop_name.title(), fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")

    plt.suptitle(
        "Prediction Quality: Unsupervised Axis Projections",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pole_separation_2d(
    embedding: np.ndarray,
    vocabulary: list[str],
    pole_words: dict[str, tuple[list[str], list[str]]],
    output_path: Path,
) -> None:
    """Visualize pole words in 2D PCA space."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    embedding_2d = pca.fit_transform(embedding)

    word_to_idx = {word.lower(): i for i, word in enumerate(vocabulary)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, (prop_name, (high_words, low_words)) in enumerate(pole_words.items()):
        if ax_idx >= len(axes):
            break

        ax = axes[ax_idx]

        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c="lightgray",
            alpha=0.2,
            s=5,
            label="All words",
        )

        high_idx = [word_to_idx[w] for w in high_words if w in word_to_idx]
        low_idx = [word_to_idx[w] for w in low_words if w in word_to_idx]

        if high_idx:
            ax.scatter(
                embedding_2d[high_idx, 0],
                embedding_2d[high_idx, 1],
                c="green",
                alpha=0.7,
                s=30,
                label=f"High {prop_name}",
                edgecolors="black",
                linewidths=0.5,
            )

        if low_idx:
            ax.scatter(
                embedding_2d[low_idx, 0],
                embedding_2d[low_idx, 1],
                c="red",
                alpha=0.7,
                s=30,
                label=f"Low {prop_name}",
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=9)
        ax.set_title(prop_name.title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    for ax_idx in range(len(pole_words), len(axes)):
        axes[ax_idx].axis("off")

    plt.suptitle(
        "Pole Word Separation in 2D Embedding Space",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_projection_scatter_grid(
    projections: pd.DataFrame,
    ratings: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot scatter grid comparing ground truth ratings (x) vs projected scores (y).

    Creates a multi-panel plot with one scatter plot per semantic dimension,
    showing the correlation between actual behavioral ratings (x) and
    predicted projection scores (y).

    Args:
        projections: DataFrame with 'word' column and projection scores per dimension
        ratings: DataFrame with 'word' column and ground truth ratings per dimension
        output_path: Path to save the figure
    """
    merged = projections.merge(ratings, on="word", how="inner")
    dimensions = [col for col in projections.columns if col != "word"]

    n_dims = len(dimensions)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for idx, dimension in enumerate(dimensions):
        ax = axes[idx]

        valid = merged[[dimension + "_x", dimension + "_y"]].dropna()
        if len(valid) < 10:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor {dimension}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(dimension.title(), fontsize=11, fontweight="bold")
            continue

        # Swapping axes: x is now ground truth, y is projected
        x = valid[dimension + "_y"].values  # ground truth
        y = valid[dimension + "_x"].values  # projected

        from scipy.stats import spearmanr

        r, p = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.4, s=20, color="steelblue", edgecolors="none")

        # Linear fit now: y vs x
        z = np.polyfit(x, y, 1)
        poly = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, poly(x_line), "r--", linewidth=2, alpha=0.7, label="Linear fit")

        ax.set_xlabel("Ground Truth Rating", fontsize=10)
        ax.set_ylabel("Predicted Score", fontsize=10)
        ax.set_title(dimension.title(), fontsize=11, fontweight="bold")

        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np = {p:.2e}\nn = {len(valid)}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        ax.grid(alpha=0.3)

    for idx in range(n_dims, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Ridge Encoding: Ground Truth vs Predicted",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_bars(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot bar chart comparing correlations across dimensions and methods.

    Args:
        results: DataFrame with columns: dimension, method, correlation
        output_path: Path to save the figure
    """
    if "method" in results.columns and results["method"].nunique() > 1:
        plot_grouped_bars(
            data=results,
            category_col="dimension",
            value_col="correlation",
            group_col="method",
            title="Semantic Validation: Method Comparison",
            ylabel="Spearman Correlation",
            output_path=output_path,
        )
    else:
        plot_bars(
            data=results,
            axis_col="dimension",
            value_col="correlation",
            title="Semantic Projection Validation",
            ylabel="Spearman Correlation",
            output_path=output_path,
        )


def plot_reconstruction_quality(
    similarity: np.ndarray,
    embedding: np.ndarray,
    output_path: Path,
    n_samples: int = 5000,
) -> None:
    """Plot reconstruction quality for observed entries.

    Args:
        similarity: Original similarity matrix (with NaN for missing)
        embedding: Word embedding matrix W (reconstruction is W @ W.T)
        output_path: Path to save the figure
        n_samples: Number of samples to plot in scatter
    """
    observed_mask = ~np.isnan(similarity)
    observed_mask[similarity == 0] = False

    similarity_clean = np.nan_to_num(similarity, nan=0)
    reconstruction = embedding @ embedding.T

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    observed_actual = similarity_clean[observed_mask]
    observed_pred = reconstruction[observed_mask]

    if len(observed_actual) > n_samples:
        idx = np.random.choice(len(observed_actual), n_samples, replace=False)
        observed_actual = observed_actual[idx]
        observed_pred = observed_pred[idx]

    ax.scatter(
        observed_actual,
        observed_pred,
        alpha=0.3,
        s=10,
        c="#3498db",
        edgecolors="none",
    )

    max_val = max(observed_actual.max(), observed_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect prediction")

    corr = np.corrcoef(observed_actual, observed_pred)[0, 1]
    mse = np.mean((observed_actual - observed_pred) ** 2)

    ax.set_xlabel("Observed association strength", fontsize=12)
    ax.set_ylabel("SRF reconstruction", fontsize=12)
    ax.set_title(
        f"Reconstruction on observed entries\nr = {corr:.3f}, MSE = {mse:.6f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    errors = observed_actual - observed_pred
    ax.hist(errors, bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(
        errors.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {errors.mean():.4f}",
    )

    ax.set_xlabel("Reconstruction error", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of reconstruction errors", fontsize=13, fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "SRF reconstruction quality with missing data",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_top_words_facet_grid(
    top_words_df: pd.DataFrame,
    output_path: Path,
    n_words: int = 10,
) -> None:
    """Plot top words per dimension using FacetGrid.

    Args:
        top_words_df: DataFrame with columns: dimension, word, loading
        output_path: Path to save the figure
        n_words: Number of words to show per dimension
    """
    df = top_words_df.copy()
    df = df.groupby("dimension").head(n_words)

    n_dims = df["dimension"].nunique()
    n_cols = 5

    g = sns.FacetGrid(
        df,
        col="dimension",
        col_wrap=n_cols,
        sharex=False,
        sharey=False,
        height=5,
        aspect=0.8,
    )

    def barplot(data, **kwargs):
        data_sorted = data.sort_values("loading", ascending=True)
        y = data_sorted["word"]
        x = data_sorted["loading"]
        ax = plt.gca()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(y)))
        ax.barh(y, x, color=colors)
        ax.set_xlabel("Loading")
        ax.set_title(f"Dimension {data['dimension'].iloc[0]}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    g.map_dataframe(barplot)
    g.set_titles(col_template="Dimension {col_name}")
    g.set_axis_labels("Loading", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_word_clouds_grid(
    top_words_df: pd.DataFrame,
    output_path: Path,
    n_words: int = 100,
) -> None:
    """Plot word clouds for all dimensions.

    Args:
        top_words_df: DataFrame with columns: dimension, word, loading
        output_path: Path to save the figure
        n_words: Number of top words to include in each cloud
    """
    from wordcloud import WordCloud

    df = top_words_df.copy()
    rank = df["dimension"].nunique()

    colormaps = ["Blues", "Oranges", "Greens", "Reds", "Purples", "YlOrBr"]

    n_cols = 2 if rank < 4 else 3 if rank < 7 else 4 if rank < 13 else 5
    n_rows = int(np.ceil(rank / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 5))

    axes = axes.flatten() if rank > 1 else [axes]

    for i, dim in enumerate(range(rank)):
        dim_words = df[df["dimension"] == dim].head(n_words)
        word_freq = {
            row["word"]: row["loading"]
            for _, row in dim_words.iterrows()
            if row["loading"] > 0
        }

        if word_freq:
            top_5 = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            top_words_str = ", ".join([w for w, _ in top_5])

            wc = WordCloud(
                width=600,
                height=400,
                background_color="white",
                colormap=colormaps[dim % len(colormaps)],
                relative_scaling=0.4,
                min_font_size=10,
                max_font_size=80,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_freq)

            ax = axes[i]
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                f"Dimension {dim}\n{top_words_str}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
        else:
            axes[i].axis("off")

    for unused_ax in axes[rank:]:
        unused_ax.axis("off")

    plt.suptitle(
        "Semantic dimensions discovered by SRF",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_analogy(
    embedding: np.ndarray,
    vocabulary: list[str],
    word_to_idx: dict[str, int],
    top_words_df: pd.DataFrame,
    a: str,
    b: str,
    c: str,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Plot analogy analysis: a - b + c = ?

    Args:
        embedding: Word embedding matrix
        vocabulary: List of words
        word_to_idx: Dict mapping words to indices
        top_words_df: DataFrame with dimension top words for wordclouds
        a: First word (positive)
        b: Second word (negative)
        c: Third word (positive)
        output_path: Path to save the figure
        top_k: Number of top predictions to show
    """
    from wordcloud import WordCloud

    for word in [a, b, c]:
        if word not in word_to_idx:
            return

    target_vec = (
        embedding[word_to_idx[a]]
        - embedding[word_to_idx[b]]
        + embedding[word_to_idx[c]]
    )

    w_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-10)
    similarities = w_norm @ target_norm

    top_indices = np.argsort(similarities)[::-1]
    results = []
    input_words = {a, b, c}

    for idx in top_indices:
        word = vocabulary[idx]
        if word not in input_words:
            results.append((word, similarities[idx]))
            if len(results) == top_k:
                break

    if not results:
        return

    top_word = results[0][0]
    dim_pos = int(np.argmax(target_vec))
    dim_neg = int(np.argmin(target_vec))

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, height_ratios=[1, 1, 0.8])

    ax1 = fig.add_subplot(gs[0, :])
    words = [a, b, c, top_word]
    vectors = np.array([embedding[word_to_idx[word]] for word in words])

    data = np.vstack([vectors])
    data_min = data.min(axis=1, keepdims=True)
    data_max = data.max(axis=1, keepdims=True)
    data_norm = (data - data_min) / (data_max - data_min + 1e-10)

    labels = words
    sns.heatmap(data_norm, yticklabels=labels, cmap="viridis", cbar=False, ax=ax1)
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("")
    ax1.set_title(f"{a} - {b} + {c} = {top_word}", fontsize=14, weight="bold")
    ax1.axvline(dim_pos, color="green", linestyle="--", linewidth=2, alpha=0.7)
    ax1.axvline(dim_neg, color="red", linestyle="--", linewidth=2, alpha=0.7)

    for i, (dim, ax_pos) in enumerate([(dim_pos, gs[1, :2]), (dim_neg, gs[1, 2])]):
        ax = fig.add_subplot(ax_pos)
        dim_words = top_words_df[top_words_df["dimension"] == int(dim)].head(80)
        word_freq = {
            row["word"]: abs(row["loading"]) for _, row in dim_words.iterrows()
        }
        if word_freq:
            wc = WordCloud(
                width=450,
                height=300,
                background_color="white",
                colormap="viridis",
            ).generate_from_frequencies(word_freq)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            title = "Most positive" if i == 0 else "Most negative"
            ax.set_title(f"{title} dimension {dim} (Δ={target_vec[dim]:.3f})")

    ax4 = fig.add_subplot(gs[2, :2])
    words_list, sims_list = zip(*results)
    y_pos = np.arange(len(words_list))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
    ax4.barh(
        y_pos,
        sims_list,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(words_list)
    ax4.set_xlabel("Cosine similarity")
    ax4.set_title(f"Top {len(results)} predictions", weight="bold")
    ax4.invert_yaxis()
    ax4.grid(axis="x", alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 2])
    top_dims = np.argsort(np.abs(target_vec))[::-1][:15]
    top_vals = target_vec[top_dims]
    colors = ["red" if v < 0 else "blue" for v in top_vals]
    ax5.barh(range(len(top_dims)), top_vals, color=colors, alpha=0.7)
    ax5.set_yticks(range(len(top_dims)))
    ax5.set_yticklabels([f"{d}" for d in top_dims], fontsize=8)
    ax5.set_xlabel("Target value")
    ax5.set_ylabel("Dimension")
    ax5.set_title("Top 15 dimensions", weight="bold")
    ax5.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax5.invert_yaxis()
    ax5.grid(axis="x", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
