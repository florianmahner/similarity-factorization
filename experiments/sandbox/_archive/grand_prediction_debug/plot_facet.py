"""FacetGrid plots for GRAND prediction analysis.

1. Per-domain predictions for shared dimensions
2. Pooled predictions across domains
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.figure_theme import (
    apply_theme,
    save_figure,
    despine,
    CMAP,
    GRAY,
)

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
GRAND_DIR = PROJECT_ROOT / "data/grand_natbehav/extracted"
OUTPUT_DIR = Path(os.environ.get("SANDBOX_OUTPUT_DIR", Path(__file__).parent / "outputs" / "dev"))


def load_data():
    """Load embedding and GRAND ratings."""
    emb_path = EMBEDDING_DIR / "50"
    embeddings = np.load(emb_path / "word_embedding.npy")
    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)
    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}

    domains = {}
    for csv_path in GRAND_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["word"] = df["word"].str.lower()
        domains[csv_path.stem] = df

    return embeddings, word_to_idx, domains


def get_predictions(embeddings, word_to_idx, words, ratings):
    """Get cross-validated predictions."""
    valid_mask = [w in word_to_idx for w in words]
    valid_words = [w for w, v in zip(words, valid_mask) if v]
    valid_ratings = ratings[valid_mask]

    if len(valid_words) < 10:
        return None

    word_indices = np.array([word_to_idx[w] for w in valid_words])
    X = embeddings[word_indices]
    y = valid_ratings

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=cv)
    r, p = spearmanr(predictions, y)

    return {
        "predictions": predictions,
        "true_values": y,
        "words": valid_words,
        "correlation": r,
        "pvalue": p,
        "n": len(y),
    }


def plot_dimension_by_domain(embeddings, word_to_idx, domains, dimension, output_dir):
    """Plot predictions for one dimension across all domains that have it."""
    apply_theme()

    # Find domains that have this dimension
    domain_results = {}
    for domain, df in domains.items():
        if dimension not in df.columns:
            continue
        valid = df[["word", dimension]].dropna()
        result = get_predictions(
            embeddings, word_to_idx,
            valid["word"].tolist(),
            valid[dimension].to_numpy()
        )
        if result is not None:
            domain_results[domain] = result

    if not domain_results:
        print(f"No valid domains for {dimension}")
        return

    n_domains = len(domain_results)
    n_cols = min(4, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
    if n_domains == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (domain, result) in enumerate(sorted(domain_results.items())):
        ax = axes[idx]
        color = CMAP[1] if result["correlation"] > 0 else CMAP[0]

        ax.scatter(result["true_values"], result["predictions"],
                   c=color, s=25, alpha=0.7, edgecolor="white", linewidth=0.5)

        # Regression line
        z = np.polyfit(result["true_values"], result["predictions"], 1)
        p = np.poly1d(z)
        xlim = (result["true_values"].min(), result["true_values"].max())
        x_line = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(x_line, p(x_line), color=GRAY["dark"], linewidth=1.5, linestyle="--")

        # Significance marker
        sig = "*" if result["pvalue"] < 0.05 else ""
        ax.set_title(f"{domain.capitalize()}\nr = {result['correlation']:.2f}{sig}, n = {result['n']}")
        ax.set_xlabel(f"Actual {dimension}")
        ax.set_ylabel(f"Predicted {dimension}")
        despine(ax)

    # Hide unused axes
    for idx in range(n_domains, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Predicting '{dimension}' from SWOW embeddings", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / f"facet_{dimension}.pdf")
    print(f"Created facet_{dimension}.pdf")


def plot_pooled_dimension(embeddings, word_to_idx, domains, dimension, output_dir):
    """Pool all domains and predict dimension."""
    apply_theme()

    # Collect all words and ratings for this dimension
    all_words = []
    all_ratings = []
    all_domains = []

    for domain, df in domains.items():
        if dimension not in df.columns:
            continue
        valid = df[["word", dimension]].dropna()
        for _, row in valid.iterrows():
            word = row["word"]
            if word in word_to_idx:
                all_words.append(word)
                all_ratings.append(row[dimension])
                all_domains.append(domain)

    if len(all_words) < 20:
        print(f"Not enough samples for pooled {dimension}")
        return

    all_ratings = np.array(all_ratings)

    # Get predictions
    result = get_predictions(embeddings, word_to_idx, all_words, all_ratings)

    # Create plot
    fig, ax = plt.subplots(figsize=(4, 4))

    # Color by domain
    domain_colors = {d: CMAP[i % len(CMAP)] for i, d in enumerate(sorted(set(all_domains)))}
    colors = [domain_colors[d] for d in all_domains]

    ax.scatter(result["true_values"], result["predictions"],
               c=colors, s=30, alpha=0.7, edgecolor="white", linewidth=0.5)

    # Regression line
    z = np.polyfit(result["true_values"], result["predictions"], 1)
    p = np.poly1d(z)
    xlim = ax.get_xlim()
    x_line = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x_line, p(x_line), color=GRAY["dark"], linewidth=2, linestyle="--")

    # Legend
    for domain, color in domain_colors.items():
        ax.scatter([], [], c=color, label=domain.capitalize(), s=30)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    sig = "*" if result["pvalue"] < 0.05 else ""
    ax.set_title(f"Pooled '{dimension}' prediction\nr = {result['correlation']:.2f}{sig}, n = {result['n']}")
    ax.set_xlabel(f"Actual {dimension}")
    ax.set_ylabel(f"Predicted {dimension}")
    despine(ax)

    plt.tight_layout()
    save_figure(fig, output_dir / f"pooled_{dimension}.pdf")
    print(f"Created pooled_{dimension}.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, word_to_idx, domains = load_data()

    # Only use domains with >40% coverage
    good_domains = ["animals", "clothing", "professions", "sports", "weather"]
    domains = {k: v for k, v in domains.items() if k in good_domains}

    # Shared dimensions to analyze
    shared_dims = ["danger", "gender", "intelligence", "arousal", "wealth", "speed", "size", "wetness"]

    print("=== Per-domain predictions ===")
    for dim in shared_dims:
        plot_dimension_by_domain(embeddings, word_to_idx, domains, dim, OUTPUT_DIR)

    print("\n=== Pooled predictions ===")
    for dim in shared_dims:
        plot_pooled_dimension(embeddings, word_to_idx, domains, dim, OUTPUT_DIR)


if __name__ == "__main__":
    main()
