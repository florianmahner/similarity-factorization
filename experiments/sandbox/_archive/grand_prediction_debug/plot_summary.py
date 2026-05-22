"""Summary plot: all dimensions per domain, no pooling."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.figure_theme import (
    create_figure,
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
    # Only load domain-level files (no underscore in name)
    domain_names = ["animals", "clothing", "professions", "sports", "weather",
                    "cities", "myth", "names", "states"]
    for name in domain_names:
        csv_path = GRAND_DIR / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["word"] = df["word"].str.lower()
            domains[name] = df

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


def plot_domain_all_dimensions(embeddings, word_to_idx, df, domain, output_dir):
    """Plot all dimensions for a single domain."""
    dims = [c for c in df.columns if c != "word"]
    n_dims = len(dims)
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    # Use create_figure with proper padding for title
    fig, axes = create_figure(
        "square",
        nrows=n_rows,
        ncols=n_cols,
        pad_top=0.5,  # Extra space for suptitle
        gridspec_kw={"hspace": 0.6, "wspace": 0.4},
    )

    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    n_samples = None
    for idx, dim in enumerate(dims):
        ax = axes[idx]
        valid = df[["word", dim]].dropna()
        result = get_predictions(
            embeddings, word_to_idx,
            valid["word"].tolist(),
            valid[dim].to_numpy()
        )

        if result is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                   transform=ax.transAxes, color=GRAY["medium"])
            ax.set_title(dim)
            despine(ax)
            continue

        n_samples = result["n"]
        color = CMAP[1] if result["correlation"] > 0 else CMAP[0]

        ax.scatter(result["true_values"], result["predictions"],
                   c=color, s=20, alpha=0.7, edgecolor="white", linewidth=0.3)

        # Regression line
        z = np.polyfit(result["true_values"], result["predictions"], 1)
        p = np.poly1d(z)
        xlim = (result["true_values"].min(), result["true_values"].max())
        x_line = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(x_line, p(x_line), color=GRAY["dark"], linewidth=1.5, linestyle="--")

        sig = "*" if result["pvalue"] < 0.05 else ""
        ax.set_title(f"{dim}\nρ = {result['correlation']:.2f}{sig}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        despine(ax)

    # Hide unused axes
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{domain.capitalize()} (n = {n_samples})", fontsize=11, fontweight="bold")
    save_figure(fig, output_dir / f"domain_{domain}.pdf")
    print(f"Created domain_{domain}.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, word_to_idx, domains = load_data()

    # Only use domains with >40% coverage
    good_domains = ["animals", "clothing", "professions", "sports", "weather"]

    for domain in good_domains:
        if domain in domains:
            plot_domain_all_dimensions(
                embeddings, word_to_idx, domains[domain], domain, OUTPUT_DIR
            )


if __name__ == "__main__":
    main()
