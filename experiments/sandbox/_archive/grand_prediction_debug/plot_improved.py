"""Improved plots for GRAND prediction analysis."""

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
    for csv_path in GRAND_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["word"] = df["word"].str.lower()
        domains[csv_path.stem] = df

    return embeddings, word_to_idx, domains


def get_predictions(embeddings, word_to_idx, df, dimension):
    """Get cross-validated predictions for a dimension."""
    valid = df[["word", dimension]].dropna()
    valid = valid[valid["word"].isin(word_to_idx)]

    if len(valid) < 20:
        return None

    word_indices = np.array([word_to_idx[w] for w in valid["word"]])
    X = embeddings[word_indices]
    y = valid[dimension].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=cv)

    return {
        "predictions": predictions,
        "true_values": y,
        "words": valid["word"].tolist(),
        "correlation": spearmanr(predictions, y)[0],
    }


def plot_scatter_predictions(embeddings, word_to_idx, domains, output_dir):
    """Scatter plots showing predicted vs actual for best/worst cases."""
    apply_theme()

    cases = [
        ("animals", "wetness", "Best: Animals/Wetness"),
        ("animals", "danger", "Animals/Danger"),
        ("sports", "danger", "Negative: Sports/Danger"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

    for ax, (domain, dim, title) in zip(axes, cases):
        result = get_predictions(embeddings, word_to_idx, domains[domain], dim)

        color = CMAP[1] if result["correlation"] > 0 else CMAP[0]
        ax.scatter(result["true_values"], result["predictions"],
                   c=color, s=30, alpha=0.7, edgecolor="white", linewidth=0.5)

        # Add regression line
        z = np.polyfit(result["true_values"], result["predictions"], 1)
        p = np.poly1d(z)
        xlim = ax.get_xlim()
        x_line = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(x_line, p(x_line), color=GRAY["dark"], linewidth=1.5, linestyle="--")

        # Label extreme words
        y_pred = result["predictions"]
        y_true = result["true_values"]
        words = result["words"]

        # Find words with largest residuals
        residuals = np.abs(y_pred - y_true)
        top_idx = np.argsort(residuals)[-3:]

        for idx in top_idx:
            ax.annotate(words[idx], (y_true[idx], y_pred[idx]),
                       fontsize=7, alpha=0.8,
                       xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel(f"Actual {dim}")
        ax.set_ylabel(f"Predicted {dim}")
        ax.set_title(f"{title}\nr = {result['correlation']:.2f}", fontsize=10)
        despine(ax)

    plt.tight_layout()
    save_figure(fig, output_dir / "scatter_predictions.pdf")


def plot_correlation_distribution(results_path, output_dir):
    """Histogram showing distribution of prediction correlations."""
    apply_theme()

    df = pd.read_csv(results_path)

    fig, ax = create_figure("single")

    # Color by significance
    sig = df[df["pvalue"] < 0.05]
    nonsig = df[df["pvalue"] >= 0.05]

    bins = np.linspace(-0.6, 0.7, 20)

    ax.hist(nonsig["correlation"], bins=bins, color=GRAY["light"],
            edgecolor="white", label="Not significant", alpha=0.8)
    ax.hist(sig["correlation"], bins=bins, color=CMAP[1],
            edgecolor="white", label="p < 0.05", alpha=0.8)

    ax.axvline(0, color=GRAY["dark"], linewidth=1, linestyle="--")
    ax.axvline(df["correlation"].mean(), color=CMAP[0], linewidth=2,
               label=f"Mean = {df['correlation'].mean():.2f}")

    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of prediction correlations")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "correlation_distribution.pdf")


def plot_all_domains(results_path, output_dir):
    """Bar plot showing mean correlation per domain (all 9 domains)."""
    apply_theme()

    df = pd.read_csv(results_path)

    # Compute domain-level statistics
    domain_stats = df.groupby("domain").agg({
        "correlation": ["mean", "std", "count"],
        "pvalue": lambda x: (x < 0.05).sum()
    }).reset_index()
    domain_stats.columns = ["domain", "mean", "std", "n_dims", "n_sig"]
    domain_stats = domain_stats.sort_values("mean", ascending=False)

    fig, ax = create_figure("wide")

    x = np.arange(len(domain_stats))
    colors = [CMAP[1] if row["n_sig"] > 0 else GRAY["medium"]
              for _, row in domain_stats.iterrows()]

    bars = ax.bar(x, domain_stats["mean"], yerr=domain_stats["std"],
                  color=colors, capsize=3, edgecolor="white")

    ax.axhline(0, color=GRAY["dark"], linewidth=0.8)
    ax.axhline(df["correlation"].mean(), color=CMAP[0], linewidth=1.5,
               linestyle="--", label=f"Overall mean")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domain_stats["domain"]],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean correlation")
    ax.set_title("Prediction performance by domain")

    # Add counts
    for i, (_, row) in enumerate(domain_stats.iterrows()):
        label = f"{int(row['n_sig'])}/{int(row['n_dims'])}"
        ax.text(i, row["mean"] + row["std"] + 0.03, label,
                ha="center", fontsize=7, color=GRAY["dark"])

    despine(ax)
    plt.tight_layout()
    save_figure(fig, output_dir / "domain_comparison.pdf")


def plot_dimension_heatmap(results_path, output_dir):
    """Heatmap showing correlations for common dimensions across domains."""
    apply_theme()

    df = pd.read_csv(results_path)

    # Find dimensions that appear in multiple domains
    dim_counts = df["dimension"].value_counts()
    common_dims = dim_counts[dim_counts >= 2].index.tolist()

    # Pivot to matrix
    pivot = df[df["dimension"].isin(common_dims)].pivot(
        index="dimension", columns="domain", values="correlation"
    )

    # Sort by mean correlation
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    # Add correlation values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > 0.3 else GRAY["dark"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title("Prediction correlations by dimension and domain")

    plt.tight_layout()
    save_figure(fig, output_dir / "dimension_heatmap.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "prediction_results.csv"

    embeddings, word_to_idx, domains = load_data()

    print("Creating scatter plots...")
    plot_scatter_predictions(embeddings, word_to_idx, domains, OUTPUT_DIR)

    print("Creating correlation distribution...")
    plot_correlation_distribution(results_path, OUTPUT_DIR)

    print("Creating domain comparison...")
    plot_all_domains(results_path, OUTPUT_DIR)

    print("Creating dimension heatmap...")
    plot_dimension_heatmap(results_path, OUTPUT_DIR)

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
