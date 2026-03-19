"""Plot psycholinguistic prediction results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from src.colors import TEAL, ROSE, CYAN, GRAY, GRAY_DARK, GRAY_LIGHT, setup_style
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, save_figure, despine, SIZES, DEFAULT_PAD

PROJECT_ROOT = Path(__file__).parents[2]
EMBEDDING_DIR = PROJECT_ROOT / "outputs/experiments/word_association/generate_embedding"
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = get_output_dir()


def load_embeddings():
    """Load SWOW word embeddings."""
    emb_path = EMBEDDING_DIR / "50"
    embeddings = np.load(emb_path / "word_embedding.npy")
    with open(emb_path / "metadata.json") as f:
        metadata = json.load(f)
    vocabulary = metadata["vocabulary"]
    word_to_idx = {w.lower(): i for i, w in enumerate(vocabulary)}
    return embeddings, word_to_idx


def predict_norms(embeddings, word_to_idx, words, ratings):
    """Get cross-validated predictions using Ridge regression."""
    valid_mask = [w in word_to_idx for w in words]
    valid_words = [w for w, v in zip(words, valid_mask) if v]
    valid_ratings = ratings[valid_mask]

    if len(valid_words) < 20:
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


def predict_similarity(embeddings, word_to_idx, word1_list, word2_list, scores):
    """Predict word pair similarity using cosine similarity."""
    valid_mask = [(w1.lower() in word_to_idx and w2.lower() in word_to_idx)
                  for w1, w2 in zip(word1_list, word2_list)]

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if len(valid_indices) < 20:
        return None

    emb_sims = []
    human_scores = []
    for i in valid_indices:
        w1, w2 = word1_list[i].lower(), word2_list[i].lower()
        e1 = embeddings[word_to_idx[w1]]
        e2 = embeddings[word_to_idx[w2]]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        emb_sims.append(cos_sim)
        human_scores.append(scores[i])

    emb_sims = np.array(emb_sims)
    human_scores = np.array(human_scores)
    r, p = spearmanr(emb_sims, human_scores)

    return {
        "embedding_similarity": emb_sims,
        "human_scores": human_scores,
        "correlation": r,
        "pvalue": p,
        "n": len(emb_sims),
    }


def create_grid_figure(nrows: int, ncols: int, hspace: float = 0.4, wspace: float = 0.3,
                       pad_top: float = 0.1):
    """Create a grid figure with proper spacing using subplots_adjust."""
    setup_style()

    plot_w, plot_h = SIZES["square"]
    pl, pr, pb, pt = 0.5, 0.2, 0.5, pad_top

    total_plot_w = plot_w * ncols
    total_plot_h = plot_h * nrows

    fig_w = pl + total_plot_w + pr
    fig_h = pb + total_plot_h + pt

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    left_frac = pl / fig_w
    right_frac = (pl + total_plot_w) / fig_w
    bottom_frac = pb / fig_h
    top_frac = (pb + total_plot_h) / fig_h

    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        bottom=bottom_frac,
        top=top_frac,
        hspace=hspace,
        wspace=wspace,
    )

    return fig, axes


def plot_glasgow_norms(embeddings, word_to_idx):
    """Plot Glasgow Norms predictions as scatter grid."""
    df = pd.read_csv(DATA_DIR / "glasgow_norms.csv")
    df["word"] = df["word"].str.lower()

    dims = {
        "AROU": "Arousal",
        "VAL": "Valence",
        "DOM": "Dominance",
        "CNC": "Concreteness",
        "IMAG": "Imageability",
        "FAM": "Familiarity",
        "AOA": "AoA",
        "SIZE": "Size",
        "GEND": "Gender",
    }

    fig, axes = create_grid_figure(3, 3, hspace=0.6, wspace=0.4, pad_top=0.5)
    axes = axes.flatten()

    for idx, (col, name) in enumerate(dims.items()):
        ax = axes[idx]
        result = predict_norms(
            embeddings, word_to_idx,
            df["word"].tolist(),
            df[col].to_numpy()
        )

        if result is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, color=GRAY)
            ax.set_title(name)
            despine(ax)
            continue

        ax.scatter(result["true_values"], result["predictions"],
                   c=TEAL, s=3, alpha=0.3, edgecolor="none")

        z = np.polyfit(result["true_values"], result["predictions"], 1)
        p = np.poly1d(z)
        xlim = (result["true_values"].min(), result["true_values"].max())
        x_line = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(x_line, p(x_line), color=GRAY_DARK, linewidth=1.5, linestyle="--")

        sig = "*" if result["pvalue"] < 0.05 else ""
        ax.set_title(f"{name} (ρ={result['correlation']:.2f}{sig})")
        ax.set_xlabel("Actual rating")
        ax.set_ylabel("Predicted")
        despine(ax)

    save_figure(fig, OUTPUT_DIR / "glasgow_norms.pdf")
    print("Created glasgow_norms.pdf")


def plot_similarity_benchmarks(embeddings, word_to_idx):
    """Plot SimLex-999 and WordSim-353 scatter plots."""
    fig, axes = create_figure("wide", nrows=1, ncols=2, pad_top=0.4)

    simlex_df = pd.read_csv(DATA_DIR / "SimLex-999/SimLex-999.txt", sep="\t")
    simlex_result = predict_similarity(
        embeddings, word_to_idx,
        simlex_df["word1"].tolist(),
        simlex_df["word2"].tolist(),
        simlex_df["SimLex999"].to_numpy()
    )

    ax = axes[0]
    ax.scatter(simlex_result["human_scores"], simlex_result["embedding_similarity"],
               c=TEAL, s=15, alpha=0.5, edgecolor="white", linewidth=0.3)

    z = np.polyfit(simlex_result["human_scores"], simlex_result["embedding_similarity"], 1)
    p = np.poly1d(z)
    xlim = (simlex_result["human_scores"].min(), simlex_result["human_scores"].max())
    x_line = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x_line, p(x_line), color=GRAY_DARK, linewidth=1.5, linestyle="--")

    sig = "*" if simlex_result["pvalue"] < 0.05 else ""
    ax.set_title(f"SimLex-999 (ρ={simlex_result['correlation']:.2f}{sig}, n={simlex_result['n']})")
    ax.set_xlabel("Human similarity")
    ax.set_ylabel("Cosine similarity")
    despine(ax)

    wordsim_df = pd.read_csv(DATA_DIR / "wordsim353.csv", sep="\t", header=None,
                             names=["word1", "word2", "score"])
    wordsim_result = predict_similarity(
        embeddings, word_to_idx,
        wordsim_df["word1"].tolist(),
        wordsim_df["word2"].tolist(),
        wordsim_df["score"].to_numpy()
    )

    ax = axes[1]
    ax.scatter(wordsim_result["human_scores"], wordsim_result["embedding_similarity"],
               c=CYAN, s=15, alpha=0.5, edgecolor="white", linewidth=0.3)

    z = np.polyfit(wordsim_result["human_scores"], wordsim_result["embedding_similarity"], 1)
    p = np.poly1d(z)
    xlim = (wordsim_result["human_scores"].min(), wordsim_result["human_scores"].max())
    x_line = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x_line, p(x_line), color=GRAY_DARK, linewidth=1.5, linestyle="--")

    sig = "*" if wordsim_result["pvalue"] < 0.05 else ""
    ax.set_title(f"WordSim-353 (ρ={wordsim_result['correlation']:.2f}{sig}, n={wordsim_result['n']})")
    ax.set_xlabel("Human relatedness")
    ax.set_ylabel("Cosine similarity")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "similarity_benchmarks.pdf")
    print("Created similarity_benchmarks.pdf")


def plot_summary_bar(results_df):
    """Plot summary bar chart of all correlations."""
    fig, ax = create_figure("wide", pad_left=1.0)

    results_df = results_df.sort_values("correlation", ascending=True)

    y_pos = np.arange(len(results_df))
    colors = [TEAL if d == "Glasgow Norms" else CYAN if d == "WordSim-353" else ROSE
              for d in results_df["dataset"]]

    ax.barh(y_pos, results_df["correlation"], color=colors, edgecolor="white", linewidth=0.5)

    labels = [f"{row['dimension']}" if row["dataset"] == "Glasgow Norms"
              else row["dataset"] for _, row in results_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(0, 1)

    for i, (_, row) in enumerate(results_df.iterrows()):
        sig = "*" if row["pvalue"] < 0.05 else ""
        ax.text(row["correlation"] + 0.02, i, f"{row['correlation']:.2f}{sig}",
                va="center", fontsize=8)

    despine(ax)
    save_figure(fig, OUTPUT_DIR / "summary_bar.pdf")
    print("Created summary_bar.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, word_to_idx = load_embeddings()

    print("Creating Glasgow Norms scatter plot...")
    plot_glasgow_norms(embeddings, word_to_idx)

    print("Creating similarity benchmarks plot...")
    plot_similarity_benchmarks(embeddings, word_to_idx)

    print("Creating summary bar chart...")
    results_df = pd.read_csv(OUTPUT_DIR / "prediction_results.csv")
    plot_summary_bar(results_df)

    print("\nAll plots created!")


if __name__ == "__main__":
    main()
