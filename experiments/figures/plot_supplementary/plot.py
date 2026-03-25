"""Generate supplementary figures for the paper, 180mm Nature double-column width.

Figures:
  S1: variance_quartile.pdf -- Power vs dimension variance + example dimensions
  S2: similarity_benchmarks.pdf -- SimLex-999 and WordSim-353 scatter plots

Data sources (all from stable experiments/analyses/ or data/):
  - experiments/analyses/rsa/spose/outputs/spose.csv
  - data/things/spose_embedding_66d.txt
  - data/things/labels_spose_66d_short.txt
  - experiments/datasets/consensus/outputs/swow/embedding.npy
  - data/word_similarity/SimLex-999/SimLex-999.txt
  - data/word_similarity/WordSim-353/wordsim353.tsv
  - data/small-world-of-words/ (vocabulary via loader)

Usage:
    poetry run python experiments/figures/plot_supplementary/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.colors import ROSE, TEAL, GRAY, GRAY_LIGHT, GRAY_DARK, soft, setup_style
from src.utils.figure_theme import despine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
ROW_H = 40 / 25.4  # 40mm row height, same as main figures


def _nature_rc(font_size: float) -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{name}.pdf"
    plt.rcParams["savefig.bbox"] = "standard"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig S1: Variance quartile -- power vs dimension variance
# ---------------------------------------------------------------------------

def _panel_power_vs_variance(ax: plt.Axes, df: pd.DataFrame, fs: float) -> None:
    """Binned power curves for RSA and SRF."""
    snr_val = 1.0 if 1.0 in df["snr"].values else df["snr"].max()
    snr_df = df[df["snr"] == snr_val]
    ann = fs - 0.5

    n_bins = 10
    for method, color, label in [("RSA", ROSE, "RSA"), ("SRF-LOO", TEAL, "SRF")]:
        m = snr_df[snr_df["method"] == method].copy()
        m["var_bin"] = pd.qcut(m["dim_var"], q=n_bins, duplicates="drop")
        grouped = m.groupby("var_bin", observed=True).agg(
            var_mean=("dim_var", "mean"),
            power=("significant", "mean"),
            power_se=("significant", "sem"),
        ).reset_index()

        ax.plot(grouped["var_mean"], grouped["power"] * 100,
                color=color, lw=0.8, label=label)
        ax.fill_between(
            grouped["var_mean"],
            (grouped["power"] - 1.96 * grouped["power_se"]) * 100,
            (grouped["power"] + 1.96 * grouped["power_se"]) * 100,
            color=color, alpha=0.15,
        )

    ax.axhline(5, color=GRAY, ls="--", lw=0.5, zorder=0)
    ax.set_xlabel("Dimension variance (sparsity)")
    ax.set_ylabel("Power (%)")
    ax.set_ylim([-5, 105])
    ax.legend(loc="lower right", frameon=False, fontsize=ann)
    despine(ax)


def _panel_example_dimensions(ax: plt.Axes, fs: float) -> None:
    """Bar chart of example SPoSE dimensions sorted by variance."""
    embed_path = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
    x = np.maximum(np.loadtxt(embed_path), 0)
    label_path = PROJECT_ROOT / "data" / "things" / "labels_spose_66d_short.txt"
    labels = [
        line.strip()
        for line in label_path.read_text().strip().split("\n")
        if line.strip()
    ]

    var_per_dim = np.var(x, axis=0)
    sorted_idx = np.argsort(var_per_dim)

    n_show = 5
    low_idx = sorted_idx[:n_show]
    high_idx = sorted_idx[-n_show:][::-1]
    show_idx = np.concatenate([high_idx, low_idx])

    show_labels = [labels[i] if i < len(labels) else f"dim_{i}" for i in show_idx]
    show_vars = var_per_dim[show_idx]

    y_pos = np.arange(len(show_idx))
    ax.barh(y_pos, show_vars, color=soft(TEAL), edgecolor=TEAL,
            linewidth=0.4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(show_labels, fontsize=fs)
    ax.set_xlabel("Dimension variance (sparsity)")
    ax.invert_yaxis()
    ax.axhline(n_show - 0.5, color=GRAY_LIGHT, lw=0.5, ls="--")
    despine(ax)


def plot_variance_quartile():
    """Fig S1: two-panel figure, power vs dimension variance + example dims."""
    from experiments.analyses.rsa.plotting import load_results

    spose_csv = (
        PROJECT_ROOT / "experiments" / "analyses" / "rsa"
        / "spose" / "outputs" / "spose.csv"
    )
    df = load_results(spose_csv)

    fs = 5
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, ROW_H),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.50},
    )

    _panel_power_vs_variance(axes[0], df, fs)
    _panel_example_dimensions(axes[1], fs)

    for label, ax in zip("ab", axes):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90)
    _save(fig, "variance_quartile")


# ---------------------------------------------------------------------------
# Fig S2: Similarity benchmarks -- SimLex-999 + WordSim-353
# ---------------------------------------------------------------------------

def _load_swow_embedding():
    """Load SWOW consensus embedding and vocabulary."""
    emb_path = (
        PROJECT_ROOT / "experiments" / "datasets" / "consensus"
        / "outputs" / "swow" / "embedding.npy"
    )
    embedding = np.load(emb_path)

    from src.datasets.swow import load_swow_ppmi
    data_dir = PROJECT_ROOT / "data" / "small-world-of-words"
    _, vocabulary, _ = load_swow_ppmi(data_dir, use_all_responses=True)

    return embedding, {w.lower(): i for i, w in enumerate(vocabulary)}


def _compute_cosine_similarity(embedding, word_to_idx, word1_list, word2_list, scores):
    """Compute cosine similarity for word pairs and return stats."""
    emb_sims = []
    human_scores = []
    for w1, w2, s in zip(word1_list, word2_list, scores):
        w1_lower, w2_lower = w1.lower(), w2.lower()
        if w1_lower not in word_to_idx or w2_lower not in word_to_idx:
            continue
        e1 = embedding[word_to_idx[w1_lower]]
        e2 = embedding[word_to_idx[w2_lower]]
        norm = np.linalg.norm(e1) * np.linalg.norm(e2)
        if norm == 0:
            continue
        emb_sims.append(np.dot(e1, e2) / norm)
        human_scores.append(s)

    emb_sims = np.array(emb_sims)
    human_scores = np.array(human_scores)
    rho, _ = spearmanr(emb_sims, human_scores)
    return {"sims": emb_sims, "scores": human_scores, "rho": rho, "n": len(emb_sims)}


def _panel_scatter(ax: plt.Axes, result: dict, xlabel: str, fs: float) -> None:
    """Scatter plot of embedding similarity vs human ratings."""
    ax.scatter(
        result["scores"], result["sims"],
        color=TEAL, alpha=0.4, s=4,
        linewidths=0.2, edgecolors="white", zorder=2, rasterized=True,
    )

    z = np.polyfit(result["scores"], result["sims"], 1)
    x_line = np.linspace(result["scores"].min(), result["scores"].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color=GRAY_DARK,
            linewidth=0.6, linestyle="--", zorder=1)

    ax.text(
        0.05, 0.95,
        f"$\\rho$ = {result['rho']:.2f}\nn = {result['n']}",
        transform=ax.transAxes, va="top", fontsize=fs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SRF embedding similarity")
    despine(ax)


def plot_similarity_benchmarks():
    """Fig S2: SimLex-999 + WordSim-353 scatter plots, side by side."""
    embedding, word_to_idx = _load_swow_embedding()

    # SimLex-999
    simlex_path = (
        PROJECT_ROOT / "data" / "word_similarity"
        / "SimLex-999" / "SimLex-999.txt"
    )
    simlex_df = pd.read_csv(simlex_path, sep="\t")
    simlex = _compute_cosine_similarity(
        embedding, word_to_idx,
        simlex_df["word1"].tolist(), simlex_df["word2"].tolist(),
        simlex_df["SimLex999"].to_numpy(),
    )

    # WordSim-353
    wordsim_path = (
        PROJECT_ROOT / "data" / "word_similarity"
        / "WordSim-353" / "wordsim353.tsv"
    )
    wordsim_df = pd.read_csv(wordsim_path, sep="\t", header=None,
                              names=["word1", "word2", "score"])
    wordsim = _compute_cosine_similarity(
        embedding, word_to_idx,
        wordsim_df["word1"].tolist(), wordsim_df["word2"].tolist(),
        wordsim_df["score"].to_numpy(),
    )

    fs = 5
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, ROW_H),
        gridspec_kw={"wspace": 0.35},
    )

    _panel_scatter(axes[0], simlex, "Human similarity (SimLex-999)", fs)
    _panel_scatter(axes[1], wordsim, "Human relatedness (WordSim-353)", fs)

    for label, ax in zip("ab", axes):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.90)
    _save(fig, "similarity_benchmarks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()
    plt.rcParams.update(_nature_rc(5))

    plot_variance_quartile()
    plot_similarity_benchmarks()
    print("Done.")


if __name__ == "__main__":
    main()
