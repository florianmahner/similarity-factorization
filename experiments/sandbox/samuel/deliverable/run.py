"""SRF factorization of Samuel mathematical concepts (rank=15).

Produces a shareable results package:
  - embedding.csv            : 429 concepts x 15 dimensions
  - predicted_similarity.csv : 429 x 429 reconstructed matrix
  - wordclouds.png           : one wordcloud per dimension
  - top_concepts.png         : bar chart of top 15 concepts per dimension
  - fit_summary.txt          : model fit statistics

Usage:
    ./scripts/submit sandbox/samuel/deliverable/run.py --bg
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud

from pysrf import SRF
from src.utils import get_output_dir
from src.colors import CYCLE
from src.utils.figure_theme import despine

OUTPUT_DIR = get_output_dir()
DATA_PATH = Path("data/samuel/similarityMatrix.csv")
RANK = 15


def load_and_preprocess() -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(DATA_PATH, index_col=0)
    labels = df.index.tolist()
    s = df.values.astype(float)
    n = s.shape[0]

    st = s.T
    has_s, has_st = ~np.isnan(s), ~np.isnan(st)
    sym = np.full((n, n), np.nan)
    sym[has_s & has_st] = (s[has_s & has_st] + st[has_s & has_st]) / 2
    sym[has_s & ~has_st] = s[has_s & ~has_st]
    sym[~has_s & has_st] = st[~has_s & has_st]

    max_val = np.nanmax(sym)
    np.fill_diagonal(sym, max_val)
    return sym, labels


def plot_wordclouds(embedding: np.ndarray, labels: list[str], path: Path) -> None:
    k = embedding.shape[1]
    n_cols = 5
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds",
             "YlGn", "BuPu", "OrRd", "GnBu", "YlOrRd"]

    for d in range(k):
        ax = axes[d // n_cols, d % n_cols]
        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:40]
        freq = {labels[i]: float(weights[i]) for i in top_idx if weights[i] > 1e-6}

        if freq:
            wc = WordCloud(
                width=600, height=400, background_color="white",
                max_words=40, colormap=cmaps[d % len(cmaps)],
                prefer_horizontal=0.7, relative_scaling=0.5, min_font_size=8,
            )
            wc.generate_from_frequencies(freq)
            ax.imshow(wc, interpolation="bilinear")

        ax.set_title(f"Dim {d+1}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for d in range(k, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].axis("off")

    fig.suptitle("SRF Dimensions (k=15) — Mathematical Concepts", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_top_concepts(embedding: np.ndarray, labels: list[str], path: Path, top_k: int = 15) -> None:
    k = embedding.shape[1]
    n_cols = 5
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 0.35 * top_k * n_rows))
    axes = np.atleast_2d(axes)
    colors = [CYCLE[i % len(CYCLE)] for i in range(k)]

    for d in range(k):
        ax = axes[d // n_cols, d % n_cols]
        weights = embedding[:, d]
        top_idx = np.argsort(weights)[::-1][:top_k]
        top_labs = [labels[i] for i in top_idx]
        top_w = weights[top_idx]

        ax.barh(range(top_k), top_w[::-1], color=colors[d], alpha=0.85)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_labs[::-1], fontsize=9)
        ax.set_xlabel("Weight")
        ax.set_title(f"Dim {d+1}", fontsize=10, fontweight="bold")
        despine(ax)

    for d in range(k, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print(f"SRF Factorization — Samuel Mathematical Concepts (rank={RANK})")
    print("=" * 60)

    sym, labels = load_and_preprocess()
    n = sym.shape[0]
    n_obs = int(np.sum(~np.isnan(sym)))
    print(f"{n} concepts, {n_obs} observed entries ({100*n_obs/(n*n):.1f}%)")

    print(f"\nFitting SRF (rank={RANK})...")
    model = SRF(rank=RANK, missing_values=np.nan, rho=3.0, verbose=1)
    embedding = model.fit_transform(sym)

    recon = embedding @ embedding.T
    mask = ~np.isnan(sym)
    mse = float(np.mean((sym[mask] - recon[mask]) ** 2))
    ss_res = float(np.sum((sym[mask] - recon[mask]) ** 2))
    ss_tot = float(np.sum((sym[mask] - np.mean(sym[mask])) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"MSE={mse:.4f}, R²={r2:.4f}")

    # Save data
    emb_df = pd.DataFrame(embedding, index=labels, columns=[f"dim_{d+1}" for d in range(RANK)])
    emb_df.index.name = "concept"
    emb_df.to_csv(OUTPUT_DIR / "embedding.csv")

    recon_df = pd.DataFrame(recon, index=labels, columns=labels)
    recon_df.to_csv(OUTPUT_DIR / "predicted_similarity.csv")

    summary_lines = [
        "SRF Factorization — Samuel Mathematical Concepts",
        "=" * 50,
        f"Concepts:           {n}",
        f"Observed entries:   {n_obs} / {n*n} ({100*n_obs/(n*n):.1f}%)",
        f"Rank (dimensions):  {RANK}",
        f"MSE (observed):     {mse:.4f}",
        f"R² (observed):      {r2:.4f}",
        "",
        "Method: Symmetric Non-negative Matrix Factorization (SRF)",
        "  S ≈ WW^T, W >= 0, W ∈ R^(n x k)",
        "  Loss: Frobenius norm on observed entries",
        "  Optimization: ADMM (rho=0.2)",
    ]
    (OUTPUT_DIR / "fit_summary.txt").write_text("\n".join(summary_lines))

    print("\nGenerating plots...")
    plot_wordclouds(embedding, labels, OUTPUT_DIR / "wordclouds.png")
    plot_top_concepts(embedding, labels, OUTPUT_DIR / "top_concepts.png")

    print(f"\nDone! Results in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
