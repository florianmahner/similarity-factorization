"""Fit rank-20 SRF embedding on SWOW PPMI and visualize."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pysrf import SRF
from src.utils import get_output_dir
from src.colors import TEAL
from src.utils.figure_theme import create_figure, despine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
EMBED_DIR = Path("outputs/experiments/word_association/generate_embedding")


def plot_top_words(w, vocabulary, name, n_dims=5, top_n=10):
    fig, axes = plt.subplots(1, n_dims, figsize=(3 * n_dims, 4))
    for d in range(n_dims):
        ax = axes[d]
        top_idx = np.argsort(w[:, d])[-top_n:][::-1]
        words = [vocabulary[i] for i in top_idx]
        loadings = w[top_idx, d]
        ax.barh(range(top_n), loadings, color=TEAL)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(words, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Dim {d+1}", fontsize=9)
        ax.set_xlabel("Loading", fontsize=7)
        despine(ax)
    fig.suptitle(f"Top words per dimension ({name})", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    s = np.load(EMBED_DIR / "50" / "similarity.npy")
    with open(EMBED_DIR / "50" / "metadata.json") as f:
        meta = json.load(f)
    vocabulary = meta["vocabulary"]

    log.info(f"Similarity: {s.shape}, NaN={np.isnan(s).sum()}, "
             f"zeros={(s==0).sum()} ({(s==0).mean()*100:.1f}%), "
             f"positive={(s>0).sum()} ({(s>0).mean()*100:.1f}%)")

    # Fit rank-20 with same params as original experiments
    log.info("Fitting SRF rank=20 (rho=3.0, missing_values=nan)...")
    model = SRF(
        rank=20,
        rho=3.0,
        max_outer=1000,
        max_inner=50,
        tol=1e-4,
        verbose=1,
        random_state=42,
        missing_values=np.nan,
    )
    model.fit(s)
    w20 = model.w_
    log.info(f"Converged in {model.n_iter_} iters, shape={w20.shape}, "
             f"sparsity={(w20 < 1e-6).mean()*100:.1f}%")

    # Load existing rank-50
    w50 = np.load(EMBED_DIR / "50" / "word_embedding.npy")
    log.info(f"Rank-50: shape={w50.shape}, sparsity={(w50 < 1e-6).mean()*100:.1f}%")

    # Top words
    fig = plot_top_words(w20, vocabulary, "PPMI rank=20")
    fig.savefig(OUTPUT_DIR / "top_words_rank20.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig = plot_top_words(w50, vocabulary, "PPMI rank=50")
    fig.savefig(OUTPUT_DIR / "top_words_rank50.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Sparsity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, w, label in [(axes[0], w20, "rank=20"), (axes[1], w50, "rank=50")]:
        pct = [(w[:, d] > 1e-6).mean() for d in range(w.shape[1])]
        ax.bar(range(1, len(pct) + 1), pct, color=TEAL, alpha=0.8)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Fraction active")
        ax.set_title(label)
        ax.set_ylim(0, 1)
        despine(ax)
    fig.suptitle("SWOW PPMI: sparsity per dimension")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sparsity.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Eigenspectrum
    fig, ax = create_figure("wide")
    s_clean = np.nan_to_num(s, nan=0.0)
    s_sym = 0.5 * (s_clean + s_clean.T)
    evals = np.linalg.eigvalsh(s_sym)[::-1][:200]
    ax.plot(range(1, 201), evals, linewidth=1.5, color=TEAL)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("SWOW PPMI eigenspectrum (top 200)")
    despine(ax)
    fig.savefig(OUTPUT_DIR / "eigenspectrum.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.save(OUTPUT_DIR / "embedding_rank20.npy", w20)
    log.info(f"\nAll saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
