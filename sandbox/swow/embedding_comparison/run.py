"""Visualize SWOW PPMI embeddings: load existing rank-50/100, fit fresh rank-20.

Uses the exact stored similarity.npy from the original experiment to reproduce.
"""

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
    # Load the exact similarity used in experiments
    s = np.load(EMBED_DIR / "50" / "similarity.npy")
    with open(EMBED_DIR / "50" / "metadata.json") as f:
        meta = json.load(f)
    vocabulary = meta["vocabulary"]

    log.info(f"Similarity: {s.shape}, NaN={np.isnan(s).sum()}, "
             f"zeros={(s==0).sum()} ({(s==0).mean()*100:.1f}%), "
             f"positive={(s>0).sum()} ({(s>0).mean()*100:.1f}%)")
    log.info(f"Range: [{np.nanmin(s):.4f}, {np.nanmax(s):.4f}]")
    log.info(f"Diagonal: all NaN = {np.isnan(np.diag(s)).all()}")

    # Load existing embeddings
    for rank in [50, 100]:
        embed_path = EMBED_DIR / str(rank)
        if not embed_path.exists():
            log.info(f"No embedding at rank={rank}, skipping")
            continue

        w = np.load(embed_path / "word_embedding.npy")
        log.info(f"\nRank-{rank} embedding: {w.shape}, "
                 f"sparsity={(w < 1e-6).mean()*100:.1f}%, "
                 f"max={w.max():.4f}")

        fig = plot_top_words(w, vocabulary, f"PPMI rank={rank}")
        fig.savefig(OUTPUT_DIR / f"top_words_rank{rank}.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Fit fresh rank-20 using same parameters as original
    log.info("\nFitting rank=20 with same params (rho=3.0, missing_values=nan)...")
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
    log.info(f"Rank-20: converged in {model.n_iter_} iters, "
             f"shape={w20.shape}, sparsity={(w20 < 1e-6).mean()*100:.1f}%")

    fig = plot_top_words(w20, vocabulary, "PPMI rank=20")
    fig.savefig(OUTPUT_DIR / "top_words_rank20.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Sparsity comparison across ranks
    fig, axes = create_figure("full_width", ncols=3)
    for ax, rank_label, w_data in [
        (axes[0], 20, w20),
        (axes[1], 50, np.load(EMBED_DIR / "50" / "word_embedding.npy")),
        (axes[2], 100, np.load(EMBED_DIR / "100" / "word_embedding.npy")),
    ]:
        pct = [(w_data[:, d] > 1e-6).mean() for d in range(w_data.shape[1])]
        ax.bar(range(1, len(pct) + 1), pct, color=TEAL, alpha=0.8, width=0.8)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Fraction active")
        ax.set_title(f"rank={rank_label}")
        ax.set_ylim(0, 1)
        despine(ax)
    fig.suptitle("SWOW PPMI: sparsity per dimension", fontsize=11, y=1.02)
    fig.savefig(OUTPUT_DIR / "sparsity_comparison.png",
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
