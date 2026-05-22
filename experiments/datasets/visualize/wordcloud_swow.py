"""Plot SWOW consensus embedding dimensions as multi-page wordcloud grids.

Each PDF page is a grid of wordcloud thumbnails (one per embedding dimension).
The dimensions are split into ``--n-parts`` PDF files; pass any number, or
let the default of 3 stand (~90 dims per page for the 268-dim SWOW embedding).

Reads:  experiments/datasets/consensus/outputs/swow/embedding.npy
Writes: experiments/datasets/visualize/outputs/swow/wordclouds_partN.pdf

Usage:
    python -m experiments.datasets.visualize.wordcloud_swow
    python -m experiments.datasets.visualize.wordcloud_swow --n-parts 5
    python -m experiments.datasets.visualize.wordcloud_swow --n-cols 8 --n-words 30
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

from src.colors import setup_style

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSENSUS_DIR = PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs" / "swow"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "swow"

ARIAL_PATH = fm.findfont(fm.FontProperties(family="Arial"))
CMAPS = ["RdPu", "BuGn", "YlOrBr", "PuBu", "OrRd", "GnBu", "PuRd", "YlGn", "BuPu"]
FONT_SIZE = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-parts", type=int, default=3,
                        help="Number of PDF parts to split the grid into (default 3).")
    parser.add_argument("--n-cols", type=int, default=9,
                        help="Wordclouds per row (default 9).")
    parser.add_argument("--n-words", type=int, default=35,
                        help="Top-N words per wordcloud (default 35).")
    return parser.parse_args()


def _load_embedding() -> np.ndarray:
    path = CONSENSUS_DIR / "embedding.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"SWOW consensus embedding not found: {path}\n"
            f"Run the SWOW consensus pipeline first."
        )
    return np.load(path)


def _load_vocabulary() -> list[str]:
    from datasets import load_swow_ppmi

    _, vocabulary, _ = load_swow_ppmi(
        DATA_DIR / "small-world-of-words", use_all_responses=True,
    )
    return vocabulary


def _wordcloud_image(embedding: np.ndarray, vocabulary: list[str], dim: int, n_words: int):
    top_idx = np.argsort(embedding[:, dim])[::-1][:n_words]
    freqs = {
        vocabulary[i]: max(float(embedding[i, dim]), 0.01)
        for i in top_idx
    }
    return WordCloud(
        width=420,
        height=420,
        background_color="white",
        font_path=ARIAL_PATH,
        colormap=CMAPS[dim % len(CMAPS)],
        relative_scaling=0.4,
        min_font_size=5,
        max_font_size=105,
        prefer_horizontal=0.75,
        margin=3,
    ).generate_from_frequencies(freqs).to_array()


def _plot_part(
    embedding: np.ndarray,
    vocabulary: list[str],
    dims: range,
    part: int,
    n_cols: int,
    n_words: int,
) -> Path:
    n_rows = ceil(len(dims) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0, n_rows * 1.25),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, dim in zip(axes, dims):
        ax.imshow(_wordcloud_image(embedding, vocabulary, dim, n_words))
        ax.set_title(f"Dim {dim}", fontsize=FONT_SIZE, pad=1)
        ax.set_axis_off()

    for ax in axes[len(dims):]:
        ax.set_axis_off()

    out_path = OUTPUT_DIR / f"wordclouds_part{part}.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    args = _parse_args()
    setup_style()
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "savefig.bbox": "tight",
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embedding = _load_embedding()
    vocabulary = _load_vocabulary()

    n_dims = embedding.shape[1]
    part_size = ceil(n_dims / args.n_parts)
    print(f"SWOW embedding: {embedding.shape}; writing {args.n_parts} PDF parts "
          f"({part_size} dims/part, n_cols={args.n_cols}, n_words={args.n_words})")

    for part in range(1, args.n_parts + 1):
        start = (part - 1) * part_size
        stop = min(part * part_size, n_dims)
        if start >= n_dims:
            break
        _plot_part(embedding, vocabulary, range(start, stop), part, args.n_cols, args.n_words)


if __name__ == "__main__":
    main()
