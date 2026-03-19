"""Plot NetMF-SRF comparison results."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(exist_ok=True)

# Results from the experiment
RESULTS = {
    "method": ["DeepWalk", "Node2Vec", "SRF (raw AA)", "SRF (log AA)", "SRF (NetMF)"],
    "rare": [0.273, 0.246, 0.218, 0.184, 0.152],
    "medium": [0.453, 0.453, 0.467, 0.438, 0.425],
    "frequent": [0.781, 0.809, 0.777, 0.797, 0.792],
}

METHOD_COLORS = {
    "DeepWalk": TEAL,
    "Node2Vec": CYAN,
    "SRF (raw AA)": ROSE,
    "SRF (log AA)": SAND,
    "SRF (NetMF)": PURPLE,
}


def main():
    df = pd.DataFrame(RESULTS)
    df["average"] = df[["rare", "medium", "frequent"]].mean(axis=1)

    bins = ["rare", "medium", "frequent"]
    n_methods = len(df)
    n_bins = len(bins)

    fig, ax = create_figure("wide", pad_bottom=0.6, pad_right=0.3)

    x = np.arange(n_bins)
    width = 0.15
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[b] for b in bins]
        color = METHOD_COLORS[row["method"]]
        ax.bar(x + offsets[i], values, width * 0.9, label=row["method"], color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bins])
    ax.set_xlabel("GO term frequency bin")
    ax.set_ylabel("Micro F1")
    ax.set_ylim(0, 0.9)

    ax.legend(loc="upper left", fontsize=7, ncol=2)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "method_comparison.pdf")
    print(f"Saved: {OUTPUT_DIR / 'method_comparison.pdf'}")

    # Also print average performance
    print("\nAverage micro_f1:")
    for _, row in df.sort_values("average", ascending=False).iterrows():
        print(f"  {row['method']:15s}: {row['average']:.3f}")


if __name__ == "__main__":
    main()
