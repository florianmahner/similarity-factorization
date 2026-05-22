#!/usr/bin/env python3
"""Plot reliability results with publication-quality figures."""

import numpy as np
from pathlib import Path
from src.colors import TEAL, ROSE, CYAN, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure

DATA_PATH = Path(__file__).parent / "outputs/260116_112528/reliability.npz"
OUTPUT_DIR = Path(__file__).parent / "outputs/260116_112528"


def main():
    data = np.load(DATA_PATH)
    reliab_n = data["reliab_n"]
    reliab_nnew = data["reliab_nnew"]
    reliab_combined = data["reliab_combined"]

    # Single combined figure
    fig, axes = create_figure("full_width", ncols=2)

    # Left: Individual recordings
    ax = axes[0]
    bins = np.linspace(-0.2, 1.0, 35)
    ax.hist(reliab_n, bins=bins, alpha=0.7, color=TEAL,
            label=f"Recording 1 (med={np.median(reliab_n):.2f})", edgecolor='white', linewidth=0.5)
    ax.hist(reliab_nnew, bins=bins, alpha=0.7, color=ROSE,
            label=f"Recording 2 (med={np.median(reliab_nnew):.2f})", edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Split-half reliability")
    ax.set_ylabel("Number of channels")
    ax.set_title("Individual recordings")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(-0.2, 1.0)
    despine(ax)

    # Right: Combined
    ax = axes[1]
    ax.hist(reliab_combined, bins=bins, alpha=0.8, color=CYAN, edgecolor='white', linewidth=0.5)
    ax.axvline(0.3, color=GRAY_DARK, ls='--', lw=1.5, label='Threshold (0.3)')
    ax.set_xlabel("Split-half reliability")
    ax.set_ylabel("Number of channels")
    n_above = (reliab_combined > 0.3).sum()
    ax.set_title(f"Combined (med={np.median(reliab_combined):.2f}, n>{0.3}: {n_above}/256)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(-0.2, 1.0)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "reliability.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'reliability.png'}")


if __name__ == "__main__":
    main()
