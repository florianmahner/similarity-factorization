"""Generate RSA comparison figure, 180mm Nature double-column width.

Layout:
  a) Factorial design schematic (placeholder)
  b) Power curve: Factorial simulation
  c) Power curve: SPoSE semantic embedding

Imports panel function from the stable analysis plotting code.
All data comes from experiments/analyses/rsa/ outputs only.

Usage:
    poetry run python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.colors import GRAY, GRAY_LIGHT, INDIGO, PURPLE, ROSE, SAND, TEAL, CYAN, setup_style
from src.utils.figure_theme import despine

from experiments.analyses.rsa.plotting import (
    load_results,
    plot_power_paper,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RSA_DIR = PROJECT_ROOT / "experiments" / "analyses" / "rsa"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4


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


def _panel_schematic(ax: plt.Axes) -> None:
    """a) Factorial design schematic placeholder."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Left side: Hypotheses -> Dimensions
    ax.text(1.2, 0.92, "Hypotheses", ha="center", va="top", fontsize=5,
            fontweight="bold", color=GRAY)

    colors = [INDIGO, SAND, TEAL, PURPLE]
    labels = ["H1", "H2", "H3", "H4"]
    for i, (c, lbl) in enumerate(zip(colors, labels)):
        y = 0.75 - i * 0.17
        ax.barh(y, 0.8, height=0.1, left=0.4, color=c, alpha=0.6)
        ax.text(0.3, y, lbl, ha="right", va="center", fontsize=4, color=c)

    # Right side: Factorial design (block RSMs)
    ax.text(4.5, 0.92, "Factorial Design", ha="center", va="top", fontsize=5,
            fontweight="bold", color=GRAY)

    for i, c in enumerate(colors):
        x0 = 3.2 + (i % 2) * 1.3
        y0 = 0.55 - (i // 2) * 0.35
        for row in range(3):
            for col in range(3):
                alpha = 0.7 if row == col else 0.15
                ax.add_patch(plt.Rectangle(
                    (x0 + col * 0.25, y0 + row * 0.08), 0.22, 0.06,
                    facecolor=c, alpha=alpha, edgecolor="none"))

    # Center: Measured RSM
    ax.text(7.5, 0.92, "Measured RSM", ha="center", va="top", fontsize=5,
            fontweight="bold", color=GRAY)

    rng = np.random.default_rng(42)
    n = 8
    w = rng.dirichlet(np.ones(4) * 0.5, size=n)
    s = w @ w.T + rng.normal(0, 0.05, (n, n))
    np.fill_diagonal(s, 1)

    inset = ax.inset_axes([0.62, 0.10, 0.25, 0.70])
    inset.imshow(s, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.3)

    # Arrows
    ax.annotate("", xy=(3.0, 0.5), xytext=(2.0, 0.5),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))
    ax.annotate("", xy=(6.0, 0.5), xytext=(5.5, 0.5),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))

    # Bottom labels
    ax.text(1.2, 0.03, "SRF", ha="center", fontsize=5, fontweight="bold", color=TEAL)
    ax.text(4.5, 0.03, "RSA", ha="center", fontsize=5, fontweight="bold", color=ROSE)


def main():
    setup_style()

    fs = 5
    plt.rcParams.update(_nature_rc(fs))

    factorial_csv = RSA_DIR / "factorial" / "outputs" / "factorial.csv"
    spose_csv = RSA_DIR / "spose" / "outputs" / "spose.csv"

    df_factorial = load_results(factorial_csv)
    df_spose = load_results(spose_csv)

    ROW_H = 40 / 25.4  # 40mm per row
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, ROW_H + 0.3),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    for label, ax in zip("abc", axes):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")

    _panel_schematic(axes[0])

    plot_power_paper(axes[1], df_factorial, srf_method="SRF-LOO")
    axes[1].set_title("Simulation", fontsize=fs, fontweight="normal", color=GRAY)

    plot_power_paper(axes[2], df_spose, srf_method="SRF-LOO")
    axes[2].set_title("SPoSE semantic embedding", fontsize=fs, fontweight="normal", color=GRAY)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.20, top=0.88,
                        wspace=0.40)
    _save(fig, "rsa_comparison")


if __name__ == "__main__":
    main()
