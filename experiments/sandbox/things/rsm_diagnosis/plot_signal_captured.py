"""THINGS figure panel: signal captured by SRF vs VICE.

Reads from validationset_comparison.csv (all evals on truly held-out data).
Two panels: signal captured bars + efficiency scatter.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.colors import ROSE, TEAL, INDIGO, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE, soft, lighten, setup_style
from src.utils.figure_theme import despine

setup_style()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "comparison" / "outputs" / "validationset_comparison.csv"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

NOISE_CEILING = 0.6694
CHANCE = 1.0 / 3.0
PCTS = [5, 10, 20, 50, 100]
FS = 5

# SRF = INDIGO (dark, authoritative), VICE = ROSE (comparison)
SRF_FILL = soft(INDIGO)
SRF_EDGE = INDIGO
VICE_FILL = soft(ROSE)
VICE_EDGE = ROSE


def _rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS + 1,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
    }


def signal_frac(acc):
    return (acc - CHANCE) / (NOISE_CEILING - CHANCE) * 100


def stats(df, model, pct):
    sub = df[(df["model"] == model) & (df["pct"] == pct)]
    if len(sub) == 0:
        return np.nan, np.nan, 0
    return sub["val_acc"].mean(), sub["val_acc"].std(), int(sub["rank"].median())


def main():
    plt.rcParams.update(_rc())
    ann = FS - 0.5

    df = pd.read_csv(CSV)

    srf_acc = [stats(df, "SRF", p)[0] for p in PCTS]
    vice_acc = [stats(df, "VICE", p)[0] for p in PCTS]
    srf_std = [stats(df, "SRF", p)[1] for p in PCTS]
    vice_std = [stats(df, "VICE", p)[1] for p in PCTS]
    srf_k = [stats(df, "SRF", p)[2] for p in PCTS]
    vice_k = [stats(df, "VICE", p)[2] for p in PCTS]

    srf_sig = [signal_frac(a) for a in srf_acc]
    vice_sig = [signal_frac(a) for a in vice_acc]
    srf_sig_std = [s / (NOISE_CEILING - CHANCE) * 100 for s in srf_std]
    vice_sig_std = [s / (NOISE_CEILING - CHANCE) * 100 for s in vice_std]

    # =========================================================================
    # Panel 1: Signal captured bars
    # =========================================================================
    fig, ax = plt.subplots(figsize=(70 / 25.4, 45 / 25.4))

    x = np.arange(len(PCTS)) * 1.4
    w = 0.34
    gap = 0.06  # gap between SRF and VICE bars within a group

    ax.bar(x - w / 2 - gap / 2, srf_sig, w, yerr=srf_sig_std,
           color=SRF_FILL, edgecolor=SRF_EDGE, linewidth=0.6,
           label="SRF", zorder=3,
           error_kw=dict(linewidth=0.4, capsize=1.5, capthick=0.4, color=GRAY_DARK))
    ax.bar(x + w / 2 + gap / 2, vice_sig, w, yerr=vice_sig_std,
           color=VICE_FILL, edgecolor=VICE_EDGE, linewidth=0.6,
           label="VICE", zorder=3,
           error_kw=dict(linewidth=0.4, capsize=1.5, capthick=0.4, color=GRAY_DARK))

    # Dimension annotations on top
    for i, (sk, vk) in enumerate(zip(srf_k, vice_k)):
        y_srf = srf_sig[i] + srf_sig_std[i] + 0.8
        y_vice = vice_sig[i] + vice_sig_std[i] + 0.8
        ax.text(x[i] - w / 2 - gap / 2, y_srf, str(sk),
                ha="center", va="bottom", fontsize=ann - 0.5, color=SRF_EDGE, fontweight="bold")
        ax.text(x[i] + w / 2 + gap / 2, y_vice, str(vk),
                ha="center", va="bottom", fontsize=ann - 0.5, color=VICE_EDGE, fontweight="bold")

    # Noise ceiling
    ax.axhline(100, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.5, zorder=1)
    ax.text(len(PCTS) - 0.55, 101, "noise ceiling", fontsize=ann, color=GRAY, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Signal captured (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=ann, loc="upper left", frameon=False,
              handlelength=1.0, handleheight=0.7)
    despine(ax)

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.92)
    fig.savefig(OUTPUT / "signal_bars.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT / "signal_bars.pdf")
    plt.close()
    print("Saved signal_bars.png/pdf")

    # =========================================================================
    # Panel 2: Efficiency -- accuracy vs dimensions, lines per method
    # =========================================================================
    fig, ax = plt.subplots(figsize=(70 / 25.4, 45 / 25.4))

    # SRF line + markers
    ax.plot(srf_k, srf_sig, color=SRF_EDGE, linewidth=0.8, zorder=3, alpha=0.6)
    ax.scatter(srf_k, srf_sig, color=SRF_FILL, s=35, zorder=5,
               edgecolors=SRF_EDGE, linewidths=0.6, label="SRF")

    # VICE line + markers
    ax.plot(vice_k, vice_sig, color=VICE_EDGE, linewidth=0.8, zorder=3, alpha=0.6)
    ax.scatter(vice_k, vice_sig, color=VICE_FILL, s=35, zorder=5,
               edgecolors=VICE_EDGE, linewidths=0.6, label="VICE")

    # Annotations next to SRF dots (right side)
    for i, pct in enumerate(PCTS):
        ax.annotate(f"{pct}%", (srf_k[i], srf_sig[i]),
                    xytext=(5, -1), textcoords="offset points",
                    fontsize=ann - 0.5, color=GRAY_DARK, va="center")

    # Noise ceiling
    ax.axhline(100, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.5)
    ax.text(70, 101, "noise ceiling", fontsize=ann, color=GRAY, va="bottom", ha="right")

    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Signal captured (%)")
    ax.set_xlim(0, 72)
    ax.set_ylim(70, 108)
    ax.legend(fontsize=ann, loc="lower right", frameon=False,
              handlelength=1.0, handleheight=0.7)
    despine(ax)

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.92)
    fig.savefig(OUTPUT / "signal_efficiency.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT / "signal_efficiency.pdf")
    plt.close()
    print("Saved signal_efficiency.png/pdf")


if __name__ == "__main__":
    main()
