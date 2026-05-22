"""Draft panels for THINGS figure: SRF vs VICE comparison.

Panel 1: Accuracy bar plot (SRF kappa vs VICE across data conditions)
Panel 2: Accuracy vs dimensions (efficiency scatter)

Uses 10 seeds for VICE, kappa_alpha0 for SRF rank selection.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.colors import ROSE, TEAL, GRAY, GRAY_LIGHT, GRAY_PALE, soft, setup_style
from src.utils.figure_theme import despine

setup_style()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior" / "lowdata" / "comparison" / "outputs" / "lowdata_comparison.csv"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

NOISE_CEILING = 0.6694
PCTS = [5, 10, 20, 50, 100]
FS = 5


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


def load():
    df = pd.read_csv(CSV)
    srf = df[df["condition"] == "kappa_alpha0"].copy()
    vice = df[df["condition"] == "vice"].copy()
    return pd.concat([srf, vice], ignore_index=True)


def stats(df, model, pct):
    sub = df[(df["model"] == model) & (df["pct"] == pct)]
    if len(sub) == 0:
        return np.nan, np.nan, 0
    return sub["val_acc"].mean(), sub["val_acc"].std(), int(sub["rank"].median())


def panel_bars(ax, df):
    """Grouped bars: SRF vs VICE accuracy across data conditions."""
    x = np.arange(len(PCTS))
    w = 0.32
    ann = FS - 0.5

    for offset, model, color in [(-w / 2, "SRF", TEAL), (w / 2, "VICE", ROSE)]:
        means, stds, ranks = [], [], []
        for pct in PCTS:
            m, s, k = stats(df, model, pct)
            means.append(m * 100)
            stds.append(s * 100)
            ranks.append(k)

        bars = ax.bar(x + offset, means, w, yerr=stds,
                      color=soft(color), edgecolor=color,
                      linewidth=0.5, label=model, zorder=3,
                      error_kw=dict(linewidth=0.4, capsize=1.5, capthick=0.4))

        for bar, k in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    str(k), ha="center", va="bottom", fontsize=ann - 0.5,
                    color=color, fontweight="bold")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5, zorder=1)
    ax.text(len(PCTS) - 0.6, NOISE_CEILING * 100 + 0.15, "noise ceiling",
            fontsize=ann, color=GRAY, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 69)
    ax.legend(fontsize=ann, loc="lower right", frameon=False)
    despine(ax)


def panel_efficiency(ax, df):
    """Accuracy vs dimensions: shows SRF achieves same with fewer dims."""
    ann = FS - 0.5

    for pct in PCTS:
        srf_m, srf_s, srf_k = stats(df, "SRF", pct)
        vice_m, vice_s, vice_k = stats(df, "VICE", pct)

        # Connecting line
        ax.plot([srf_k, vice_k], [srf_m * 100, vice_m * 100],
                color=GRAY_LIGHT, linewidth=0.4, zorder=2)

        ax.scatter(srf_k, srf_m * 100, color=TEAL, s=20, zorder=5,
                   edgecolors="white", linewidths=0.3)
        ax.scatter(vice_k, vice_m * 100, color=ROSE, s=20, zorder=5,
                   edgecolors="white", linewidths=0.3)

        # Label
        mid_x = (srf_k + vice_k) / 2
        mid_y = max(srf_m, vice_m) * 100 + 0.3
        ax.text(mid_x, mid_y, f"{pct}%", ha="center", va="bottom",
                fontsize=ann - 1, color=GRAY)

    ax.scatter([], [], color=TEAL, s=15, label="SRF")
    ax.scatter([], [], color=ROSE, s=15, label="VICE")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 75)
    ax.set_ylim(55, 69)
    ax.legend(fontsize=ann, loc="lower right", frameon=False)
    despine(ax)


def panel_lines(ax, df):
    """Line plot with confidence bands: accuracy vs data amount."""
    ann = FS - 0.5

    for model, color in [("SRF", TEAL), ("VICE", ROSE)]:
        means, stds = [], []
        for pct in PCTS:
            m, s, _ = stats(df, model, pct)
            means.append(m * 100)
            stds.append(s * 100)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(PCTS, means, color=color, marker="o", markersize=3,
                linewidth=0.8, label=model, zorder=3)
        ax.fill_between(PCTS, means - stds, means + stds,
                        color=color, alpha=0.15, linewidth=0, zorder=2)

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax.text(95, NOISE_CEILING * 100 + 0.15, "noise ceiling",
            fontsize=ann, color=GRAY, va="bottom", ha="right")

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 69)
    ax.set_xscale("log")
    ax.set_xticks(PCTS)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.minorticks_off()
    ax.legend(fontsize=ann, loc="lower right", frameon=False)
    despine(ax)


def main():
    plt.rcParams.update(_rc())
    df = load()

    # Option A: bars + efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 / 25.4, 40 / 25.4))
    panel_bars(ax1, df)
    panel_efficiency(ax2, df)
    ax1.text(-0.15, 1.12, "d", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")
    ax2.text(-0.15, 1.12, "e", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90, wspace=0.35)
    fig.savefig(OUTPUT / "option_a_bars_efficiency.png", dpi=300, facecolor="white")
    plt.close()
    print(f"Saved option_a_bars_efficiency.png")

    # Option B: bars + line plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 / 25.4, 40 / 25.4))
    panel_bars(ax1, df)
    panel_lines(ax2, df)
    ax1.text(-0.15, 1.12, "d", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")
    ax2.text(-0.15, 1.12, "e", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90, wspace=0.35)
    fig.savefig(OUTPUT / "option_b_bars_lines.png", dpi=300, facecolor="white")
    plt.close()
    print(f"Saved option_b_bars_lines.png")

    # Option C: line plot + efficiency (no bars)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 / 25.4, 40 / 25.4))
    panel_lines(ax1, df)
    panel_efficiency(ax2, df)
    ax1.text(-0.15, 1.12, "d", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")
    ax2.text(-0.15, 1.12, "e", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90, wspace=0.35)
    fig.savefig(OUTPUT / "option_c_lines_efficiency.png", dpi=300, facecolor="white")
    plt.close()
    print(f"Saved option_c_lines_efficiency.png")

    # Print summary table
    print("\nSummary:")
    for model in ["SRF", "VICE"]:
        for pct in PCTS:
            m, s, k = stats(df, model, pct)
            print(f"  {model} {pct:>3d}%: {m*100:.2f} +/- {s*100:.2f}%, k={k}")


if __name__ == "__main__":
    main()
