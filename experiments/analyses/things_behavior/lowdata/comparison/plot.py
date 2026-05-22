"""Low-data comparison: SRF vs VICE.

Reads ONLY from lowdata_comparison.csv (produced by run.py).
No hardcoded numbers. No separate data sources.

Figures:
  1. accuracy_vs_data.pdf  -- grouped bars with dimension annotations
  2. efficiency.pdf        -- accuracy vs dimensions scatter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import GRAY, GRAY_LIGHT, GRAY_DARK, INDIGO, ROSE, TEAL, soft
from src.utils.figure_theme import create_figure, despine, save_figure

COMPARISON_CSV = Path(__file__).resolve().parent / "outputs" / "lowdata_comparison.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
NOISE_CEILING = 0.6694
CHANCE = 1.0 / 3.0
PCTS = [5, 10, 20, 50, 100]

# Colors for signal plot: SRF = INDIGO, VICE = ROSE
_SRF_FILL = soft(INDIGO)
_SRF_EDGE = INDIGO
_VICE_FILL = soft(ROSE)
_VICE_EDGE = ROSE


def load() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_CSV)


def load_validation() -> pd.DataFrame:
    """Load comparison data (now evaluated on validationset.txt)."""
    return pd.read_csv(COMPARISON_CSV)


def _stats(df: pd.DataFrame, model: str, pct: int) -> tuple[float, float, int]:
    sub = df[(df["model"] == model) & (df["pct"] == pct)]
    if len(sub) == 0:
        return np.nan, np.nan, 0
    return sub["val_acc"].mean(), sub["val_acc"].std(), int(sub["rank"].median())


# ---------------------------------------------------------------------------
# Panel functions (reusable: accept ax + data, no figure creation)
# ---------------------------------------------------------------------------

def panel_accuracy_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    annotation_fontsize: float = 5,
) -> None:
    """Grouped bars with dimension count annotated on top."""
    x = np.arange(len(PCTS))
    w = 0.32
    afs = annotation_fontsize

    for offset, model, color in [(-w / 2, "SRF", TEAL), (w / 2, "VICE", ROSE)]:
        means, ranks = [], []
        for pct in PCTS:
            m, _, k = _stats(df, model, pct)
            means.append(m * 100)
            ranks.append(k)

        bars = ax.bar(x + offset, means, w, color=soft(color), edgecolor=color,
                      linewidth=0.5, label=model, zorder=3)

        for bar, k in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                    str(k), ha="center", va="bottom", fontsize=afs - 0.5,
                    color=color, fontweight="bold")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5, zorder=1)
    ax.text(len(PCTS) - 0.6, NOISE_CEILING * 100 + 0.2, "noise ceiling",
            fontsize=afs, color=GRAY_LIGHT, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 69)
    ax.legend(fontsize=afs, loc="lower right")
    despine(ax)


def panel_efficiency(
    ax: plt.Axes,
    df: pd.DataFrame,
    annotation_fontsize: float = 5,
) -> None:
    """Accuracy vs dimensions scatter with paired connections."""
    afs = annotation_fontsize

    for i, pct in enumerate(PCTS):
        srf_m, _, srf_k = _stats(df, "SRF", pct)
        vice_m, _, vice_k = _stats(df, "VICE", pct)

        ax.plot([srf_k, vice_k], [srf_m * 100, vice_m * 100],
                color=GRAY_LIGHT, linewidth=0.4, zorder=2)
        ax.scatter(srf_k, srf_m * 100, color=TEAL, s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)
        ax.scatter(vice_k, vice_m * 100, color=ROSE, s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)

        mid_x = (srf_k + vice_k) / 2
        mid_y = (srf_m + vice_m) / 2 * 100
        ax.text(mid_x, mid_y + 0.4, f"{pct}%", ha="center", va="bottom",
                fontsize=afs - 1, color=GRAY)

    ax.scatter([], [], color=TEAL, s=20, label="SRF")
    ax.scatter([], [], color=ROSE, s=20, label="VICE")
    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 75)
    ax.set_ylim(55, 69)
    ax.legend(fontsize=afs, loc="lower right")
    despine(ax)


def _signal_frac(acc: float) -> float:
    return (acc - CHANCE) / (NOISE_CEILING - CHANCE) * 100


def _stats_cond(df: pd.DataFrame, model: str, condition: str, pct: int) -> tuple[float, float, int]:
    sub = df[(df["model"] == model) & (df["condition"] == condition) & (df["pct"] == pct)]
    if len(sub) == 0:
        return np.nan, np.nan, 0
    return sub["val_acc"].mean(), sub["val_acc"].std(), int(sub["rank"].median())


def panel_signal_captured(
    ax: plt.Axes,
    df: pd.DataFrame,
    srf_condition: str = "kappa_alpha0",
    annotation_fontsize: float = 5,
) -> None:
    """Grouped bars showing fraction of recoverable signal captured."""
    afs = annotation_fontsize
    x = np.arange(len(PCTS)) * 1.4
    w = 0.34
    gap = 0.06

    srf_sig, vice_sig = [], []
    srf_std, vice_std = [], []
    srf_k, vice_k = [], []

    for pct in PCTS:
        sm, ss, sk = _stats_cond(df, "SRF", srf_condition, pct)
        vm, vs, vk = _stats_cond(df, "VICE", "vice", pct)
        srf_sig.append(_signal_frac(sm))
        vice_sig.append(_signal_frac(vm))
        srf_std.append(ss / (NOISE_CEILING - CHANCE) * 100)
        vice_std.append(vs / (NOISE_CEILING - CHANCE) * 100)
        srf_k.append(sk)
        vice_k.append(vk)

    ax.bar(x - w / 2 - gap / 2, srf_sig, w, yerr=srf_std,
           color=_SRF_FILL, edgecolor=_SRF_EDGE, linewidth=0.6,
           label="SRF", zorder=3,
           error_kw=dict(linewidth=0.4, capsize=1.5, capthick=0.4, color=GRAY_DARK))
    ax.bar(x + w / 2 + gap / 2, vice_sig, w, yerr=vice_std,
           color=_VICE_FILL, edgecolor=_VICE_EDGE, linewidth=0.6,
           label="VICE", zorder=3,
           error_kw=dict(linewidth=0.4, capsize=1.5, capthick=0.4, color=GRAY_DARK))

    for i in range(len(PCTS)):
        y_srf = srf_sig[i] + srf_std[i] + 0.8
        y_vice = vice_sig[i] + vice_std[i] + 0.8
        ax.text(x[i] - w / 2 - gap / 2, y_srf, str(srf_k[i]),
                ha="center", va="bottom", fontsize=afs - 0.5,
                color=_SRF_EDGE, fontweight="bold")
        ax.text(x[i] + w / 2 + gap / 2, y_vice, str(vice_k[i]),
                ha="center", va="bottom", fontsize=afs - 0.5,
                color=_VICE_EDGE, fontweight="bold")

    ax.axhline(100, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.5, zorder=1)
    ax.text(x[-1] + 0.5, 101, "noise ceiling", fontsize=afs, color=GRAY,
            va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Signal captured (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=afs, loc="upper left", frameon=False,
              handlelength=1.0, handleheight=0.7)
    despine(ax)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

def main():
    df = load()

    print("Data summary (test_10):")
    for pct in PCTS:
        for model in ["SRF", "VICE"]:
            m, s, k = _stats(df, model, pct)
            if not np.isnan(m):
                print(f"  {model} {pct:>3d}%: {m*100:.2f}% +/- {s*100:.2f}% (k={k})")

    fig, ax = create_figure("wide")
    panel_accuracy_bars(ax, df)
    save_figure(fig, OUTPUT_DIR / "accuracy_vs_data.pdf")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'accuracy_vs_data.pdf'}")

    fig, ax = create_figure("wide")
    panel_efficiency(ax, df)
    save_figure(fig, OUTPUT_DIR / "efficiency.pdf")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'efficiency.pdf'}")

    # Signal captured (kappa_alpha0 for SRF, vice for VICE)
    print("\nSignal captured (kappa_alpha0 vs VICE):")
    for pct in PCTS:
        for model, cond in [("SRF", "kappa_alpha0"), ("VICE", "vice")]:
            m, s, k = _stats_cond(df, model, cond, pct)
            if not np.isnan(m):
                sig = _signal_frac(m)
                print(f"  {model} {pct:>3d}%: signal={sig:.1f}% (acc={m*100:.2f}%, k={k})")

    fig, ax = create_figure("wide")
    panel_signal_captured(ax, df)
    save_figure(fig, OUTPUT_DIR / "signal_captured.pdf")
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'signal_captured.pdf'}")


if __name__ == "__main__":
    main()
