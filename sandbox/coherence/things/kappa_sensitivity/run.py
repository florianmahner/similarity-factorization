"""Kappa changepoint parameter sensitivity on THINGS behavioral (100%).

Sweeps smooth_window, hi_band_quantile, changepoint_mode, jump_metric
on the already-computed coherence diagnostics. No heavy computation needed.
"""

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from src.coherence.analysis import estimate_kappa, kappa_changepoint
from src.colors import ROSE, TEAL, CYAN, SAND, setup_style
from src.utils import get_output_dir
from src.utils.figure_theme import despine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()

DIAG_PATH = Path(
    "sandbox/coherence/things/outputs/latest/diagnostics/pct100_tau0.95/coherence.npz"
)

SMOOTH_WINDOWS = [1, 3, 5, 7, 11]
HI_BAND_QUANTILES = [0.70, 0.80, 0.85, 0.90, 0.95]
MODES = ["largest_jump", "first_above_quantile"]
METRICS = ["abs", "rel"]


def main():
    d = np.load(DIAG_PATH)
    x_median = d["x_median"]
    p_list = d["p_list"]
    k_list = d["k_list"]

    records = []
    for sw, hbq, mode, metric in product(
        SMOOTH_WINDOWS, HI_BAND_QUANTILES, MODES, METRICS,
    ):
        kappa, _ = estimate_kappa(x_median, p_list, hi_band_quantile=hbq)
        k_cut, _ = kappa_changepoint(
            kappa, k_list,
            smooth_window=sw,
            changepoint_mode=mode,
            jump_metric=metric,
        )
        records.append({
            "smooth_window": sw,
            "hi_band_quantile": hbq,
            "changepoint_mode": mode,
            "jump_metric": metric,
            "k_star": k_cut,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "kappa_sensitivity.csv", index=False)
    log.info(f"Saved {len(df)} combos to {OUTPUT_DIR / 'kappa_sensitivity.csv'}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"k* range: {df['k_star'].min()} -- {df['k_star'].max()}")
    print(f"k* median: {df['k_star'].median():.0f}")
    print(f"k* mode: {df['k_star'].mode().values[0]}")
    print()
    print("By changepoint_mode + jump_metric:")
    for (mode, metric), g in df.groupby(["changepoint_mode", "jump_metric"]):
        print(f"  {mode} / {metric}: median k*={g['k_star'].median():.0f}, "
              f"range=[{g['k_star'].min()}, {g['k_star'].max()}]")
    print("=" * 60)

    _plot_heatmap(df)
    _plot_overview(df)


def _plot_heatmap(df: pd.DataFrame):
    """Heatmap of k* for largest_jump/abs (the default mode)."""
    setup_style()

    sub = df[(df["changepoint_mode"] == "largest_jump") & (df["jump_metric"] == "abs")]
    pivot = sub.pivot(index="smooth_window", columns="hi_band_quantile", values="k_star")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    im = ax.imshow(
        pivot.values, aspect="auto", cmap="YlOrRd",
        vmin=pivot.values.min() - 1, vmax=pivot.values.max() + 1,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("High-band quantile")
    ax.set_ylabel("Smooth window")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = int(pivot.values[i, j])
            color = "white" if val > (pivot.values.max() + pivot.values.min()) / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, aspect=20)
    cb.set_label("$\\hat{k}$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUTPUT_DIR / "kappa_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "kappa_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved heatmap to {OUTPUT_DIR / 'kappa_heatmap.png'}")


def _plot_overview(df: pd.DataFrame):
    """Overview: k* distribution across all parameter combos."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # Left: k* by smooth_window, colored by hi_band_quantile
    ax = axes[0]
    sub = df[(df["changepoint_mode"] == "largest_jump") & (df["jump_metric"] == "abs")]
    for hbq in HI_BAND_QUANTILES:
        g = sub[sub["hi_band_quantile"] == hbq]
        ax.plot(g["smooth_window"], g["k_star"], "o-", markersize=4, linewidth=1,
                label=f"q={hbq:.2f}")
    ax.set_xlabel("Smooth window")
    ax.set_ylabel("$\\hat{k}$")
    ax.legend(fontsize=6, title="hi-band q", title_fontsize=6.5,
              labelspacing=0.2, handletextpad=0.3)
    despine(ax)

    # Right: k* by mode/metric combo (strip)
    ax = axes[1]
    combos = df.groupby(["changepoint_mode", "jump_metric"])["k_star"].agg(list).reset_index()
    rng = np.random.default_rng(0)
    colors = [TEAL, CYAN, SAND, ROSE]
    for i, (_, row) in enumerate(combos.iterrows()):
        label = f"{row['changepoint_mode'][:7]}/{row['jump_metric']}"
        vals = row["k_star"]
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, color=colors[i],
                   alpha=0.6, edgecolors="none")
        ax.scatter(i, np.median(vals), s=40, color=colors[i], marker="D",
                   edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0, label,
                ha="center", va="top", fontsize=5.5, rotation=30)

    ax.set_xticks([])
    ax.set_ylabel("$\\hat{k}$")
    despine(ax)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kappa_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "kappa_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved overview to {OUTPUT_DIR / 'kappa_overview.png'}")


if __name__ == "__main__":
    main()
