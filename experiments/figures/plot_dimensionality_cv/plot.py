"""Appendix-style overview of dimensionality cross-validation curves.

The plot uses the current dimensionality CV JSON files directly. Validation
MSE is normalized within each dataset by its minimum so curves with different
absolute scales can be compared as rank-selection profiles.

Usage:
    PYTHONPATH=src .venv/bin/python experiments/figures/plot_dimensionality_cv/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.colors import (
    CYAN,
    GRAY,
    GRAY_DARK,
    GRAY_LIGHT,
    GRAY_PALE,
    GREEN,
    INDIGO,
    PURPLE,
    ROSE,
    SAND,
    TEAL,
    WINE,
    lighten,
    setup_style,
)
from src.utils.figure_theme import despine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CV_DIR = PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"
OUTPUT = Path(__file__).resolve().parent / "outputs"

DATASETS = [
    ("mur92", "Mur-92", ROSE),
    ("peterson_animals", "Peterson animals", TEAL),
    ("peterson_various", "Peterson various", INDIGO),
    ("things_behavior", "THINGS behavior", SAND),
    ("clip_vit_l14", "CLIP ViT-L/14", PURPLE),
    ("swow", "SWOW", GREEN),
    ("nsd_subj01", "NSD subj01", CYAN),
    ("things_macaque22k", "Macaque IT", WINE),
]

ANNOTATION_POS = {
    "mur92": (0.08, 0.90, "left", "top"),
    "peterson_animals": (0.68, 0.90, "left", "top"),
    "peterson_various": (0.16, 0.90, "left", "top"),
    "things_behavior": (0.68, 0.90, "left", "top"),
    "clip_vit_l14": (0.58, 0.90, "left", "top"),
    "swow": (0.38, 0.90, "left", "top"),
    "nsd_subj01": (0.62, 0.90, "left", "top"),
    "things_macaque22k": (0.55, 0.90, "left", "top"),
}

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FS = 5


def _nature_rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS + 0.8,
        "xtick.labelsize": FS - 0.3,
        "ytick.labelsize": FS - 0.3,
        "legend.fontsize": FS,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.15,
    }


def _rank_spread(block: dict, rank: float) -> float:
    """Per-rank std across all (fold, repeat) fits.

    Prefers raw per-fold scores when available; falls back to ``sem * sqrt(count)``
    for blocks (e.g. batched longer-training runs) that only stored summary stats.
    """
    rank_key = str(int(rank))
    scores = block.get("scores", {}).get(rank_key)
    if scores:
        return float(np.std(np.asarray(scores, dtype=float).ravel(), ddof=1))
    ranks = list(block.get("ranks", block.get("completed_ranks", [])))
    idx = ranks.index(int(rank))
    sem = float(block["val_mse_sem"][idx])
    count = int(block["val_mse_count"][idx])
    return sem * np.sqrt(count)


def _load_curve(dataset: str, variant: str = "5fold") -> dict:
    path = CV_DIR / dataset / "cross_validation.json"
    payload = json.loads(path.read_text())
    block = payload["validations"][variant]
    ranks = np.array(block.get("ranks", block["completed_ranks"]), dtype=float)
    mean = np.array(block["val_mse_mean"], dtype=float)
    spread = np.array([_rank_spread(block, rank) for rank in ranks])
    order = np.argsort(ranks)
    return {
        "dataset": dataset,
        "n": int(payload["n"]),
        "p_star": float(payload["sampling_fraction"]),
        "rank0": int(payload["rank_estimate"]),
        "rank_star": int(block["argmin_rank"]),
        "rank_one_se": int(block["one_se_rank"]),
        "status": block.get("status", ""),
        "ranks": ranks[order],
        "mean": mean[order],
        "spread": spread[order],
    }


def _relative_curve(curve: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_min = np.nanmin(curve["mean"])
    rel = curve["mean"] / y_min
    rel_spread = curve["spread"] / y_min
    return curve["ranks"], rel, rel_spread


def _yticks(y_max: float) -> list[float]:
    if y_max <= 1.25:
        ticks = [1.0, 1.05, 1.1, 1.2]
    elif y_max <= 2.5:
        ticks = [1.0, 1.1, 1.25, 1.5, 2.0]
    elif y_max <= 6:
        ticks = [1.0, 1.5, 2.0, 3.0, 5.0]
    else:
        ticks = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    return [t for t in ticks if t <= y_max * 1.01]


def _format_tick(value: float) -> str:
    if value < 1.1:
        return f"{value:.2f}"
    if value < 2:
        return f"{value:.1f}"
    return f"{value:g}"


def _smooth_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 4:
        return x, y
    x_dense = np.linspace(float(x.min()), float(x.max()), 320)
    y_dense = PchipInterpolator(x, y)(x_dense)
    return x_dense, y_dense


def _rank_ticks(x: np.ndarray, rank_star: int, rank_one_se: int, rank0: int) -> list[int]:
    return sorted({int(x.min()), int(rank_star), int(x.max())})


def _draw_panel(ax: plt.Axes, curve: dict, label: str, color: str) -> None:
    x, y, spread = _relative_curve(curve)
    y_lo = np.maximum(y - spread, 1e-6)
    y_hi = y + spread
    x_s, y_s = _smooth_curve(x, y)
    _, lo_s = _smooth_curve(x, y_lo)
    _, hi_s = _smooth_curve(x, y_hi)

    ax.fill_between(x_s, lo_s, hi_s, color=lighten(color, 0.30), alpha=0.42, linewidth=0)
    ax.plot(x_s, y_s, color=color, linewidth=1.35, solid_capstyle="round")
    ax.axhline(1.0, color=GRAY_LIGHT, linewidth=0.55, zorder=0)

    rank_star = curve["rank_star"]
    rank_one_se = curve["rank_one_se"]
    rank0 = curve["rank0"]
    ax.axvline(rank_star, color=color, linewidth=0.9, zorder=1)

    y_max = float(np.nanmax(y_hi) * 1.18)
    y_max = max(y_max, 1.12)
    ax.set_yscale("log")
    ax.set_ylim(0.985, y_max)
    ticks = _yticks(y_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_format_tick(t) for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    pad = max((x.max() - x.min()) * 0.04, 1.0)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_xticks(_rank_ticks(x, rank_star, rank_one_se, rank0))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", labelrotation=35, labelsize=FS - 0.8)

    # Rank-sampling rug: makes adaptive/uneven grids explicit without point markers.
    rug_top = 0.995 + (y_max - 0.985) * 0.006
    ax.vlines(x, 0.985, rug_top, color=GRAY_LIGHT, linewidth=0.45, alpha=0.9)

    ax.set_title(label, loc="left", pad=3, fontweight="normal")
    _annotate_rank(ax, curve)
    despine(ax)


def _annotate_rank(ax: plt.Axes, curve: dict) -> None:
    rank_star = curve["rank_star"]
    x, y, ha, va = ANNOTATION_POS[curve["dataset"]]

    ax.text(
        x,
        y,
        f"$k^*$={rank_star}\n$p^*$={curve['p_star']:.2f}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=FS - 0.35,
        color=GRAY_DARK,
        linespacing=1.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        zorder=4,
    )


def main() -> None:
    setup_style()
    plt.rcParams.update(_nature_rc())

    curves = [(label, color, _load_curve(dataset)) for dataset, label, color in DATASETS]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(FIG_WIDTH_IN, 4.55),
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()
    for ax, (label, color, curve) in zip(axes[:len(curves)], curves, strict=True):
        _draw_panel(ax, curve, label, color)
    for ax in axes[len(curves):]:
        ax.set_axis_off()

    for row in range(2):
        axes[row * 4].set_ylabel("Relative validation MSE")
    for ax in axes[4:]:
        ax.set_xlabel("Rank")

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.91, wspace=0.32, hspace=0.34)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "dimensionality_cv_overview.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
