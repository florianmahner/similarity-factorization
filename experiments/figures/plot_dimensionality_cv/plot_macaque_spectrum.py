"""Diagnostic spectrum plot for the macaque bandwidth sensitivity.

Usage:
    PYTHONPATH=src .venv/bin/python experiments/figures/plot_dimensionality_cv/plot_macaque_spectrum.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.colors import (
    CYAN,
    GRAY_DARK,
    GRAY_LIGHT,
    PURPLE,
    WINE,
    lighten,
    setup_style,
)
from src.utils.figure_theme import despine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIM_DIR = PROJECT_ROOT / "experiments" / "datasets" / "dimensionality" / "outputs"
CACHE_DIR = DIM_DIR / "cache"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FS = 5

SPECTRUM_DATASETS = [
    ("clip_vit_l14", "CLIP ViT-L/14", PURPLE),
    ("nsd_subj01", "NSD subj01", CYAN),
    ("things_macaque22k", "Macaque IT", WINE),
    ("things_macaque22k_tight", "Macaque IT tight", GRAY_DARK),
]


def _nature_rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS + 0.8,
        "xtick.labelsize": FS - 0.3,
        "ytick.labelsize": FS - 0.3,
        "legend.fontsize": FS - 0.25,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.15,
    }


def _load_estimate(dataset: str) -> dict:
    path = DIM_DIR / dataset / "coherence_estimate.json"
    payload = json.loads(path.read_text())
    estimate = payload["estimate"]
    return {
        "n": int(payload.get("n", estimate.get("n_features_in"))),
        "rank": int(estimate["rank"]),
        "p_star": float(estimate["sampling_fraction"]),
        "floor": float(estimate["detectability_floor"]),
        "eigenvalues": np.asarray(estimate["eigenvalues"], dtype=float),
        "leakage": np.asarray(estimate["leakage"], dtype=float),
    }


def _sample_offdiag(dataset: str, n_samples: int = 300_000) -> np.ndarray:
    matrix = np.load(CACHE_DIR / f"{dataset}.npy", mmap_mode="r")
    rng = np.random.default_rng(12_345)
    n = matrix.shape[0]
    i = rng.integers(0, n, size=n_samples)
    j = rng.integers(0, n, size=n_samples)
    keep = i != j
    return np.asarray(matrix[i[keep], j[keep]], dtype=float)


def _effective_rank(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.sum() ** 2 / np.sum(values ** 2))


def _draw_similarity_hist(ax: plt.Axes) -> None:
    bins = np.linspace(0.0, 0.9, 80)
    series = [
        ("things_macaque22k", "default", WINE),
        ("things_macaque22k_tight", "tight", GRAY_DARK),
    ]
    for dataset, label, color in series:
        values = _sample_offdiag(dataset)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=lighten(color, 0.35),
            edgecolor=color,
            linewidth=0.8,
            alpha=0.52,
            label=label,
        )
    ax.set_title("Macaque kernel values", loc="left", fontweight="normal", pad=3)
    ax.set_xlabel("Off-diagonal similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 0.9)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    despine(ax)


def _draw_spectrum(ax: plt.Axes, estimates: dict[str, dict]) -> None:
    for dataset, label, color in SPECTRUM_DATASETS:
        eig = estimates[dataset]["eigenvalues"]
        x = np.arange(1, len(eig) + 1)
        y = eig / eig[0]
        ax.plot(x, y, color=color, label=label)
        rank = estimates[dataset]["rank"]
        ax.axvline(rank, color=color, linewidth=0.55, alpha=0.45)
    ax.set_title("Normalized eigenspectrum", loc="left", fontweight="normal", pad=3)
    ax.set_xlabel("Component")
    ax.set_ylabel(r"$\lambda_k / \lambda_1$")
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-4, 1.1)
    ax.legend(frameon=False, loc="upper right", handlelength=1.3)
    despine(ax)


def _draw_leakage(ax: plt.Axes, estimates: dict[str, dict]) -> None:
    for dataset, label, color in SPECTRUM_DATASETS:
        leakage = estimates[dataset]["leakage"]
        x = np.arange(1, len(leakage) + 1)
        ax.plot(x, leakage, color=color, label=label)
        rank = estimates[dataset]["rank"]
        ax.axvline(rank, color=color, linewidth=0.55, alpha=0.45)
        if dataset.startswith("things_macaque"):
            y = float(np.interp(rank, x, leakage))
            ax.text(
                rank + 2,
                y * 1.07,
                f"$k$={rank}",
                color=color,
                fontsize=FS - 0.35,
                va="bottom",
            )
    ax.set_title("Coherence leakage", loc="left", fontweight="normal", pad=3)
    ax.set_xlabel("Component")
    ax.set_ylabel("Leakage")
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.015, 20)
    despine(ax)


def _draw_summary(ax: plt.Axes, estimates: dict[str, dict]) -> None:
    ax.axis("off")
    rows = []
    for dataset, label, _ in SPECTRUM_DATASETS:
        est = estimates[dataset]
        eig = est["eigenvalues"]
        top1 = eig[0] / eig.sum()
        top10 = eig[:10].sum() / eig.sum()
        rows.append(
            f"{label}: k={est['rank']}, p*={est['p_star']:.2f}, "
            f"ER200={_effective_rank(eig):.1f}, top1={top1:.2f}, top10={top10:.2f}"
        )
    ax.text(
        0.0,
        0.94,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS - 0.15,
        color=GRAY_DARK,
        linespacing=1.35,
    )
    ax.text(
        0.0,
        0.18,
        "Tighter bandwidth shifts macaque from a broad, high-baseline kernel to a local kernel.\n"
        "That flattens the top spectrum and delays the leakage changepoint.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS - 0.15,
        color=GRAY_DARK,
        linespacing=1.25,
    )


def main() -> None:
    setup_style()
    plt.rcParams.update(_nature_rc())
    estimates = {dataset: _load_estimate(dataset) for dataset, _, _ in SPECTRUM_DATASETS}

    fig = plt.figure(figsize=(FIG_WIDTH_IN, 4.35))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.56])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, :]),
    ]

    _draw_similarity_hist(axes[0])
    _draw_spectrum(axes[1], estimates)
    _draw_leakage(axes[2], estimates)
    _draw_summary(axes[3], estimates)

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.08, top=0.92, wspace=0.38, hspace=0.22)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "macaque_spectrum_diagnostic.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
