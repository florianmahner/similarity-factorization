"""One overview figure across all datasets.

Reads every outputs/<dataset>/{coherence_estimate.json, cross_validation.json} and produces a
single PDF with one row per dataset (spectrum | CV curve). Matches the
sandbox spectrum_vs_cv style but uses the project palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.colors import GRAY, GRAY_DARK, ROSE, TEAL, setup_style, soft

from . import io as _io

setup_style()
sns.set_theme(
    style="ticks", context="paper",
    rc={
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)

SPECTRUM_COLOR = TEAL
CV_COLOR = ROSE
KCUT_COLOR = GRAY_DARK


def save_overview(output_dir: Path, dataset_names: list[str], path: Path) -> None:
    """Build one figure: rows = datasets, columns = (spectrum, CV curve)."""
    rows = []
    for name in dataset_names:
        try:
            coherence = _io.read_coherence(name, output_dir)
        except FileNotFoundError:
            continue
        cv_payload = _io.read_cross_validation(name, output_dir)
        cv = _cv_from_payload(cv_payload)
        rows.append((name, coherence, cv_payload, cv))
    if not rows:
        return

    n = len(rows)
    fig, axes = plt.subplots(
        n, 2, figsize=(9.5, 2.6 * n), squeeze=False,
        gridspec_kw=dict(wspace=0.32, hspace=0.55, width_ratios=[1.0, 1.2]),
    )

    for r, (name, payload, cv_payload, cv) in enumerate(rows):
        _plot_spectrum_row(axes[r, 0], payload, name)
        _plot_cv_row(axes[r, 1], payload, cv_payload, cv)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def _plot_spectrum_row(ax, payload, name, max_shown=30):
    estimate = payload.get("estimate", {})
    eigvals = np.asarray(estimate.get("eigenvalues", []), dtype=float)
    if eigvals.size == 0:
        return
    k_cut = int(estimate["rank"])
    p_star = float(estimate["sampling_fraction"])
    n_objects = int(payload.get("n", 0))

    n_shown = int(min(max_shown, eigvals.size))
    x = np.arange(1, n_shown + 1)
    y = eigvals[:n_shown]
    ax.plot(x, y, "-", color=SPECTRUM_COLOR, lw=1.6, alpha=0.95)
    ax.plot(x, y, "o", color=SPECTRUM_COLOR, ms=3.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.axvline(k_cut, color=KCUT_COLOR, lw=0.9, ls=(0, (4, 3)), alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlim(0, n_shown + 1)
    ax.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Eigenvalue")

    # Row label on the LEFT — dataset name + n + p*
    ax.text(
        -0.34, 0.5, f"{name}\nn = {n_objects}\n$p^* = {p_star:.2f}$",
        transform=ax.transAxes, ha="right", va="center",
        fontsize=10, linespacing=1.6,
    )
    # Column titles on the top row
    if ax.get_subplotspec().rowspan.start == 0:
        ax.set_title("Spectrum of $S$", loc="left", fontweight="bold", pad=10)
    # X-label on the bottom row
    if ax.get_subplotspec().rowspan.start == ax.get_subplotspec().get_gridspec().nrows - 1:
        ax.set_xlabel("Eigenvalue index $r$")

    ax.text(
        0.97, 0.94, f"$k_{{\\rm cut}}={k_cut}$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color=KCUT_COLOR, fontweight="bold",
    )
    sns.despine(ax=ax)


def _plot_cv_row(ax, payload, cv_payload, cv):
    estimate = payload.get("estimate", {})
    validate = _primary_validation(cv_payload)
    k_cut = int(estimate.get("rank", 0))

    if validate is None or cv is None:
        ax.text(0.5, 0.5, "no CV result", transform=ax.transAxes,
                ha="center", va="center", color=GRAY, fontsize=10, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        return

    # Compute mean and SEM directly from the CV CSV so the band reflects
    # the actual per-(rep, fold) cells.
    g = cv.groupby("rank")["val_mse"].agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(np.maximum(g["count"], 1))
    g = g.sort_values("rank")
    ranks = g["rank"].to_numpy()
    mean = g["mean"].to_numpy()
    sem = g["sem"].to_numpy()
    argmin_rank = int(validate["argmin_rank"])
    i_min = int(np.argmin(mean))

    # SEM band — clip floor so log scale doesn't blow up
    lower = np.maximum(mean - sem, np.nanmin(mean) * 1e-3)
    upper = mean + sem
    ax.fill_between(ranks, lower, upper, color=soft(CV_COLOR), alpha=0.55,
                    linewidth=0, zorder=1)
    ax.plot(ranks, mean, "-", color=CV_COLOR, lw=1.8, zorder=2)
    ax.scatter([ranks[i_min]], [mean[i_min]], marker="*", s=160, color=CV_COLOR,
               edgecolors="white", linewidths=1.3, zorder=10)
    ax.scatter([ranks[i_min]], [mean[i_min]], marker="*", s=160,
               facecolor="none", edgecolors=KCUT_COLOR, linewidths=0.5, zorder=11)
    ax.axvline(k_cut, color=KCUT_COLOR, lw=0.9, ls=(0, (4, 3)), alpha=0.75)

    ax.set_yscale("log")
    ax.set_xlim(0, max(ranks[-1] + 1, k_cut + 5))
    ax.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Validation MSE")

    if ax.get_subplotspec().rowspan.start == 0:
        ax.set_title("Cross-validation at $p^*$", loc="left",
                     fontweight="bold", pad=10)
    if ax.get_subplotspec().rowspan.start == ax.get_subplotspec().get_gridspec().nrows - 1:
        ax.set_xlabel("Rank $r$")

    ax.text(
        0.97, 0.94, f"$k_{{\\rm cut}}={k_cut}$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color=KCUT_COLOR, fontweight="bold",
    )
    ax.text(
        0.97, 0.84, f"argmin $={argmin_rank}$",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color=CV_COLOR, fontweight="bold",
    )
    sns.despine(ax=ax)


def _primary_validation(payload: dict) -> dict | None:
    validations = payload.get("validations")
    if isinstance(validations, dict):
        if "5fold" in validations:
            return validations["5fold"]
        if validations:
            return next(iter(validations.values()))
    return payload.get("validate")


def _cv_from_payload(payload: dict) -> pd.DataFrame | None:
    validate = _primary_validation(payload)
    if not validate:
        return None
    rows = []
    for rank_text, matrix in validate.get("scores", {}).items():
        rank = int(rank_text)
        for rep, fold_values in enumerate(matrix):
            for fold, value in enumerate(fold_values):
                if value is not None and np.isfinite(value):
                    rows.append({
                        "rep": rep,
                        "fold": fold,
                        "rank": rank,
                        "val_mse": float(value),
                    })
    return pd.DataFrame(rows) if rows else None
