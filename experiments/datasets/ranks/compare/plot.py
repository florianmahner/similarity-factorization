"""Compare kappa and PCT rank estimation across datasets.

Reads outputs from ../kappa/outputs/ and ../pct/outputs/, produces:
  1. k* overview bar chart (all methods)
  2. Diagnostic grid (coherence, kappa, PCT panels per dataset)
  3. THINGS k* scaling across methods

Usage:
    ./scripts/submit experiments/datasets/ranks/compare/plot.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.figure_theme import (
    CMAP,
    GRAY,
    add_reference_line,
    create_figure,
    despine,
    save_figure,
)

log = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
KAPPA_DIR = TASK_DIR.parent / "kappa" / "outputs"
PCT_DIR = TASK_DIR.parent / "pct" / "outputs"
OUTPUT_DIR = TASK_DIR / "outputs"

METHOD_COLORS = {
    "activation": CMAP[1],
    "kappa": CMAP[0],
    "cluster": CMAP[2],
    "pct": CMAP[3],
}

METHOD_LABELS = {
    "activation": "Activation",
    "kappa": "Kappa",
    "cluster": "Cluster",
    "pct": "PCT",
}

DATASET_LABELS = {
    "mur92": "Mur92",
    "peterson_animals": "Peterson (anim.)",
    "peterson_various": "Peterson (var.)",
    "dinov3": "DINOv3",
    "swow": "SWOW",
    "nsd_subj01": "NSD (s01)",
    "things_100pct": "THINGS",
}

DATASET_ORDER = [
    "mur92",
    "peterson_animals",
    "peterson_various",
    "things_100pct",
    "dinov3",
    "swow",
    "nsd_subj01",
]


def _load_all_json(result_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(result_dir.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def plot_kstar_overview(
    kappa_results: dict[str, dict],
    pct_results: dict[str, dict],
    output_dir: Path,
) -> None:
    datasets = [d for d in DATASET_ORDER if d in kappa_results]
    if not datasets:
        return

    labels = [DATASET_LABELS.get(d, d) for d in datasets]
    methods = ["activation", "kappa", "cluster", "pct"]
    n = len(datasets)
    n_m = len(methods)

    fig, ax = create_figure("full_width", pad_bottom=0.7, pad_left=0.6, pad_top=0.3)
    bar_width = 0.19
    x = np.arange(n)

    all_values = {}
    for method in methods:
        values = []
        for d in datasets:
            if method == "pct":
                val = pct_results.get(d, {}).get("k_star_pct", 0)
            else:
                val = kappa_results[d].get(f"k_star_{method}", 0)
            values.append(val)
        all_values[method] = values

    max_val = max(v for vals in all_values.values() for v in vals)

    for i, method in enumerate(methods):
        values = all_values[method]
        offset = (i - (n_m - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.015,
                    str(val), ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Estimated rank (k*)")
    ax.legend(fontsize=7, ncol=4, loc="upper left", columnspacing=1.0)
    despine(ax)
    save_figure(fig, output_dir / "kstar_overview.pdf")
    plt.close(fig)
    log.info("Saved kstar_overview.pdf")


def _plot_coherence_panel(ax, kappa_dir, name):
    npz = np.load(kappa_dir / f"{name}.npz")
    k_list, p_list = npz["k_list"], npz["p_list"]
    x_median, tau_kp = npz["x_median"], npz["tau_kp"]

    n_k = len(k_list)
    selected = np.arange(n_k) if n_k <= 6 else np.linspace(0, n_k - 1, 6, dtype=int)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(selected)))

    for i, idx in enumerate(selected):
        ax.plot(p_list, x_median[idx], color=cmap[i], linewidth=1.2, label=f"k={k_list[idx]}")
    if tau_kp.size > 0:
        ax.plot(p_list, tau_kp[selected[0]], color=GRAY["medium"], linestyle="--", linewidth=0.8, label="null")
    ax.set_xlabel("Sampling fraction (p)")
    ax.set_ylabel("Coherence (Iproj)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5.5, ncol=2, loc="lower right", handlelength=1.2)
    despine(ax)


def _plot_kappa_panel(ax, kappa_dir, name, k_star):
    npz = np.load(kappa_dir / f"{name}.npz")
    k_list, kappa = npz["k_list"], npz["kappa"]
    ax.plot(k_list[:len(kappa)], kappa, color=CMAP[0], linewidth=1.5)
    ax.axvline(k_star, color=GRAY["medium"], linestyle="--", linewidth=0.8, zorder=0)
    ax.text(k_star + (k_list[-1] - k_list[0]) * 0.02, kappa.max() * 0.92,
            f"k*={k_star}", ha="left", va="top", fontsize=7, color=GRAY["dark"])
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Kappa")
    despine(ax)


def _plot_pct_panel(ax, pct_dir, name, k_star):
    npz = np.load(pct_dir / f"{name}.npz")
    pvalues = npz["pvalues"]
    k_range = np.arange(1, len(pvalues) + 1)
    sig_mask = pvalues < 0.05

    ax.scatter(k_range[sig_mask], pvalues[sig_mask], s=12, color=CMAP[3], zorder=3, label="p < 0.05")
    ax.scatter(k_range[~sig_mask], pvalues[~sig_mask], s=12, color=GRAY["light"], zorder=2, label="n.s.")
    add_reference_line(ax, 0.05, color=CMAP[0], linestyle="--", linewidth=0.8)
    ax.axvline(k_star, color=GRAY["medium"], linestyle="--", linewidth=0.8, zorder=0)
    ax.text(k_star, 0.85, f"k*={k_star}", ha="left", va="top", fontsize=7, color=GRAY["dark"])
    ax.set_xlabel("Dimension (k)")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="center right")
    despine(ax)


def plot_diagnostic_grid(
    kappa_results: dict[str, dict],
    pct_results: dict[str, dict],
    kappa_dir: Path,
    pct_dir: Path,
    output_dir: Path,
) -> None:
    common = [
        d for d in DATASET_ORDER
        if d in kappa_results and d in pct_results
        and not (d.startswith("things_") and d != "things_100pct")
    ]
    if not common:
        return

    n_rows = len(common)
    col_titles = ["Coherence", "Kappa", "PCT"]
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(7.5, 1.9 * n_rows + 0.6),
        gridspec_kw={"hspace": 0.5, "wspace": 0.4},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    for row, name in enumerate(common):
        label = DATASET_LABELS.get(name, name)
        _plot_coherence_panel(axes[row, 0], kappa_dir, name)
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-0.45, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=8, fontweight="bold", ha="right", va="center", rotation=90,
        )
        _plot_kappa_panel(axes[row, 1], kappa_dir, name, kappa_results[name]["k_star_kappa"])
        _plot_pct_panel(axes[row, 2], pct_dir, name, pct_results[name]["k_star_pct"])

    fig.subplots_adjust(left=0.12)
    save_figure(fig, output_dir / "diagnostic_grid.pdf")
    plt.close(fig)
    log.info("Saved diagnostic_grid.pdf")


def plot_things_scaling(
    kappa_results: dict[str, dict],
    pct_results: dict[str, dict],
    output_dir: Path,
) -> None:
    pct_map = {"5pct": 5, "10pct": 10, "20pct": 20, "50pct": 50, "100pct": 100}
    rows = []
    for suffix, pct in pct_map.items():
        name = f"things_{suffix}"
        if name not in kappa_results:
            continue
        kr = kappa_results[name]
        row = {
            "pct": pct,
            "activation": kr.get("k_star_activation", 0),
            "kappa": kr.get("k_star_kappa", 0),
            "cluster": kr.get("k_star_cluster", 0),
        }
        if name in pct_results:
            row["pct_method"] = pct_results[name]["k_star_pct"]
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("pct")
    fig, ax = create_figure("single")

    for method in ["activation", "kappa", "cluster", "pct_method"]:
        if method not in df.columns:
            continue
        key = method if method != "pct_method" else "pct"
        ax.plot(df["pct"], df[method], marker="o", color=METHOD_COLORS[key],
                label=METHOD_LABELS[key], linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_xticks(df["pct"].values)
    ax.set_xticklabels([f"{p}%" for p in df["pct"].values])
    ax.minorticks_off()
    ax.set_xlabel("Triplet data (%)")
    ax.set_ylabel("Estimated rank (k*)")
    ax.legend(fontsize=7, loc="upper left")
    despine(ax)
    save_figure(fig, output_dir / "things_scaling.pdf")
    plt.close(fig)
    log.info("Saved things_scaling.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kappa_results = _load_all_json(KAPPA_DIR) if KAPPA_DIR.exists() else {}
    pct_results = _load_all_json(PCT_DIR) if PCT_DIR.exists() else {}
    log.info(f"Kappa: {len(kappa_results)} datasets, PCT: {len(pct_results)} datasets")

    if not kappa_results:
        log.error("No kappa results found")
        raise SystemExit(1)

    plot_kstar_overview(kappa_results, pct_results, OUTPUT_DIR)
    plot_diagnostic_grid(kappa_results, pct_results, KAPPA_DIR, PCT_DIR, OUTPUT_DIR)
    plot_things_scaling(kappa_results, pct_results, OUTPUT_DIR)
    log.info(f"All plots saved to {OUTPUT_DIR}")
