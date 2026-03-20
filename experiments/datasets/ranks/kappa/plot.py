"""Plot kappa-based rank estimation results.

Reads JSON/NPZ from outputs/ and produces figures there.

Usage:
    ./scripts/submit experiments/datasets/ranks/kappa/plot.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY, CYCLE
from src.utils.figure_theme import create_figure, despine, save_figure

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

METHOD_COLORS = {
    "activation": TEAL,
    "kappa": ROSE,
    "cluster": CYAN,
}

METHOD_LABELS = {
    "activation": "Activation",
    "kappa": "Kappa changepoint",
    "cluster": "Cluster consensus",
}

DATASET_LABELS = {
    "mur92": "Mur92",
    "peterson_animals": "Peterson (animals)",
    "peterson_various": "Peterson (various)",
    "dinov3": "DINOv3",
    "swow": "SWOW",
    "nsd_subj01": "NSD (subj01)",
    "nsd_subj02": "NSD (subj02)",
    "things_100pct": "THINGS (100%)",
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


def _load_results(result_dir: Path) -> list[dict]:
    records = []
    for json_path in sorted(result_dir.glob("*.json")):
        with open(json_path) as f:
            records.append(json.load(f))
    return records


def _load_npz(result_dir: Path, name: str) -> dict:
    return dict(np.load(result_dir / f"{name}.npz"))


def plot_kstar_comparison(records: list[dict], output_dir: Path) -> None:
    non_split = [
        r for r in records
        if not r["dataset"].startswith("things_") or r["dataset"] == "things_100pct"
    ]
    if not non_split:
        return

    datasets = [r["dataset"] for r in non_split]
    labels = [DATASET_LABELS.get(d, d) for d in datasets]
    methods = ["activation", "kappa", "cluster"]
    n_datasets = len(datasets)

    fig, ax = create_figure("full_width", pad_bottom=1.0, pad_left=0.6)
    bar_width = 0.25
    x = np.arange(n_datasets)

    for i, method in enumerate(methods):
        values = [r[f"k_star_{method}"] for r in non_split]
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Estimated rank (k*)")
    ax.legend(loc="upper left", fontsize=7)
    despine(ax)
    save_figure(fig, output_dir / "kstar_comparison.pdf")
    plt.close(fig)
    log.info("Saved kstar_comparison.pdf")


def plot_things_kstar_vs_data(records: list[dict], output_dir: Path) -> None:
    things_records = [r for r in records if r["dataset"].startswith("things_")]
    if not things_records:
        return

    pct_map = {"5pct": 5, "10pct": 10, "20pct": 20, "50pct": 50, "100pct": 100}
    data = []
    for r in things_records:
        suffix = r["dataset"].replace("things_", "")
        pct = pct_map.get(suffix)
        if pct is not None:
            data.append({"pct": pct, **{m: r[f"k_star_{m}"] for m in ["activation", "kappa", "cluster"]}})

    if not data:
        return

    df = pd.DataFrame(data).sort_values("pct")
    fig, ax = create_figure("single")

    for method in ["activation", "kappa", "cluster"]:
        ax.plot(
            df["pct"], df[method],
            marker="o", color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("Triplet data (%)")
    ax.set_ylabel("Estimated rank (k*)")
    ax.set_xticks(df["pct"].values)
    ax.set_xticklabels([f"{p}%" for p in df["pct"].values])
    ax.legend(fontsize=7)
    despine(ax)
    save_figure(fig, output_dir / "things_kstar_vs_data.pdf")
    plt.close(fig)
    log.info("Saved things_kstar_vs_data.pdf")


def plot_coherence_curves(records: list[dict], result_dir: Path, output_dir: Path) -> None:
    for record in records:
        name = record["dataset"]
        if not (result_dir / f"{name}.npz").exists():
            continue

        arrays = _load_npz(result_dir, name)
        k_list = arrays["k_list"]
        p_list = arrays["p_list"]
        x_median = arrays["x_median"]
        tau_kp = arrays["tau_kp"] if arrays["tau_kp"].size > 0 else None

        n_k = len(k_list)
        selected = np.arange(n_k) if n_k <= 8 else np.linspace(0, n_k - 1, 8, dtype=int)

        fig, ax = create_figure("wide")
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected)))

        for i, idx in enumerate(selected):
            ax.plot(p_list, x_median[idx], color=cmap[i], label=f"k={k_list[idx]}", linewidth=1.2)

        if tau_kp is not None:
            ax.plot(p_list, tau_kp[selected[0]], color=GRAY, linestyle="--", linewidth=0.8, label="null threshold")

        ax.set_xlabel("Sampling fraction (p)")
        ax.set_ylabel("Iproj (median)")
        ax.set_title(DATASET_LABELS.get(name, name), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        despine(ax)
        save_figure(fig, output_dir / f"coherence_{name}.pdf")
        plt.close(fig)

    log.info("Saved coherence curve plots")


def plot_eigenvalue_spectra(records: list[dict], result_dir: Path, output_dir: Path) -> None:
    fig, ax = create_figure("wide")

    for record in records:
        name = record["dataset"]
        if name.startswith("things_") and name != "things_100pct":
            continue
        if not (result_dir / f"{name}.npz").exists():
            continue

        arrays = _load_npz(result_dir, name)
        evals = arrays["evals_ref"]
        evals_norm = evals / evals[0]
        ax.plot(np.arange(1, len(evals) + 1), evals_norm, linewidth=1.2,
                label=DATASET_LABELS.get(name, name))

    ax.set_xlabel("Component index")
    ax.set_ylabel("Normalized eigenvalue")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="upper right")
    despine(ax)
    save_figure(fig, output_dir / "eigenvalue_spectra.pdf")
    plt.close(fig)
    log.info("Saved eigenvalue_spectra.pdf")


def plot_kappa_overlay(records: list[dict], result_dir: Path, output_dir: Path) -> None:
    kappa_results = {r["dataset"]: r for r in records}
    datasets = [
        d for d in DATASET_ORDER
        if d in kappa_results
        and not (d.startswith("things_") and d != "things_100pct")
    ]
    if not datasets:
        return

    fig, ax = create_figure("wide")
    n_ds = len(datasets)
    colors = CYCLE[:n_ds] if n_ds <= len(CYCLE) else plt.cm.tab10(np.linspace(0, 1, n_ds))

    for i, name in enumerate(datasets):
        npz = np.load(result_dir / f"{name}.npz")
        k_list = npz["k_list"]
        kappa = npz["kappa"]
        kmax = kappa.max()
        kappa_norm = kappa / kmax if kmax > 0 else kappa

        ax.plot(k_list[:len(kappa)], kappa_norm, color=colors[i], linewidth=1.3,
                label=DATASET_LABELS.get(name, name))

        k_star = kappa_results[name]["k_star_kappa"]
        idx = np.searchsorted(k_list, k_star)
        if idx < len(kappa_norm):
            ax.scatter(k_star, kappa_norm[idx], color=colors[i], s=30, zorder=5,
                       edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Normalized kappa")
    ax.legend(fontsize=6.5, loc="center right")
    despine(ax)
    save_figure(fig, output_dir / "kappa_overlay.pdf")
    plt.close(fig)
    log.info("Saved kappa_overlay.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = _load_results(OUTPUT_DIR)
    if not records:
        log.error(f"No JSON results in {OUTPUT_DIR}")
        raise SystemExit(1)
    log.info(f"Loaded {len(records)} dataset results")
    plot_kstar_comparison(records, OUTPUT_DIR)
    plot_things_kstar_vs_data(records, OUTPUT_DIR)
    plot_coherence_curves(records, OUTPUT_DIR, OUTPUT_DIR)
    plot_eigenvalue_spectra(records, OUTPUT_DIR, OUTPUT_DIR)
    plot_kappa_overlay(records, OUTPUT_DIR, OUTPUT_DIR)
    log.info(f"All plots saved to {OUTPUT_DIR}")
