"""Plot coherence-based rank estimation results.

Reads JSON and NPZ outputs from kappa.py and produces:
  1. Bar chart comparing k* across datasets and methods
  2. THINGS k* vs triplet percentage (dimensionality vs data)
  3. Coherence curves (Iproj vs p) per dataset

Usage:
    ./scripts/submit experiments/coherence/plot_rank.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.colors import ROSE, TEAL, CYAN, GRAY
from src.utils.figure_theme import (
    create_figure,
    despine,
    save_figure,
)

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_results(result_dir: Path) -> list[dict]:
    """Load all JSON result files from the output directory."""
    records = []
    for json_path in sorted(result_dir.glob("*.json")):
        with open(json_path) as f:
            records.append(json.load(f))
    return records


def _load_npz(result_dir: Path, name: str) -> dict:
    """Load NPZ arrays for a dataset."""
    npz_path = result_dir / f"{name}.npz"
    return dict(np.load(npz_path))


# ---------------------------------------------------------------------------
# Plot 1: k* comparison across datasets
# ---------------------------------------------------------------------------


def plot_kstar_comparison(records: list[dict], output_dir: Path) -> None:
    """Bar chart of k* across datasets, grouped by method."""
    # Filter to non-THINGS-split datasets
    non_split = [r for r in records if not r["dataset"].startswith("things_") or r["dataset"] == "things_100pct"]
    if not non_split:
        log.warning("No non-split datasets found, skipping k* comparison plot")
        return

    datasets = [r["dataset"] for r in non_split]
    labels = [DATASET_LABELS.get(d, d) for d in datasets]
    methods = ["activation", "kappa", "cluster"]
    n_datasets = len(datasets)
    n_methods = len(methods)

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
            edgecolor="white",
            linewidth=0.5,
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
    log.info(f"Saved kstar_comparison.pdf")


# ---------------------------------------------------------------------------
# Plot 2: THINGS k* vs triplet percentage
# ---------------------------------------------------------------------------


def plot_things_kstar_vs_data(records: list[dict], output_dir: Path) -> None:
    """Line plot showing how k* increases with triplet percentage."""
    things_records = [r for r in records if r["dataset"].startswith("things_")]
    if not things_records:
        log.warning("No THINGS split records found, skipping")
        return

    # Extract percentage from name
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
    log.info(f"Saved things_kstar_vs_data.pdf")


# ---------------------------------------------------------------------------
# Plot 3: Coherence curves (Iproj vs p) per dataset
# ---------------------------------------------------------------------------


def plot_coherence_curves(records: list[dict], result_dir: Path, output_dir: Path) -> None:
    """For each dataset, plot median Iproj(p) for selected k values."""
    for record in records:
        name = record["dataset"]
        npz_path = result_dir / f"{name}.npz"
        if not npz_path.exists():
            continue

        arrays = _load_npz(result_dir, name)
        k_list = arrays["k_list"]
        p_list = arrays["p_list"]
        x_median = arrays["x_median"]
        tau_kp = arrays["tau_kp"] if arrays["tau_kp"].size > 0 else None

        # Select ~8 evenly spaced k values for readability
        n_k = len(k_list)
        if n_k <= 8:
            selected = np.arange(n_k)
        else:
            selected = np.linspace(0, n_k - 1, 8, dtype=int)

        fig, ax = create_figure("wide")
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected)))

        for i, idx in enumerate(selected):
            k = k_list[idx]
            ax.plot(p_list, x_median[idx], color=cmap[i], label=f"k={k}", linewidth=1.2)

        # Overlay null threshold for first selected k if available
        if tau_kp is not None:
            ax.plot(
                p_list, tau_kp[selected[0]],
                color=GRAY, linestyle="--", linewidth=0.8,
                label="null threshold",
            )

        ax.set_xlabel("Sampling fraction (p)")
        ax.set_ylabel("Iproj (median)")
        ax.set_title(DATASET_LABELS.get(name, name), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        despine(ax)

        save_figure(fig, output_dir / f"coherence_{name}.pdf")

    log.info(f"Saved coherence curve plots")


# ---------------------------------------------------------------------------
# Plot 4: Eigenvalue spectra
# ---------------------------------------------------------------------------


def plot_eigenvalue_spectra(records: list[dict], result_dir: Path, output_dir: Path) -> None:
    """Plot reference eigenvalue spectra for all datasets on one figure."""
    fig, ax = create_figure("wide")

    for i, record in enumerate(records):
        name = record["dataset"]
        if name.startswith("things_") and name != "things_100pct":
            continue

        npz_path = result_dir / f"{name}.npz"
        if not npz_path.exists():
            continue

        arrays = _load_npz(result_dir, name)
        evals = arrays["evals_ref"]
        # Normalize eigenvalues to show relative spectrum
        evals_norm = evals / evals[0]
        label = DATASET_LABELS.get(name, name)
        ax.plot(np.arange(1, len(evals) + 1), evals_norm, linewidth=1.2, label=label)

    ax.set_xlabel("Component index")
    ax.set_ylabel("Normalized eigenvalue")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="upper right")
    despine(ax)

    save_figure(fig, output_dir / "eigenvalue_spectra.pdf")
    log.info(f"Saved eigenvalue_spectra.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    result_dir = Path(cfg.project_root) / "outputs" / "experiments" / "coherence" / "kappa"
    output_dir = Path.cwd()

    if not result_dir.exists():
        raise FileNotFoundError(f"No results at {result_dir}. Run kappa.py first.")

    records = _load_results(result_dir)
    if not records:
        raise FileNotFoundError(f"No JSON result files in {result_dir}")

    log.info(f"Loaded {len(records)} dataset results")

    plot_kstar_comparison(records, output_dir)
    plot_things_kstar_vs_data(records, output_dir)
    plot_coherence_curves(records, result_dir, output_dir)
    plot_eigenvalue_spectra(records, result_dir, output_dir)

    log.info(f"All plots saved to {output_dir}")
