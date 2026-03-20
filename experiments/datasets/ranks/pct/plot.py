"""Plot PCT rank estimation results.

Reads JSON/NPZ from outputs/ and produces figures there.

Usage:
    ./scripts/submit experiments/datasets/ranks/pct/plot.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.figure_theme import CMAP, GRAY, add_reference_line, create_figure, despine, save_figure

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DATASET_LABELS = {
    "mur92": "Mur92",
    "peterson_animals": "Peterson (animals)",
    "peterson_various": "Peterson (various)",
    "dinov3": "DINOv3",
    "swow": "SWOW",
    "nsd_subj01": "NSD (subj01)",
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


def _load_all_json(result_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(result_dir.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def plot_pct_overlay(results: dict[str, dict], result_dir: Path, output_dir: Path) -> None:
    datasets = [
        d for d in DATASET_ORDER
        if d in results
        and not (d.startswith("things_") and d != "things_100pct")
    ]
    if not datasets:
        return

    fig, ax = create_figure("wide")
    n_ds = len(datasets)
    colors = CMAP[:n_ds] if n_ds <= len(CMAP) else plt.cm.tab10(np.linspace(0, 1, n_ds))

    for i, name in enumerate(datasets):
        npz = np.load(result_dir / f"{name}.npz")
        pvalues = npz["pvalues"]
        k_range = np.arange(1, len(pvalues) + 1)
        ax.plot(k_range, pvalues, color=colors[i], linewidth=1.2,
                label=DATASET_LABELS.get(name, name), alpha=0.85)

    add_reference_line(ax, 0.05, color=GRAY["dark"], linestyle="--", linewidth=0.8)
    ax.text(1, 0.06, r"$\alpha = 0.05$", fontsize=7, color=GRAY["dark"], va="bottom")
    ax.set_xlabel("Dimension (k)")
    ax.set_ylabel("p-value")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6.5, loc="upper left")
    despine(ax)
    save_figure(fig, output_dir / "pct_overlay.pdf")
    plt.close(fig)
    log.info("Saved pct_overlay.pdf")


def plot_pct_per_dataset(results: dict[str, dict], result_dir: Path, output_dir: Path) -> None:
    for name, record in results.items():
        npz_path = result_dir / f"{name}.npz"
        if not npz_path.exists():
            continue

        npz = np.load(npz_path)
        pvalues = npz["pvalues"]
        k_range = np.arange(1, len(pvalues) + 1)
        k_star = record["k_star_pct"]

        sig_mask = pvalues < 0.05
        nonsig_mask = ~sig_mask

        fig, ax = create_figure("single")
        ax.scatter(k_range[sig_mask], pvalues[sig_mask], s=12, color=CMAP[3], zorder=3, label="p < 0.05")
        ax.scatter(k_range[nonsig_mask], pvalues[nonsig_mask], s=12, color=GRAY["light"], zorder=2, label="n.s.")
        add_reference_line(ax, 0.05, color=CMAP[0], linestyle="--", linewidth=0.8)
        ax.axvline(k_star, color=GRAY["medium"], linestyle="--", linewidth=0.8, zorder=0)
        ax.text(k_star, 0.85, f"k*={k_star}", ha="left", va="top", fontsize=7, color=GRAY["dark"])
        ax.set_xlabel("Dimension (k)")
        ax.set_ylabel("p-value")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(DATASET_LABELS.get(name, name), fontsize=10)
        ax.legend(fontsize=6, loc="center right")
        despine(ax)
        save_figure(fig, output_dir / f"pct_{name}.pdf")
        plt.close(fig)

    log.info("Saved per-dataset PCT plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = _load_all_json(OUTPUT_DIR)
    if not results:
        log.error(f"No JSON results in {OUTPUT_DIR}")
        raise SystemExit(1)
    log.info(f"Loaded {len(results)} dataset results")
    plot_pct_overlay(results, OUTPUT_DIR, OUTPUT_DIR)
    plot_pct_per_dataset(results, OUTPUT_DIR, OUTPUT_DIR)
    log.info(f"All plots saved to {OUTPUT_DIR}")
