"""Embedding overview: individual panels for manual composition.

Saves each panel as a separate file:
  rsm_{dataset}.png          -- observed vs SRF heatmap pair (cluster-sorted)
  bar_rank.png               -- estimated k* across datasets
  bar_reliability.png        -- mean reliability across datasets
  reliability_profiles.png   -- per-dimension reliability (sorted, one line per dataset)

Usage:
    poetry run python experiments/sandbox/figures/embedding_overview/run.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

from src.colors import (
    ROSE, TEAL, INDIGO, SAND, PURPLE, CYAN,
    GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE,
    CMAP_SEQ, soft, lighten,
)
from src.utils.figure_theme import create_figure, despine, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONSENSUS_DIR = PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs"
KAPPA_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ranks" / "kappa" / "outputs"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
ROW_HEIGHT_MM = 35
ROW_HEIGHT_IN = ROW_HEIGHT_MM / 25.4
FONT_SIZE = 5

DATASETS = [
    {"label": "Mur et al.", "consensus": "mur92", "kappa": "mur92", "subject_id": None,
     "color": ROSE},
    {"label": "Peterson (Animals)", "consensus": "peterson-animals", "kappa": "peterson_animals",
     "subject_id": None, "color": TEAL},
    {"label": "Peterson (Various)", "consensus": "peterson-various", "kappa": "peterson_various",
     "subject_id": None, "color": INDIGO},
    {"label": "VGG-16", "consensus": "vgg16", "kappa": "vgg16", "subject_id": None,
     "color": SAND},
    {"label": "NSD (subj01)", "consensus": "nsd", "kappa": "nsd_subj01", "subject_id": 1,
     "color": PURPLE},
]


def _nature_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


# -- Data loading --

def _consensus_dir(ds):
    d = CONSENSUS_DIR / ds["consensus"]
    if ds["subject_id"] is not None:
        d = d / f"subj{ds['subject_id']:02d}"
    return d


def _load_summary(ds):
    return json.loads((_consensus_dir(ds) / "summary.json").read_text())


def _load_embedding(ds):
    return np.load(_consensus_dir(ds) / "embedding.npy")


def _load_kappa(ds):
    return json.loads((KAPPA_DIR / f"{ds['kappa']}.json").read_text())


def _load_similarity(ds):
    """Load similarity matrix (cached after first compute)."""
    cache_dir = OUTPUT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = ds["consensus"]
    if ds["subject_id"] is not None:
        cache_key += f"_subj{ds['subject_id']:02d}"
    cache_path = cache_dir / f"sim_{cache_key}.npy"

    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    from datasets import load_dataset
    from tools.rsa import compute_similarity

    name = ds["consensus"]
    if name in ("mur92", "peterson-animals", "peterson-various"):
        root = "/SSD/datasets/similarity_datasets/"
        if name == "mur92":
            root += "mur92"
        else:
            root += "peterson"
        sim = load_dataset(name, root=root).rsm
    elif name == "vgg16":
        feat_path = PROJECT_ROOT / "data" / "features" / "vgg16" / "vgg16_features.npy"
        features = np.load(feat_path)
        sim = compute_similarity(features, features, "linear")
        sim = sim / sim.max()
    elif name == "nsd":
        result = load_dataset(
            "nsd", subject_id=ds["subject_id"], roi_name="streams",
            root="/LOCAL/LABSHARE/natural-scenes-dataset",
            space="func1pt8mm", zscore_betas=True,
        )
        sim = compute_similarity(result.data, result.data, "gaussian_kernel")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    np.save(cache_path, sim)
    return sim


def _compute_reliability(runs):
    """Per-dimension reliability via Fisher-z averaged pairwise correlations."""
    n_runs, _, rank = runs.shape
    reliability = np.zeros(rank)
    for d in range(rank):
        r_pairs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                r = np.corrcoef(runs[i, :, d], runs[j, :, d])[0, 1]
                if np.isfinite(r):
                    r_pairs.append(r)
        if r_pairs:
            z = np.arctanh(np.clip(r_pairs, -0.999, 0.999))
            reliability[d] = float(np.tanh(np.mean(z)))
    return reliability


def _get_reliability(ds):
    """Load existing reliability or compute from runs."""
    path = _consensus_dir(ds) / "reliability.npy"
    if path.exists():
        return np.load(path)
    print(f"  Computing reliability for {ds['consensus']}...")
    runs = np.load(_consensus_dir(ds) / "runs.npy")
    rel = _compute_reliability(runs)
    np.save(path, rel)
    return rel


def _cluster_order(rsm):
    """Hierarchical clustering leaf order for RSM visualization."""
    n = rsm.shape[0]
    dist = 1.0 - rsm / (rsm.max() + 1e-10)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = dist[np.triu_indices(n, k=1)]
    z = linkage(condensed, method="average")
    return leaves_list(z)


def _save(fig, name):
    out_path = OUTPUT / f"{name}.png"
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -- Individual panels --

def plot_rsm_pair(ds, metrics):
    """Single dataset: observed vs SRF RSM side by side."""
    w = _load_embedding(ds)
    predicted = w @ w.T

    try:
        true_rsm = np.array(_load_similarity(ds))
    except Exception as e:
        print(f"  Could not load similarity for {ds['consensus']}: {e}")
        true_rsm = predicted.copy()

    order = _cluster_order(true_rsm)
    true_sorted = true_rsm[np.ix_(order, order)]
    pred_sorted = predicted[np.ix_(order, order)]
    np.fill_diagonal(true_sorted, 0)
    np.fill_diagonal(pred_sorted, 0)

    vmax = np.percentile(true_sorted, 98)

    panel_w = 1.4
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2 + 0.15, panel_w + 0.35))
    fig.subplots_adjust(wspace=0.08, top=0.82, bottom=0.02, left=0.02, right=0.98)

    for ax, data, title in [(axes[0], true_sorted, "Observed"),
                             (axes[1], pred_sorted, "SRF")]:
        ax.imshow(data, cmap=CMAP_SEQ, vmin=0, vmax=vmax, aspect="equal",
                  interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color(GRAY_LIGHT)
        ax.set_title(title, fontsize=FONT_SIZE, pad=3)

    r2 = metrics["r2"]
    fig.suptitle(f"{ds['label']}  ($R^2 = {r2:.2f}$)",
                 fontsize=FONT_SIZE + 1, fontweight="bold", y=0.97)

    name = ds["consensus"]
    if ds["subject_id"] is not None:
        name += f"_subj{ds['subject_id']:02d}"
    _save(fig, f"rsm_{name}")


def plot_bar_rank(metrics_list):
    """Bar chart of estimated k* across datasets."""
    fig, ax = create_figure("single")

    labels = [m["label"] for m in metrics_list]
    k_vals = [m["k_star"] for m in metrics_list]
    colors = [m["color"] for m in metrics_list]
    x = np.arange(len(labels))
    bar_width = 0.6

    bars = ax.bar(
        x, k_vals, width=bar_width,
        color=[soft(c) for c in colors],
        edgecolor=colors, linewidth=1.2,
    )

    for bar, k in zip(bars, k_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            str(k), ha="center", va="bottom",
            fontsize=FONT_SIZE, fontweight="bold", color=GRAY_DARK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 0.5, rotation=30, ha="right")
    ax.set_ylabel("Estimated rank ($k^*$)")
    ax.set_ylim(0, max(k_vals) * 1.2)
    despine(ax)

    _save(fig, "bar_rank")


def plot_bar_reliability(metrics_list):
    """Bar chart of mean reliability across datasets."""
    fig, ax = create_figure("single")

    labels = [m["label"] for m in metrics_list]
    rel_vals = [m["mean_reliability"] for m in metrics_list]
    colors = [m["color"] for m in metrics_list]
    x = np.arange(len(labels))
    bar_width = 0.6

    bars = ax.bar(
        x, rel_vals, width=bar_width,
        color=[soft(c) for c in colors],
        edgecolor=colors, linewidth=1.2,
    )

    for bar, r in zip(bars, rel_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{r:.2f}", ha="center", va="bottom",
            fontsize=FONT_SIZE, color=GRAY_DARK,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linewidth=0.4, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 0.5, rotation=30, ha="right")
    ax.set_ylabel("Mean reliability")
    ax.set_ylim(0, 1.1)
    despine(ax)

    _save(fig, "bar_reliability")


def plot_reliability_profiles(all_reliabilities):
    """Per-dimension reliability curves, sorted descending, one line per dataset."""
    fig, ax = create_figure("wide")

    for ds, rel in all_reliabilities:
        rel_sorted = np.sort(rel)[::-1]
        frac = np.linspace(0, 1, len(rel_sorted))
        ax.plot(
            frac, rel_sorted,
            color=ds["color"], linewidth=1.0,
            label=ds["label"],
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linewidth=0.4, linestyle="--", zorder=0)
    ax.set_xlabel("Dimension (fraction of $k^*$)")
    ax.set_ylabel("Reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=FONT_SIZE - 0.5, loc="lower left", frameon=False,
        handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
    )
    despine(ax)

    _save(fig, "reliability_profiles")


def plot_bar_r2(metrics_list):
    """Bar chart of reconstruction R² across datasets."""
    fig, ax = create_figure("single")

    labels = [m["label"] for m in metrics_list]
    r2_vals = [m["r2"] for m in metrics_list]
    colors = [m["color"] for m in metrics_list]
    x = np.arange(len(labels))
    bar_width = 0.6

    bars = ax.bar(
        x, r2_vals, width=bar_width,
        color=[soft(c) for c in colors],
        edgecolor=colors, linewidth=1.2,
    )

    for bar, r2 in zip(bars, r2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{r2:.2f}", ha="center", va="bottom",
            fontsize=FONT_SIZE, color=GRAY_DARK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 0.5, rotation=30, ha="right")
    ax.set_ylabel("Reconstruction $R^2$")
    ax.set_ylim(0, 1.12)
    despine(ax)

    _save(fig, "bar_r2")


# -- Main --

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_nature_rc())

    # Gather all metrics
    metrics_list = []
    all_reliabilities = []
    for ds in DATASETS:
        summary = _load_summary(ds)
        kappa = _load_kappa(ds)
        rel = _get_reliability(ds)
        r = summary.get("reconstruction_r", np.nan)
        metrics_list.append({
            "label": ds["label"],
            "color": ds["color"],
            "k_star": kappa["k_star"],
            "r2": r ** 2,
            "mean_reliability": float(np.mean(rel)),
        })
        all_reliabilities.append((ds, rel))

    # Individual RSM pairs
    for ds, m in zip(DATASETS, metrics_list):
        print(f"RSM: {ds['label']}...")
        plot_rsm_pair(ds, m)

    # Bar charts
    print("Bar: rank...")
    plot_bar_rank(metrics_list)

    print("Bar: R²...")
    plot_bar_r2(metrics_list)

    print("Bar: reliability...")
    plot_bar_reliability(metrics_list)

    # Per-dimension reliability profiles
    print("Reliability profiles...")
    plot_reliability_profiles(all_reliabilities)

    print("Done.")


if __name__ == "__main__":
    main()
