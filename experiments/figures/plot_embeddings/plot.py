"""Embedding overview figure: observed vs reconstructed RSMs and factorization quality.

Layout (180mm, 2 rows):
  Row 1 (a): [placeholder for example dimensions -- added manually]
  Row 2 (b): Observed vs SRF RSM heatmap pairs per dataset (cluster-sorted)
  Row 3 (c): Estimated rank k*  |  (d): Per-dimension reliability profiles

Saves individual panels as separate PDFs + a composed overview PNG.

Data sources:
  - experiments/datasets/consensus/outputs/{dataset}/  (embeddings, reliability)
  - experiments/datasets/ranks/kappa/outputs/           (k* estimates)
  - Similarity matrices loaded via dataset loaders (cached to .cache/)

Usage:
    poetry run python experiments/figures/plot_embeddings/plot.py
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
    ROSE, TEAL, INDIGO, SAND, PURPLE, CYAN, WINE,
    GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE,
    CMAP_SEQ, CMAP_IRID, soft, lighten,
)
from src.utils.figure_theme import despine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSENSUS_DIR = PROJECT_ROOT / "experiments" / "datasets" / "consensus" / "outputs"
KAPPA_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ranks" / "kappa" / "outputs"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
ROW_HEIGHT_MM = 35
ROW_HEIGHT_IN = ROW_HEIGHT_MM / 25.4
FS = 5

# All datasets for bar chart + reliability profiles
DATASETS = [
    {"label": "Mur et al.", "consensus": "mur92", "kappa": "mur92", "subject_id": None},
    {"label": "Peterson\n(Animals)", "consensus": "peterson-animals",
     "kappa": "peterson_animals", "subject_id": None},
    {"label": "Peterson\n(Various)", "consensus": "peterson-various",
     "kappa": "peterson_various", "subject_id": None},
    {"label": "VGG-16\n(THINGS+)", "consensus": "vgg16", "kappa": "vgg16", "subject_id": None},
    {"label": "THINGS\n(Monkey)", "consensus": None, "kappa": "things-monkey-22k",
     "subject_id": None},
    {"label": "SWOW", "consensus": "swow", "kappa": "swow", "subject_id": None},
    {"label": "NSD\n(subj01)", "consensus": "nsd", "kappa": "nsd_subj01", "subject_id": 1},
]

# Subset with RSMs and image grids
RSM_DATASETS = ["mur92", "peterson-animals", "peterson-various", "vgg16"]

# Muted line colors for reliability profiles only
PROFILE_COLORS = [ROSE, TEAL, INDIGO, SAND, PURPLE, CYAN, WINE]

# Neutral bar styling
BAR_FILL = "#D4D4D4"
BAR_EDGE = GRAY


def _nature_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS + 1,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


# -- Data loading -------------------------------------------------------

def _consensus_dir(ds):
    if ds["consensus"] is None:
        return None
    d = CONSENSUS_DIR / ds["consensus"]
    if ds["subject_id"] is not None:
        d = d / f"subj{ds['subject_id']:02d}"
    return d


def _load_embedding(ds):
    d = _consensus_dir(ds)
    if d is None:
        return None
    return np.load(d / "embedding.npy")


def _load_summary(ds):
    d = _consensus_dir(ds)
    if d is None:
        return {}
    return json.loads((d / "summary.json").read_text())


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
        root += "mur92" if name == "mur92" else "peterson"
        sim = load_dataset(name, root=root).rsm
    elif name == "vgg16":
        feat_path = PROJECT_ROOT / "data" / "features" / "vgg16" / "vgg16_features.npy"
        features = np.load(feat_path)
        sim = compute_similarity(features, features, "linear")
        sim = sim / sim.max()
    elif name == "swow":
        result = load_dataset("swow", root=str(PROJECT_ROOT / "data" / "small-world-of-words"))
        sim = result.rsm
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


def _get_reliability(ds):
    """Load CV split-half reliability (precomputed) or fall back to naive."""
    d = _consensus_dir(ds)
    if d is None:
        return None
    cv_path = d / "cv_reliability.npy"
    if cv_path.exists():
        return np.load(cv_path)
    naive_path = d / "reliability.npy"
    if naive_path.exists():
        return np.load(naive_path)
    return None


def _load_reliability_summary():
    """Load precomputed CV reliability summary CSV."""
    path = CONSENSUS_DIR / "reliability_summary.csv"
    if path.exists():
        import pandas as pd
        return pd.read_csv(path)
    return None


def _cluster_order(rsm, ds):
    """Hierarchical clustering leaf order for RSM visualization (cached)."""
    cache_dir = OUTPUT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = ds["consensus"]
    if ds["subject_id"] is not None:
        cache_key += f"_subj{ds['subject_id']:02d}"
    cache_path = cache_dir / f"order_{cache_key}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    print(f"  Computing cluster order for {cache_key}...")
    rsm = np.array(rsm, dtype=np.float64)
    rsm = np.nan_to_num(rsm, nan=0.0, posinf=0.0, neginf=0.0)
    n = rsm.shape[0]
    dist = 1.0 - rsm / (rsm.max() + 1e-10)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = dist[np.triu_indices(n, k=1)]
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)
    z = linkage(condensed, method="average")
    order = leaves_list(z)
    np.save(cache_path, order)
    return order


def _save(fig, name, fmt="pdf"):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{name}.{fmt}"
    if fmt == "svg":
        fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    elif fmt == "pdf":
        plt.rcParams["savefig.bbox"] = "standard"
        fig.savefig(out_path, format="pdf")
    else:
        fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -- Individual panels ---------------------------------------------------

def _prepare_rsm_pair(ds):
    """Load embedding, similarity, cluster-sort, return (true_sorted, pred_sorted, vmax)."""
    w = _load_embedding(ds)
    if w is None:
        return None, None, None
    predicted = w @ w.T

    try:
        true_rsm = np.array(_load_similarity(ds))
    except Exception as e:
        print(f"  Could not load similarity for {ds['consensus']}: {e}")
        true_rsm = predicted.copy()

    n_emb = w.shape[0]
    if true_rsm.shape[0] != n_emb:
        n = min(true_rsm.shape[0], n_emb)
        true_rsm = true_rsm[:n, :n]
        predicted = predicted[:n, :n]

    order = _cluster_order(true_rsm, ds)
    true_sorted = true_rsm[np.ix_(order, order)]
    pred_sorted = predicted[np.ix_(order, order)]
    np.fill_diagonal(true_sorted, 0)
    np.fill_diagonal(pred_sorted, 0)

    vmax = np.percentile(true_sorted, 98)
    return true_sorted, pred_sorted, vmax


def _rsm_to_image(rsm, vmax, cmap=CMAP_IRID):
    """Convert RSM matrix to RGBA image array via colormap."""
    norm = plt.Normalize(vmin=0, vmax=vmax)
    return cmap(norm(rsm))


def save_rsm_pair_svg(ds, r2):
    """Save observed + SRF RSM side by side as one SVG (heatmaps rasterized, text vector)."""
    true_sorted, pred_sorted, vmax = _prepare_rsm_pair(ds)
    if true_sorted is None:
        print(f"  Skipping RSM for {ds['label']} (no consensus embedding)")
        return

    panel_w = 1.4
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2 + 0.15, panel_w + 0.35))
    fig.subplots_adjust(wspace=0.08, top=0.82, bottom=0.02, left=0.02, right=0.98)

    for ax, data, title in [(axes[0], true_sorted, "Observed"),
                             (axes[1], pred_sorted, "Predicted")]:
        ax.imshow(data, cmap=CMAP_IRID, vmin=0, vmax=vmax, aspect="equal",
                  interpolation="none", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color(GRAY_LIGHT)
        ax.set_title(title, fontsize=FS, pad=3)

    fig.suptitle(f"{ds['label']}  ($R^2 = {r2:.2f}$)",
                 fontsize=FS + 1, fontweight="bold", y=0.97)

    name = ds["consensus"]
    if ds["subject_id"] is not None:
        name += f"_subj{ds['subject_id']:02d}"
    _save(fig, f"rsm_{name}", fmt="svg")


def panel_rsm_pair(fig, gs_slot, ds, r2):
    """Observed vs SRF RSM heatmap pair for one dataset (rasterized in vector output)."""
    true_sorted, pred_sorted, vmax = _prepare_rsm_pair(ds)
    if true_sorted is None:
        ax = fig.add_subplot(gs_slot)
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{ds['label']}\n(no consensus yet)",
                ha="center", va="center", fontsize=FS, color=GRAY,
                transform=ax.transAxes)
        return

    inner = gs_slot.subgridspec(1, 2, wspace=0.06)

    for j, (data, title) in enumerate([(true_sorted, "Observed"), (pred_sorted, "Predicted")]):
        ax = fig.add_subplot(inner[j])
        ax.imshow(data, cmap=CMAP_IRID, vmin=0, vmax=vmax, aspect="equal",
                  interpolation="none", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color(GRAY_LIGHT)
        ax.set_title(title, fontsize=FS - 0.5, pad=2)

    # Dataset label + R² above the pair
    label_ax = fig.add_subplot(gs_slot)
    label_ax.set_axis_off()
    label_ax.set_title(
        f"{ds['label']}\n$R^2 = {r2:.2f}$",
        fontsize=FS, fontweight="bold", pad=6, linespacing=1.15,
    )


def panel_rank_bars(ax, metrics):
    """Vertical bar chart of estimated k*, sorted ascending."""
    order = np.argsort([m["k_star"] for m in metrics])
    labels = [metrics[i]["label"] for i in order]
    k_vals = [metrics[i]["k_star"] for i in order]
    n_ds = len(labels)
    x = np.arange(n_ds)

    # Subtle horizontal grid behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRAY_PALE, linewidth=0.3, zorder=0)

    # Bars with subtle fill gradient -- darker for higher rank
    max_k = max(k_vals)
    for i, (xi, k) in enumerate(zip(x, k_vals)):
        t = k / max_k
        fill = lighten(GRAY, 0.55 - 0.35 * t)
        edge = lighten(GRAY_DARK, 0.3 - 0.2 * t)
        ax.bar(
            xi, k, width=0.52,
            color=fill, edgecolor=edge, linewidth=0.7,
            zorder=2,
        )
        ax.text(
            xi, k + max_k * 0.03,
            str(k), ha="center", va="bottom",
            fontsize=FS, fontweight="bold", color=GRAY_DARK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS, linespacing=0.9, rotation=45, ha="right")
    ax.set_ylabel("Estimated rank ($k^*$)")
    ax.set_ylim(0, max_k * 1.15)
    despine(ax)
    ax.tick_params(bottom=False)


def panel_reliability_bars(ax, metrics):
    """Vertical bar chart of CV split-half reliability, sorted ascending."""
    rel_summary = _load_reliability_summary()

    # Map dataset labels to CV reliability
    label_to_cv = {}
    if rel_summary is not None:
        for _, row in rel_summary.iterrows():
            label_to_cv[row["dataset"]] = row["cv_rel_mean"]

    # Match metrics order and get CV values
    order = np.argsort([m["k_star"] for m in metrics])
    labels = [metrics[i]["label"] for i in order]

    # Map metric labels back to dataset keys for CSV lookup
    label_to_key = {}
    for ds in DATASETS:
        key = ds["consensus"]
        if ds["subject_id"] is not None:
            key = f"{key}_subj{ds['subject_id']:02d}"
        label_to_key[ds["label"]] = key

    cv_vals = []
    for i in order:
        lab = metrics[i]["label"]
        key = label_to_key.get(lab, "")
        cv_vals.append(label_to_cv.get(key, np.nan))

    n_ds = len(labels)
    x = np.arange(n_ds)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRAY_PALE, linewidth=0.3, zorder=0)

    for i, (xi, val) in enumerate(zip(x, cv_vals)):
        if np.isnan(val):
            continue
        t = val
        fill = lighten(ROSE, 0.65 - 0.35 * t)
        edge = ROSE
        ax.bar(xi, val, width=0.52, color=fill, edgecolor=edge,
               linewidth=0.7, zorder=2)
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom",
                fontsize=FS, fontweight="bold", color=GRAY_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS, linespacing=0.9, rotation=45, ha="right")
    ax.set_ylabel("CV split-half reliability")
    ax.set_ylim(0.75, 1.08)
    ax.axhline(0.8, color=GRAY_LIGHT, linewidth=0.4, linestyle="--", zorder=0)
    despine(ax)
    ax.tick_params(bottom=False)


def panel_reliability_profiles(ax, all_reliabilities):
    """Per-dimension reliability curves, sorted descending, one line per dataset."""
    for i, (ds, rel) in enumerate(all_reliabilities):
        if rel is None:
            continue
        rel_sorted = np.sort(rel)[::-1]
        frac = np.linspace(0, 1, len(rel_sorted))
        short = ds["label"].replace("\n", " ")
        ax.plot(
            frac, rel_sorted,
            color=PROFILE_COLORS[i], linewidth=0.9,
            label=short,
        )

    ax.axhline(0.5, color=GRAY_LIGHT, linewidth=0.4, linestyle="--", zorder=0)
    ax.set_xlabel("Dimension (fraction of $k^*$)")
    ax.set_ylabel("Reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=FS - 0.5, loc="lower left", frameon=False,
        handlelength=1.5, handletextpad=0.4, labelspacing=0.25,
    )
    despine(ax)


# -- Dimension image cards (one sparse dim per dataset, 2x3 top images) --

def _load_images(ds):
    """Load images for a dataset."""
    import pandas as pd
    from datasets import load_dataset

    name = ds["consensus"]
    if name in ("mur92", "peterson-animals", "peterson-various"):
        roots = {
            "peterson-animals": "/SSD/datasets/similarity_datasets/peterson",
            "peterson-various": "/SSD/datasets/similarity_datasets/peterson",
            "mur92": "/SSD/datasets/similarity_datasets/mur92",
        }
        result = load_dataset(name, root=roots[name])
        if hasattr(result, "metadata") and "images" in result.metadata:
            imgs = result.metadata["images"]
            return imgs if isinstance(imgs, np.ndarray) else [Path(p) for p in imgs]
    elif name == "vgg16":
        meta = pd.read_csv(PROJECT_ROOT / "data" / "features" / "vgg16" / "metadata.csv")
        return [Path(p) for p in meta["path"]]
    return None


def _get_img(images, idx):
    """Load a single image by index."""
    from PIL import Image as PILImage
    if isinstance(images, np.ndarray):
        return PILImage.fromarray(images[idx])
    p = Path(images[idx])
    if not p.exists():
        return None
    return PILImage.open(p).convert("RGB")


def _pick_sparse_dim(embedding, rng):
    """Pick one sparse dimension above median sparsity."""
    sparsity = (embedding == 0).mean(axis=0)
    good = np.where(sparsity > np.median(sparsity))[0]
    if len(good) == 0:
        good = np.arange(embedding.shape[1])
    return rng.choice(good)


def save_dimension_card(ds):
    """One example dimension as a 2x3 image card (matching plot_datasets style)."""
    w = _load_embedding(ds)
    if w is None:
        return
    images = _load_images(ds)
    if images is None:
        return

    rng = np.random.default_rng(42)
    dim = _pick_sparse_dim(w, rng)

    img_rows, img_cols = 2, 3
    top_k = img_rows * img_cols
    top_idx = np.argsort(w[:, dim])[::-1][:top_k]

    cell = 0.55
    fig, axes = plt.subplots(
        img_rows, img_cols,
        figsize=(img_cols * cell, img_rows * cell + 0.3),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02)

    for j in range(top_k):
        r, c = j // img_cols, j % img_cols
        ax = axes[r, c]
        img = _get_img(images, top_idx[j])
        if img is not None:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color(GRAY_LIGHT)

    short_label = ds["label"].replace("\n", " ")
    fig.suptitle(short_label, fontsize=FS + 1, fontweight="bold", y=0.98)

    _save(fig, f"dim_{ds['consensus']}", fmt="svg")


# -- MDS with images -----------------------------------------------------

def save_mds_plot(ds):
    """MDS scatter with actual images at coordinates, colored by dominant dimension."""
    from sklearn.manifold import MDS
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    w = _load_embedding(ds)
    if w is None:
        return
    images = _load_images(ds)
    if images is None:
        return

    # MDS from the SRF reconstruction
    sim = w @ w.T
    np.fill_diagonal(sim, 0)
    dist = sim.max() - sim
    np.fill_diagonal(dist, 0)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(dist)

    # Dominant dimension per item
    dominant = np.argmax(w, axis=1)
    k = w.shape[1]

    # Use a limited palette for the top dimensions
    from matplotlib.colors import ListedColormap
    dim_colors = plt.cm.tab20(np.linspace(0, 1, min(k, 20)))

    fig_w = 3.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    img_zoom = 0.12 if w.shape[0] > 100 else 0.18

    for i in range(w.shape[0]):
        img = _get_img(images, i)
        if img is None:
            continue
        img_arr = np.array(img.resize((64, 64)))
        im = OffsetImage(img_arr, zoom=img_zoom)
        ab = AnnotationBbox(
            im, (coords[i, 0], coords[i, 1]),
            frameon=True,
            bboxprops=dict(
                edgecolor=dim_colors[dominant[i] % len(dim_colors)],
                linewidth=1.2, boxstyle="round,pad=0.05",
            ),
        )
        ax.add_artist(ab)

    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    despine(ax, left=True, bottom=True)
    ax.tick_params(left=False, bottom=False)

    short_label = ds["label"].replace("\n", " ")
    fig.suptitle(f"{short_label} -- MDS", fontsize=FS + 1, fontweight="bold", y=0.98)

    _save(fig, f"mds_{ds['consensus']}", fmt="svg")


# -- Save individual panels as standalone PDFs ---------------------------

def save_individual_panels(metrics, all_reliabilities):
    """Save each panel as a standalone file for manual composition."""

    # RSM pairs only for selected datasets
    for i, ds in enumerate(DATASETS):
        if ds["consensus"] in RSM_DATASETS:
            save_rsm_pair_svg(ds, metrics[i]["r2"])

    # One example dimension per dataset (2x3 image card)
    for ds in DATASETS:
        if ds["consensus"] in RSM_DATASETS:
            print(f"Dimension card: {ds['label']}...")
            save_dimension_card(ds)

    # MDS plot for Peterson Animals
    peterson_animals = [d for d in DATASETS if d["consensus"] == "peterson-animals"][0]
    print("MDS: Peterson (Animals)...")
    save_mds_plot(peterson_animals)

    # Rank bars
    fig_k, ax_k = plt.subplots(figsize=(2.7, 2.0))
    fig_k.subplots_adjust(left=0.18, right=0.95, bottom=0.25, top=0.95)
    panel_rank_bars(ax_k, metrics)
    _save(fig_k, "bar_rank", fmt="svg")

    # Reliability bars (CV split-half)
    fig_rel_bar, ax_rel_bar = plt.subplots(figsize=(2.7, 2.0))
    fig_rel_bar.subplots_adjust(left=0.18, right=0.95, bottom=0.25, top=0.95)
    panel_reliability_bars(ax_rel_bar, metrics)
    _save(fig_rel_bar, "bar_reliability", fmt="svg")

    # Reliability profiles
    fig_rel, ax_rel = plt.subplots(figsize=(3.2, 2.0))
    fig_rel.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.95)
    panel_reliability_profiles(ax_rel, all_reliabilities)
    _save(fig_rel, "reliability_profiles", fmt="svg")


# -- Assembled figure (single SVG) ----------------------------------------

def save_assembled(metrics, all_reliabilities):
    """Full assembled figure as one SVG.

    Layout (180mm wide, 2 columns):
      LEFT (narrow):  k* horizontal bars for all datasets
      RIGHT (wide):   2 rows x 4 cols -- dim cards (top) + RSM pairs (bottom)
    """
    rsm_datasets = [d for d in DATASETS if d["consensus"] in RSM_DATASETS]
    n_rsm = len(rsm_datasets)
    rsm_metrics = [metrics[DATASETS.index(d)] for d in rsm_datasets]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * 2.6))
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.28, 0.72],
        wspace=0.12,
        left=0.04, right=0.98, top=0.95, bottom=0.04,
    )

    # -- LEFT: k* horizontal bars --
    ax_k = fig.add_subplot(outer[0])
    panel_rank_bars(ax_k, metrics)

    # -- RIGHT: dim cards (top) + RSM pairs (bottom) --
    right_gs = outer[1].subgridspec(
        2, n_rsm, wspace=0.22, hspace=0.35,
        height_ratios=[1.0, 1.1],
    )

    rng = np.random.default_rng(42)
    img_rows, img_cols = 2, 3
    top_k = img_rows * img_cols

    for di, ds in enumerate(rsm_datasets):
        # Top: dimension image card
        w = _load_embedding(ds)
        images = _load_images(ds)
        if w is not None and images is not None:
            dim = _pick_sparse_dim(w, rng)
            top_idx = np.argsort(w[:, dim])[::-1][:top_k]

            card = right_gs[0, di].subgridspec(img_rows, img_cols, wspace=0.03, hspace=0.03)
            for j in range(top_k):
                r, c = j // img_cols, j % img_cols
                ax = fig.add_subplot(card[r, c])
                img = _get_img(images, top_idx[j])
                if img is not None:
                    ax.imshow(img, interpolation="lanczos", rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                    spine.set_color(GRAY_LIGHT)

            title_ax = fig.add_subplot(right_gs[0, di])
            title_ax.set_axis_off()
            short = ds["label"].replace("\n", " ")
            title_ax.set_title(short, fontsize=FS, fontweight="bold",
                               pad=3, linespacing=1.1)

        # Bottom: RSM pair
        panel_rsm_pair(fig, right_gs[1, di], ds, rsm_metrics[di]["r2"])

    # Panel labels
    fig.text(0.005, 0.98, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.30, 0.98, "b", fontsize=8, fontweight="bold", va="top")
    fig.text(0.30, 0.48, "c", fontsize=8, fontweight="bold", va="top")

    _save(fig, "figure_embeddings", fmt="svg")
    _save(fig, "figure_embeddings_preview", fmt="png")


# -- Main ----------------------------------------------------------------

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_nature_rc())

    # Gather metrics
    metrics = []
    all_reliabilities = []
    for ds in DATASETS:
        summary = _load_summary(ds)
        kappa = _load_kappa(ds)
        rel = _get_reliability(ds)
        r = summary.get("reconstruction_r", np.nan)
        metrics.append({
            "label": ds["label"],
            "k_star": kappa["k_star"],
            "r2": r ** 2,
        })
        all_reliabilities.append((ds, rel))

    print("Saving individual panels...")
    save_individual_panels(metrics, all_reliabilities)

    print("Saving assembled figure...")
    save_assembled(metrics, all_reliabilities)

    print("Done.")


if __name__ == "__main__":
    main()
