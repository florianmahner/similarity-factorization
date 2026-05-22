"""Plot dimensionality CV curves from compact JSON outputs.

Examples
--------
Plot all completed dimensionality experiment CV JSONs:

    python -m experiments.datasets.dimensionality.plot_cv

Plot sandbox checks, such as VGG16 max_outer diagnostics:

    python -m experiments.datasets.dimensionality.plot_cv \
      --extra-json 'sandbox/dimensionality/vgg16_cv_protocol_check/outputs/*.json' \
      --output experiments/datasets/dimensionality/outputs/vgg16/cv_budget_check.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.colors import CYCLE, INDIGO, ROSE, TEAL

from . import io as _io


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    rows.extend(_load_experiment_rows(output_dir, args.datasets))
    rows.extend(_load_extra_rows(args.extra_json, args.only_engine))
    if not rows:
        raise SystemExit("No CV curves found.")

    df = pd.DataFrame(rows)
    if args.variant != "all":
        # Match anything whose label starts with "<variant>" (e.g. "5fold | outer=50 ...")
        df = df[df["label"].astype(str).str.startswith(args.variant)].reset_index(drop=True)
    if args.exclude:
        df = df[~df["dataset"].astype(str).isin(args.exclude)].reset_index(drop=True)
    df = _apply_min_rank(df, args.min_rank)
    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out
    _plot(df, out, log_x=args.log_x, log_y=args.log_y)
    print(f"wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_io.OUTPUT_DIR),
        help="Dimensionality output directory containing <dataset>/cross_validation.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to include from the dimensionality outputs.",
    )
    parser.add_argument(
        "--extra-json",
        nargs="*",
        default=[],
        help="Additional sandbox summary JSON paths or glob patterns.",
    )
    parser.add_argument(
        "--only-engine",
        default="pysrf_cross_val_score",
        help="Engine filter for sandbox summary JSONs; use 'all' to include every engine.",
    )
    parser.add_argument(
        "--output",
        default=str(_io.OUTPUT_DIR / "cv_curves.pdf"),
        help="Output figure path. Suffix controls format, usually .pdf or .png.",
    )
    parser.add_argument("--log-x", action="store_true", help="Log scale on rank axis.")
    parser.add_argument("--log-y", dest="log_y", action="store_true", help="Force log scale on V-MSE axis (default: linear).")
    parser.add_argument("--min-rank", type=int, default=10, help="Default rank cutoff (overridden per-dataset for small datasets in PER_DATASET_MIN_RANK).")
    parser.add_argument("--variant", default="5fold", help="Filter to this CV variant only (default 5fold). Use 'all' to keep both 5fold and 10fold panels.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset names to exclude from the plot.")
    parser.set_defaults(log_y=False)
    return parser.parse_args()


def _load_experiment_rows(output_dir: Path, datasets: list[str] | None) -> list[dict[str, Any]]:
    names = datasets
    if names is None:
        names = [
            p.name for p in sorted(output_dir.iterdir())
            if p.is_dir() and _io.cross_validation_path(p.name, output_dir).exists()
        ] if output_dir.exists() else []

    rows: list[dict[str, Any]] = []
    for dataset in names:
        payload = _io.read_cross_validation(dataset, output_dir)
        for variant, block in payload.get("validations", {}).items():
            params = block.get("params", {})
            srf_kwargs = params.get("srf_kwargs", {})
            label = _label(
                dataset=dataset,
                variant=variant,
                max_outer=srf_kwargs.get("max_outer"),
                p_cv=_p_cv(params),
                source="experiment",
            )
            ranks = [int(r) for r in block.get("ranks", block.get("completed_ranks", []))]
            means = block.get("val_mse_mean", [])
            sems = block.get("val_mse_sem", [None] * len(means))
            for rank, mean, sem in zip(ranks, means, sems, strict=False):
                rows.append({
                    "dataset": dataset,
                    "source": "experiment",
                    "label": label,
                    "rank": int(rank),
                    "mean": float(mean),
                    "sem": None if sem is None else float(sem),
                    "argmin_rank": block.get("argmin_rank"),
                    "max_outer": srf_kwargs.get("max_outer"),
                })
    return rows


def _load_extra_rows(patterns: list[str], only_engine: str) -> list[dict[str, Any]]:
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(p) for p in (matches if matches else [pattern]))

    rows: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metadata = payload.get("metadata", {})
        dataset = str(metadata.get("dataset", path.parent.parent.name))
        max_outer = metadata.get("max_outer")
        p_cv = metadata.get("p_cv")
        for row in payload.get("summary", []):
            engine = str(row.get("engine", "sandbox"))
            if only_engine != "all" and engine != only_engine:
                continue
            label = _label(
                dataset=dataset,
                variant=engine,
                max_outer=max_outer,
                p_cv=p_cv,
                source="sandbox",
            )
            rows.append({
                "dataset": dataset,
                "source": "sandbox",
                "label": label,
                "rank": int(row["rank"]),
                "mean": float(row["mean_val_mse"]),
                "sem": row.get("sem_val_mse"),
                "argmin_rank": row.get("argmin_rank"),
                "max_outer": max_outer,
            })
    return rows


def _label(dataset: str, variant: str, max_outer: Any, p_cv: Any, source: str) -> str:
    bits = []
    if source == "sandbox":
        bits.append("sandbox")
    bits.append(str(variant))
    if max_outer is not None:
        bits.append(f"outer={int(max_outer)}")
    if p_cv is not None:
        bits.append(f"p_cv={float(p_cv):.3f}")
    return " | ".join(bits)


def _p_cv(params: dict[str, Any]) -> float | None:
    p_train = params.get("sampling_fraction")
    n_folds = params.get("n_folds")
    if p_train is None or n_folds in (None, 1):
        return None
    return float(p_train) * float(n_folds) / float(n_folds - 1)


def _plot(df: pd.DataFrame, out: Path, log_x: bool = False, log_y: bool = True) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.85)

    datasets = list(dict.fromkeys(df["dataset"].astype(str)))
    n = len(datasets)
    ncols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 9 else 4))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, dataset in zip(axes_flat[:n], datasets, strict=True):
        sub = df[df["dataset"].astype(str) == dataset]
        labels = list(dict.fromkeys(sub["label"].astype(str)))
        for idx, label in enumerate(labels):
            curve = sub[sub["label"].astype(str) == label].sort_values("rank")
            color = _color_for_variant(_short_label(label), idx, len(labels))
            x = curve["rank"].to_numpy(dtype=int)
            y = curve["mean"].to_numpy(dtype=float)
            sem = curve["sem"].to_numpy(dtype=float)
            sns.lineplot(
                x=x, y=y, ax=ax, color=color, label=_short_label(label),
                marker="o", linewidth=1.6, markersize=4, legend=False,
            )
            if np.isfinite(sem).any():
                # 2-SEM band ("~95% normal CI") so it's actually visible at tight y-ranges.
                ax.fill_between(x, y - 2 * sem, y + 2 * sem, color=color, alpha=0.25, linewidth=0)
                # Tiny error-bar caps at each point — most visible cue for the small-band case.
                ax.errorbar(x, y, yerr=2 * sem, fmt="none", ecolor=color, alpha=0.6,
                            elinewidth=0.6, capsize=2, capthick=0.6, zorder=2)
            argmin = curve["argmin_rank"].dropna()
            if not argmin.empty:
                ax.axvline(int(argmin.iloc[0]), color=color, linestyle=":", linewidth=0.9, alpha=0.7)

        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_xlabel("rank")
        ax.set_ylabel("Validation MSE")
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if len(labels) > 1:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    if out.suffix.lower() != ".png":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _short_label(label: str) -> str:
    # "5fold | outer=50 | p_cv=0.892" -> "5fold"
    return label.split(" | ")[0]


# Stable variant colors so the same variant has the same color across panels.
VARIANT_COLORS = {
    "5fold": INDIGO,
    "10fold": TEAL,
}


def _color_for_variant(variant: str, idx: int, n_labels: int) -> str:
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    if n_labels == 1:
        return INDIGO
    return CYCLE[idx % len(CYCLE)]


# Per-dataset min-rank for the plot — small datasets show from rank=2 because
# rank=2 is a legitimate data point in their CV grid; larger datasets start at
# rank=10 because rank<10 is typically an extreme outlier that compresses the
# y-axis. Datasets not listed use the CLI --min-rank value.
PER_DATASET_MIN_RANK = {
    "mur92": 2,
    "peterson_animals": 2,
    "peterson_various": 2,
    "clip_vit_l14": 5,
}


def _apply_min_rank(df: pd.DataFrame, default_min: int) -> pd.DataFrame:
    keep_masks = []
    for dataset, sub in df.groupby("dataset"):
        cutoff = PER_DATASET_MIN_RANK.get(str(dataset), default_min)
        keep_masks.append(sub["rank"] >= cutoff)
    if not keep_masks:
        return df
    mask = pd.concat(keep_masks).sort_index()
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
