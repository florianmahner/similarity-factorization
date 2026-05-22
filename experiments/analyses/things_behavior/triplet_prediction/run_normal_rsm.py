"""Triplet prediction at the consensus rank using the normal RSM.

Uses the same similarity matrix as the dimensionality CV (Laplace-smoothed
build_similarity output, alpha=1.0, full 4.7mio trainset) and the same SRF
hyperparameters the CV used (rho=3.0, max_outer=200, max_inner=30, tol=0.0).

This replaces the previous bias-aware approach (run.py with weight_mode +
prior_strength tuning) because the matrix is now fixed to match what
selected the rank — so the bias-aware tuning is no longer applicable.

Outputs (under outputs/triplets_normal_rsm/):
    final_results.csv  -- per-seed val accuracy
    summary.csv        -- aggregated mean +/- std vs SPoSE baseline
    selection.json     -- run metadata
    triplet_prediction_barplot.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.analyses.things_behavior.summary_stats import (
    NOISE_CEILING_PCT,
    NOISE_CEILING_STD_PCT,
)
from similarity import build_similarity
from src.colors import GRAY_DARK
from src.triplet_prediction import (
    load_optimal_rank,
    load_triplets_file,
    triplet_accuracy,
)
from src.utils.figure_theme import create_figure, despine, save_figure
from pysrf import SRF

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "triplets_normal_rsm"
CHANCE_PCT = 100.0 / 3.0

# Final-eval SRF hyperparameters: keep rho from the dimensionality CV (3.0),
# but train until convergence (the CV used max_outer=200 / tol=0.0 to pick a
# rank quickly; for best per-fit performance we let it run longer).
# Reference: a rank=35 fit with these settings converges around iter 320 on
# this matrix (see similarity_48/run_direct.py).
SRF_KWARGS = {
    "rho": 3.0,
    "max_outer": 2000,
    "max_inner": 50,
    "tol": 1e-4,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-kind",
        choices=["argmin", "one_se"],
        default="argmin",
        help="Which rank to load from the primary CV variant.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated SRF seeds.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--spose-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "things" / "triplets_47" / "validationset.txt",
    )
    return parser.parse_args()


def _load_dataset_cfg() -> "OmegaConf":
    """Mirror experiments/datasets/dimensionality/_loader.py:_load_dataset_cfg."""
    raw = OmegaConf.load(PROJECT_ROOT / "configs" / "dataset" / "things_behavior.yaml")
    parent = OmegaConf.create({
        "paths": OmegaConf.load(PROJECT_ROOT / "configs" / "paths" / "local.yaml"),
        "dataset": raw,
        "project_root": str(PROJECT_ROOT),
    })
    OmegaConf.resolve(parent)
    return parent.dataset


def _fit_and_score(
    similarity: np.ndarray,
    val_triplets: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    t0 = time.time()
    model = SRF(rank=rank, random_state=seed, verbose=0, **SRF_KWARGS)
    embedding = model.fit_transform(similarity)
    acc = triplet_accuracy(embedding, val_triplets)
    elapsed = time.time() - t0
    log.info("  seed=%d  acc=%.4f  (%.1fs)", seed, acc, elapsed)
    return {"model": "SRF", "rank": int(rank), "seed": int(seed), "val_acc": float(acc)}


def _plot_bar(summary_df: pd.DataFrame, output_dir: Path, rank: int) -> None:
    fig, ax = create_figure(size="single", pad_left=0.8, pad_bottom=0.7, pad_top=0.25)
    plot_df = summary_df[summary_df["model"].isin(["SRF", "SPoSE"])].copy()
    colors = ["#000000", GRAY_DARK]
    chance_color = "#4DBA6A"
    x = np.arange(len(plot_df))

    bars = ax.bar(
        x,
        plot_df["accuracy_pct"],
        yerr=plot_df["accuracy_std_pct"],
        color=colors,
        edgecolor=colors,
        linewidth=0.8,
        width=0.42,
        capsize=2.5,
        zorder=2,
    )
    ax.axhspan(
        NOISE_CEILING_PCT - NOISE_CEILING_STD_PCT,
        NOISE_CEILING_PCT + NOISE_CEILING_STD_PCT,
        color="#C7C7C7",
        alpha=1.0,
        zorder=0,
    )
    ax.axhline(CHANCE_PCT, color=chance_color, linestyle="-", linewidth=1.0, zorder=1)
    ax.set_xticks(x, plot_df["model"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30.0, 72.5)
    ax.set_yticks([30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
    ax.set_title("Behavioural prediction (normal RSM)")
    despine(ax)
    ax.text(1.48, CHANCE_PCT + 0.35, "chance", color=chance_color, ha="right", va="bottom", fontsize=8)
    ax.text(
        1.48,
        NOISE_CEILING_PCT + NOISE_CEILING_STD_PCT + 0.15,
        "noise ceiling",
        color=GRAY_DARK,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    for bar, (_, row) in zip(bars, plot_df.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            row["accuracy_pct"] + row["accuracy_std_pct"] + 0.15,
            f"{row['accuracy_pct']:.2f}%",
            ha="center", va="bottom", fontsize=8,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            31.1,
            f"{int(row['rank'])}d",
            ha="center", va="bottom", fontsize=8, color="white",
        )
    pdf_path = output_dir / "triplet_prediction_barplot"
    save_figure(fig, pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rank = load_optimal_rank("things_behavior", kind=args.rank_kind, project_root=PROJECT_ROOT)
    log.info("=== triplet_prediction (normal RSM) ===")
    log.info("rank=%d (%s)  seeds=%s  srf_kwargs=%s", rank, args.rank_kind, seeds, SRF_KWARGS)

    log.info("Building normal RSM via build_similarity(things_behavior) ...")
    t0 = time.time()
    dataset_cfg = _load_dataset_cfg()
    similarity = build_similarity(dataset_cfg)
    log.info("  shape=%s  nan_frac=%.2e  built in %.1fs",
             similarity.shape, float(np.mean(np.isnan(similarity))), time.time() - t0)

    log.info("Loading validation triplets and SPoSE embedding ...")
    val_triplets = load_triplets_file(args.val_file)
    spose = np.maximum(np.loadtxt(args.spose_file), 0.0)
    log.info("  val_triplets=%d  spose=%s", len(val_triplets), spose.shape)

    log.info("Fitting SRF rank=%d for %d seeds ...", rank, len(seeds))
    t_all = time.time()
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_fit_and_score)(similarity, val_triplets, rank, seed)
        for seed in seeds
    )
    log.info("All %d seeds done in %.1fs", len(seeds), time.time() - t_all)

    final_df = pd.DataFrame(rows)
    final_df["eval_split"] = "validationset"
    final_df.to_csv(args.output_dir / "final_results.csv", index=False)

    spose_acc = triplet_accuracy(spose, val_triplets)
    srf_acc = final_df["val_acc"].astype(float)
    summary_df = pd.DataFrame([
        {
            "model": "SRF",
            "rank": rank,
            "accuracy_pct": float(srf_acc.mean() * 100),
            "accuracy_std_pct": float(srf_acc.std(ddof=1) * 100) if len(srf_acc) > 1 else 0.0,
            "noise_ceiling_pct": float(100.0 * srf_acc.mean() / (NOISE_CEILING_PCT / 100.0)),
        },
        {
            "model": "SPoSE",
            "rank": spose.shape[1],
            "accuracy_pct": float(spose_acc * 100),
            "accuracy_std_pct": 0.0,
            "noise_ceiling_pct": float(100.0 * spose_acc / (NOISE_CEILING_PCT / 100.0)),
        },
        {
            "model": "Noise ceiling",
            "rank": np.nan,
            "accuracy_pct": NOISE_CEILING_PCT,
            "accuracy_std_pct": NOISE_CEILING_STD_PCT,
            "noise_ceiling_pct": 100.0,
        },
    ])
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    (args.output_dir / "selection.json").write_text(json.dumps({
        "rank": rank,
        "rank_kind": args.rank_kind,
        "rsm_source": "build_similarity(things_behavior) -- Laplace alpha=1.0, full 4.7mio trainset",
        "srf_kwargs": SRF_KWARGS,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "noise_ceiling_pct": NOISE_CEILING_PCT,
        "noise_ceiling_std_pct": NOISE_CEILING_STD_PCT,
    }, indent=2))

    _plot_bar(summary_df, args.output_dir, rank)

    log.info("\n=== summary ===")
    log.info("%s", summary_df.to_string(index=False))
    log.info("Wrote %s", args.output_dir)


if __name__ == "__main__":
    main()
