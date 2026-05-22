"""Triplet prediction on the bias-aware THINGS-behavior matrix.

Reads the CV-selected rank from
``experiments/analyses/things_behavior/bias_aware/dimensionality/outputs/cross_validation.json``
and fits SRF at that rank with the same paper-fixed SRF hyperparams (rho=3.0,
training to convergence). Evaluates on data/things/triplets_47/validationset.txt.

Run via:
    poetry run python experiments/analyses/things_behavior/bias_aware/triplet_prediction/run.py
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
from pysrf import SRF

from experiments.analyses.things_behavior.bias_aware._matrix import (
    ALPHA,
    PRIOR_STRENGTH,
    WEIGHT_MODE,
    build_bias_aware_things_matrix,
    load_things_train_triplets,
)
from experiments.analyses.things_behavior.summary_stats import (
    NOISE_CEILING_PCT,
    NOISE_CEILING_STD_PCT,
)
from src.colors import GRAY_DARK
from src.triplet_prediction import load_triplets_file, triplet_accuracy
from src.utils.figure_theme import create_figure, despine, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
CV_JSON = HERE.parent / "dimensionality" / "outputs" / "cross_validation.json"
DEFAULT_VAL_FILE = PROJECT_ROOT / "data" / "things" / "triplets_47" / "validationset.txt"
DEFAULT_SPOSE = PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt"
CHANCE_PCT = 100.0 / 3.0

SRF_KWARGS = {
    "rho": 3.0,
    "max_outer": 2000,
    "max_inner": 50,
    "tol": 1e-4,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-kind",
        choices=["argmin", "one_se"],
        default="argmin",
        help="Which rank to load from cross_validation.json.",
    )
    parser.add_argument("--rank-override", type=int, default=None,
                        help="Bypass cross_validation.json; use this rank instead.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _load_cv_rank(kind: str) -> tuple[int, str]:
    payload = json.loads(CV_JSON.read_text())
    key = f"{kind}_rank"
    if key not in payload:
        raise KeyError(f"{key!r} not in {CV_JSON}")
    return int(payload[key]), str(CV_JSON)


def _fit_and_score(similarity, val_triplets, rank: int, seed: int) -> dict:
    t0 = time.time()
    model = SRF(rank=rank, random_state=seed, verbose=0, **SRF_KWARGS)
    embedding = model.fit_transform(similarity)
    acc = triplet_accuracy(embedding, val_triplets)
    log.info("  seed=%d  acc=%.4f  (%.1fs)", seed, acc, time.time() - t0)
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
        color=colors, edgecolor=colors, linewidth=0.8, width=0.42,
        capsize=2.5, zorder=2,
    )
    ax.axhspan(
        NOISE_CEILING_PCT - NOISE_CEILING_STD_PCT,
        NOISE_CEILING_PCT + NOISE_CEILING_STD_PCT,
        color="#C7C7C7", alpha=1.0, zorder=0,
    )
    ax.axhline(CHANCE_PCT, color=chance_color, linestyle="-", linewidth=1.0, zorder=1)
    ax.set_xticks(x, plot_df["model"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30.0, 72.5)
    ax.set_yticks([30, 35, 40, 45, 50, 55, 60, 65, 70])
    ax.set_title("Behavioural prediction (bias-aware RSM)")
    despine(ax)
    ax.text(1.48, CHANCE_PCT + 0.35, "chance",
            color=chance_color, ha="right", va="bottom", fontsize=8)
    ax.text(1.48, NOISE_CEILING_PCT + NOISE_CEILING_STD_PCT + 0.15, "noise ceiling",
            color=GRAY_DARK, ha="right", va="bottom", fontsize=8)
    for bar, (_, row) in zip(bars, plot_df.iterrows(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                row["accuracy_pct"] + row["accuracy_std_pct"] + 0.15,
                f"{row['accuracy_pct']:.2f}%",
                ha="center", va="bottom", fontsize=8)
        ax.text(bar.get_x() + bar.get_width() / 2.0, 31.1,
                f"{int(row['rank'])}d", ha="center", va="bottom", fontsize=8, color="white")
    pdf_path = output_dir / "triplet_prediction_barplot"
    save_figure(fig, pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.rank_override is not None:
        rank, rank_source = int(args.rank_override), "override"
    else:
        rank, rank_source = _load_cv_rank(args.rank_kind)

    log.info("=== triplet_prediction (bias-aware RSM) ===")
    log.info("rank=%d  rank_source=%s  rank_kind=%s",
             rank, rank_source, args.rank_kind if args.rank_override is None else "override")
    log.info("srf_kwargs=%s", SRF_KWARGS)
    log.info("matrix: alpha=%s, weight_mode=%s, lambda=%s", ALPHA, WEIGHT_MODE, PRIOR_STRENGTH)
    log.info("seeds=%s", seeds)

    log.info("Building bias-aware matrix ...")
    M = build_bias_aware_things_matrix(load_things_train_triplets(), n_objects=1854)
    similarity = M.similarity

    val_triplets = load_triplets_file(DEFAULT_VAL_FILE)
    spose = np.maximum(np.loadtxt(DEFAULT_SPOSE), 0.0)
    log.info("val_triplets=%d  spose=%s", len(val_triplets), spose.shape)

    log.info("Fitting SRF rank=%d for %d seeds ...", rank, len(seeds))
    t_all = time.time()
    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_fit_and_score)(similarity, val_triplets, rank, seed) for seed in seeds
    )
    log.info("All %d seeds done in %.1fs", len(seeds), time.time() - t_all)

    final_df = pd.DataFrame(rows)
    final_df["eval_split"] = "validationset"
    final_df.to_csv(OUTPUT_DIR / "final_results.csv", index=False)

    spose_acc = triplet_accuracy(spose, val_triplets)
    srf_acc = final_df["val_acc"].astype(float)
    summary_df = pd.DataFrame([
        {"model": "SRF", "rank": rank,
         "accuracy_pct": float(srf_acc.mean() * 100),
         "accuracy_std_pct": float(srf_acc.std(ddof=1) * 100) if len(srf_acc) > 1 else 0.0,
         "noise_ceiling_pct": float(100.0 * srf_acc.mean() / (NOISE_CEILING_PCT / 100.0))},
        {"model": "SPoSE", "rank": spose.shape[1],
         "accuracy_pct": float(spose_acc * 100),
         "accuracy_std_pct": 0.0,
         "noise_ceiling_pct": float(100.0 * spose_acc / (NOISE_CEILING_PCT / 100.0))},
        {"model": "Noise ceiling", "rank": np.nan,
         "accuracy_pct": NOISE_CEILING_PCT,
         "accuracy_std_pct": NOISE_CEILING_STD_PCT,
         "noise_ceiling_pct": 100.0},
    ])
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    chance_corrected_pct = float(
        100.0
        * (srf_acc.mean() * 100 - CHANCE_PCT)
        / (NOISE_CEILING_PCT - CHANCE_PCT)
    )

    (OUTPUT_DIR / "selection.json").write_text(json.dumps({
        "rank": rank,
        "rank_source": rank_source,
        "rsm": {
            "kind": "bias_aware_fisher",
            "alpha": ALPHA,
            "weight_mode": WEIGHT_MODE,
            "prior_strength": PRIOR_STRENGTH,
            "source": "build_similarity-style normal RSM, then Fisher-weighted "
                      "item-bias shrinkage; see ../README.md.",
        },
        "srf_kwargs": SRF_KWARGS,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "noise_ceiling_pct": NOISE_CEILING_PCT,
        "noise_ceiling_std_pct": NOISE_CEILING_STD_PCT,
        "chance_pct": CHANCE_PCT,
        "chance_corrected_ceiling_pct": chance_corrected_pct,
    }, indent=2))

    _plot_bar(summary_df, OUTPUT_DIR, rank)

    log.info("\n=== summary ===")
    log.info("%s", summary_df.to_string(index=False))
    log.info("chance-corrected: %.2f%% of (ceiling - chance)", chance_corrected_pct)
    log.info("Wrote %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
