from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.analyses.things_behavior.summary_stats import (
    NOISE_CEILING_PCT,
    NOISE_CEILING_STD_PCT,
)
from src.colors import GRAY, GRAY_DARK, GRAY_LIGHT, SAND, TEAL, soft
from src.similarity.triplet_rsm import (
    build_bias_aware_triplet_matrix,
    fit_triplet_bias_prior,
)
from src.triplet_prediction import (
    SimplePriorConfig,
    fit_and_score_srf,
    load_optimal_rank,
    load_triplets_file,
    triplet_accuracy,
)
from src.utils.figure_theme import create_figure, despine, save_figure

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "triplets47_simple_prior"
CHANCE_PCT = 100.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "things" / "triplets_47")
    parser.add_argument("--spose-file", type=Path, default=PROJECT_ROOT / "data" / "things" / "spose_embedding_66d.txt")
    parser.add_argument("--n-objects", type=int, default=1854)
    parser.add_argument(
        "--rank-kind",
        type=str,
        default="argmin",
        choices=["argmin", "one_se"],
        help="Which rank to load from the primary CV variant.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tune-train-file", type=str, default="train_90.txt")
    parser.add_argument("--tune-eval-file", type=str, default="test_10.txt")
    parser.add_argument("--final-train-file", type=str, default="trainset.txt")
    parser.add_argument("--final-eval-file", type=str, default="validationset.txt")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weight-modes", nargs="+", choices=["shown", "fisher"], default=["shown", "fisher"])
    parser.add_argument("--prior-strengths", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--rhos", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--inits", nargs="+", choices=["random_sqrt", "random"], default=["random_sqrt", "random"])
    parser.add_argument("--tune-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tune-max-outer", type=int, default=30)
    parser.add_argument("--tune-max-inner", type=int, default=30)
    parser.add_argument("--final-max-outer", type=int, default=100)
    parser.add_argument("--final-max-inner", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def build_similarity_cache(
    triplets: np.ndarray,
    n_objects: int,
    alpha: float,
    weight_modes: list[str],
    prior_strengths: list[float],
) -> dict[tuple[str, float], np.ndarray]:
    cache: dict[tuple[str, float], np.ndarray] = {}
    for weight_mode in weight_modes:
        wins, trials, prior = fit_triplet_bias_prior(
            n_objects=n_objects,
            triplets=triplets,
            alpha=alpha,
            weight_mode=weight_mode,
        )
        for prior_strength in prior_strengths:
            similarity = build_bias_aware_triplet_matrix(
                wins=wins,
                trials=trials,
                prior=prior,
                alpha=alpha,
                prior_strength=prior_strength,
                fill_missing_with_prior=True,
            )
            np.fill_diagonal(similarity, np.nan)
            cache[(weight_mode, prior_strength)] = similarity
    return cache


def tune_simple_prior(
    rank: int,
    similarity_cache: dict[tuple[str, float], np.ndarray],
    eval_triplets: np.ndarray,
    weight_modes: list[str],
    prior_strengths: list[float],
    rhos: list[float],
    inits: list[str],
    seeds: list[int],
    max_outer: int,
    max_inner: int,
    tol: float,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SimplePriorConfig]:
    tasks: list = []
    for weight_mode in weight_modes:
        for prior_strength in prior_strengths:
            similarity = similarity_cache[(weight_mode, prior_strength)]
            for rho in rhos:
                for init in inits:
                    config = SimplePriorConfig(
                        weight_mode=weight_mode,
                        prior_strength=prior_strength,
                        rho=rho,
                        init=init,
                    )
                    for seed in seeds:
                        tasks.append(
                            delayed(fit_and_score_srf)(
                                similarity=similarity,
                                triplets=eval_triplets,
                                rank=rank,
                                config=config,
                                seed=seed,
                                max_outer=max_outer,
                                max_inner=max_inner,
                                tol=tol,
                            )
                        )

    log.info("Tuning %d rank-%d fits on %s", len(tasks), rank, "test_10")
    rows = Parallel(n_jobs=n_jobs, verbose=10)(tasks)
    tune_df = pd.DataFrame(rows)
    summary_df = (
        tune_df.groupby(["weight_mode", "prior_strength", "rho", "init"], as_index=False)
        .agg(mean_val_acc=("val_acc", "mean"), std_val_acc=("val_acc", "std"), best_val_acc=("val_acc", "max"))
        .sort_values(["mean_val_acc", "std_val_acc", "best_val_acc"], ascending=[False, True, False])
    )
    best = summary_df.iloc[0]
    best_config = SimplePriorConfig(
        weight_mode=str(best["weight_mode"]),
        prior_strength=float(best["prior_strength"]),
        rho=float(best["rho"]),
        init=str(best["init"]),
    )
    return tune_df, summary_df, best_config


def run_final_evaluation(
    rank: int,
    config: SimplePriorConfig,
    similarity: np.ndarray,
    eval_triplets: np.ndarray,
    spose_embedding: np.ndarray,
    seeds: list[int],
    max_outer: int,
    max_inner: int,
    tol: float,
    n_jobs: int,
) -> pd.DataFrame:
    tasks = [
        delayed(fit_and_score_srf)(
            similarity=similarity,
            triplets=eval_triplets,
            rank=rank,
            config=config,
            seed=seed,
            max_outer=max_outer,
            max_inner=max_inner,
            tol=tol,
        )
        for seed in seeds
    ]
    log.info("Running %d final rank-%d fits on validationset", len(tasks), rank)
    rows = Parallel(n_jobs=n_jobs, verbose=10)(tasks)
    final_df = pd.DataFrame(rows)
    final_df.insert(0, "model", "SRF")
    final_df["rank"] = rank
    final_df["eval_split"] = "validationset"

    spose_row = pd.DataFrame(
        [
            {
                "model": "SPoSE",
                "weight_mode": "baseline",
                "prior_strength": np.nan,
                "rho": np.nan,
                "init": "embedding",
                "seed": 0,
                "val_acc": triplet_accuracy(spose_embedding, eval_triplets),
                "rank": spose_embedding.shape[1],
                "eval_split": "validationset",
            }
        ]
    )
    return pd.concat([final_df, spose_row], ignore_index=True)


def build_summary(final_df: pd.DataFrame, rank: int) -> pd.DataFrame:
    srf = final_df[final_df["model"] == "SRF"]["val_acc"]
    spose = float(final_df.loc[final_df["model"] == "SPoSE", "val_acc"].iloc[0])

    rows = [
        {
            "model": "SRF",
            "label": f"SRF\n{rank}d",
            "rank": rank,
            "accuracy_pct": float(srf.mean() * 100.0),
            "accuracy_std_pct": float(srf.std(ddof=1) * 100.0),
            "noise_ceiling_pct": float(100.0 * srf.mean() / (NOISE_CEILING_PCT / 100.0)),
        },
        {
            "model": "SPoSE",
            "label": "SPoSE\n66d",
            "rank": 66,
            "accuracy_pct": float(spose * 100.0),
            "accuracy_std_pct": 0.0,
            "noise_ceiling_pct": float(100.0 * spose / (NOISE_CEILING_PCT / 100.0)),
        },
        {
            "model": "Noise ceiling",
            "label": "Noise ceiling",
            "rank": np.nan,
            "accuracy_pct": NOISE_CEILING_PCT,
            "accuracy_std_pct": NOISE_CEILING_STD_PCT,
            "noise_ceiling_pct": 100.0,
        },
    ]
    return pd.DataFrame(rows)


def plot_final_bar(summary_df: pd.DataFrame, output_dir: Path) -> None:
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
    ax.set_title("Behavioural prediction")
    despine(ax)
    ax.text(
        1.48,
        CHANCE_PCT + 0.35,
        "chance",
        color=chance_color,
        ha="right",
        va="bottom",
        fontsize=8,
    )
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
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            31.1,
            f"{int(row['rank'])}d",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
        )

    pdf_path = output_dir / "triplet_prediction_barplot"
    save_figure(fig, pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = load_optimal_rank("things_behavior", kind=args.rank_kind, project_root=PROJECT_ROOT)
    rank_source = str(
        PROJECT_ROOT
        / "experiments"
        / "datasets"
        / "dimensionality"
        / "outputs"
        / "things_behavior"
        / "cross_validation.json"
    )
    log.info("Using rank %d (%s) from dimensionality CV", rank, args.rank_kind)

    tune_train_triplets = load_triplets_file(args.data_dir / args.tune_train_file)
    tune_eval_triplets = load_triplets_file(args.data_dir / args.tune_eval_file)
    tune_similarity_cache = build_similarity_cache(
        triplets=tune_train_triplets,
        n_objects=args.n_objects,
        alpha=args.alpha,
        weight_modes=args.weight_modes,
        prior_strengths=args.prior_strengths,
    )

    tune_df, tune_summary_df, best_config = tune_simple_prior(
        rank=rank,
        similarity_cache=tune_similarity_cache,
        eval_triplets=tune_eval_triplets,
        weight_modes=args.weight_modes,
        prior_strengths=args.prior_strengths,
        rhos=args.rhos,
        inits=args.inits,
        seeds=args.tune_seeds,
        max_outer=args.tune_max_outer,
        max_inner=args.tune_max_inner,
        tol=args.tol,
        n_jobs=args.n_jobs,
    )

    final_train_triplets = load_triplets_file(args.data_dir / args.final_train_file)
    final_eval_triplets = load_triplets_file(args.data_dir / args.final_eval_file)
    final_similarity = build_similarity_cache(
        triplets=final_train_triplets,
        n_objects=args.n_objects,
        alpha=args.alpha,
        weight_modes=[best_config.weight_mode],
        prior_strengths=[best_config.prior_strength],
    )[(best_config.weight_mode, best_config.prior_strength)]
    spose_embedding = np.maximum(np.loadtxt(args.spose_file), 0.0)

    final_df = run_final_evaluation(
        rank=rank,
        config=best_config,
        similarity=final_similarity,
        eval_triplets=final_eval_triplets,
        spose_embedding=spose_embedding,
        seeds=args.final_seeds,
        max_outer=args.final_max_outer,
        max_inner=args.final_max_inner,
        tol=args.tol,
        n_jobs=args.n_jobs,
    )
    summary_df = build_summary(final_df=final_df, rank=rank)

    tune_df.to_csv(args.output_dir / "tuning_results.csv", index=False)
    tune_summary_df.to_csv(args.output_dir / "tuning_summary.csv", index=False)
    final_df.to_csv(args.output_dir / "final_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "rank_source": rank_source,
                "rank": rank,
                "selected_config": {
                    "weight_mode": best_config.weight_mode,
                    "prior_strength": best_config.prior_strength,
                    "rho": best_config.rho,
                    "init": best_config.init,
                },
                "noise_ceiling_pct": NOISE_CEILING_PCT,
                "noise_ceiling_std_pct": NOISE_CEILING_STD_PCT,
            },
            indent=2,
        )
    )
    plot_final_bar(summary_df=summary_df, output_dir=args.output_dir)

    log.info("\nBest config: %s", best_config)
    log.info("\nFinal summary:")
    log.info("%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
