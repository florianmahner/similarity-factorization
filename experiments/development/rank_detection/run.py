#!/usr/bin/env python3

"""Rank detection sweep with median plot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysrf import SRF
from pysrf.cross_validation import cross_val_score
from utils.helpers import add_positive_noise_with_snr
from utils.simulation import simulation

CV_REPEATS = 10
GRID_SPAN = 10


@dataclass(frozen=True)
class ExperimentConfig:
    true_rank: int
    snr: float
    seed: int


def build_configs(
    true_ranks: list[int], snrs: list[float], n_seeds: int
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(true_rank=rank, snr=snr, seed=seed)
        for rank in true_ranks
        for snr in snrs
        for seed in range(n_seeds)
    ]


def make_similarity(n: int, k: int, snr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation(n, k, rng=rng, primary_concentration=2.0, base_concentration=0.1)
    similarity = w @ w.T
    similarity = add_positive_noise_with_snr(similarity, snr, rng)
    similarity = (similarity + similarity.T) * 0.5
    return similarity


def rank_grid(true_rank: int) -> list[int]:
    lower = max(1, true_rank - GRID_SPAN)
    upper = true_rank + GRID_SPAN
    start = lower if lower % 5 == 0 else lower + (5 - lower % 5)
    grid = list(range(start, upper + 1, 5))
    grid.extend([lower, upper, true_rank])
    return sorted(set(rank for rank in grid if rank >= 1))


def evaluate_config(config: ExperimentConfig, n_observations: int) -> dict[str, object]:
    record: dict[str, object] = {
        "true_rank": config.true_rank,
        "snr": config.snr,
        "seed": config.seed,
        "status": "ok",
        "error": "",
    }
    candidate_ranks = rank_grid(config.true_rank)
    similarity = make_similarity(
        n_observations, config.true_rank, config.snr, config.seed
    )
    grid = cross_val_score(
        similarity,
        estimator=SRF(
            init="random_sqrt",
            random_state=config.seed,
            max_outer=30,
            max_inner=20,
        ),
        param_grid={"rank": candidate_ranks},
        n_repeats=CV_REPEATS,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        random_state=config.seed,
        verbose=0,
        n_jobs=-1,
        fit_final_estimator=False,
    )
    cv_df = grid.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean().sort_index()
    best_rank = int(mean_scores.idxmin())
    best_score = float(mean_scores.loc[best_rank])
    second_scores = mean_scores.drop(best_rank, errors="ignore")
    if not second_scores.empty:
        second_rank = int(second_scores.idxmin())
        second_score = float(second_scores.loc[second_rank])
    else:
        second_rank = best_rank
        second_score = best_score
    record.update(
        {
            "selected_rank": best_rank,
            "best_score": best_score,
            "second_rank": second_rank,
            "second_score": second_score,
            "score_gap": second_score - best_score,
            "abs_error": abs(best_rank - config.true_rank),
            "signed_error": best_rank - config.true_rank,
        }
    )
    return record


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    summary = (
        ok.groupby(["true_rank", "snr"])
        .agg(
            trials=("seed", "count"),
            accuracy=("is_correct", "mean"),
            median_selected_rank=("selected_rank", "median"),
            mean_selected_rank=("selected_rank", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_abs_error=("abs_error", "mean"),
            median_signed_error=("signed_error", "median"),
            score_gap_median=("score_gap", "median"),
        )
        .reset_index()
    )
    return summary


def save_results(
    run_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path
) -> Path:
    run_block = run_df.copy()
    run_block["kind"] = "run"
    if summary_df.empty:
        combined = run_block
    else:
        summary_block = summary_df.copy()
        summary_block["seed"] = np.nan
        summary_block["status"] = "summary"
        summary_block["error"] = ""
        summary_block["kind"] = "summary"
        combined = pd.concat([run_block, summary_block], ignore_index=True, sort=False)
    path = output_dir / "rank_detection_results.csv"
    combined.to_csv(path, index=False)
    return path


def plot_median_selected_rank(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    true_vals = sorted(summary_df["true_rank"].unique())
    ax.plot(true_vals, true_vals, linestyle="--", color="black", linewidth=1)
    for snr in sorted(summary_df["snr"].unique()):
        subset = summary_df[summary_df["snr"] == snr].sort_values("true_rank")
        ax.plot(
            subset["true_rank"],
            subset["median_selected_rank"],
            marker="o",
            label=f"snr {snr}",
        )
    ax.set_xlabel("True rank")
    ax.set_ylabel("Median selected rank")
    ax.set_title("Median selected rank across snr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "median_selected_rank.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank detection sweep")
    parser.add_argument(
        "--true-ranks", type=int, nargs="+", default=[3, 5, 10, 15, 20, 30]
    )
    parser.add_argument(
        "--snrs", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/preview"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_output_path(output: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return output


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_path(args.output, args.output_dir)
    configs = build_configs(args.true_ranks, args.snrs, args.seeds)
    records = []
    for config in configs:
        output = evaluate_config(config, args.n)
        output["is_correct"] = output["selected_rank"] == config.true_rank
        records.append(output)
    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(df)
    save_results(df, summary, output_dir)
    plot_median_selected_rank(summary, output_dir)


if __name__ == "__main__":
    main()
