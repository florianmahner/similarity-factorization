"""Rank detection via cross-validation for SRF.

This task evaluates the accuracy of rank selection using cross-validation
across different true ranks, noise levels (SNR), and random seeds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF
from pysrf.cross_validation import cross_val_score

from utils.helpers import add_positive_noise_with_snr
from utils.simulation import simulation

from ..lib.plotting import create_rank_detection_plot

log = logging.getLogger(__name__)


def _make_similarity(n: int, k: int, snr: float, seed: int) -> np.ndarray:
    """Generate synthetic similarity matrix with noise."""
    rng = np.random.default_rng(seed)
    w = simulation(n, k, rng=rng, primary_concentration=2.0, base_concentration=0.1)
    similarity = w @ w.T
    similarity = add_positive_noise_with_snr(similarity, snr, rng)
    similarity = (similarity + similarity.T) * 0.5
    return similarity


def _rank_grid(true_rank: int, grid_span: int) -> list[int]:
    """Generate candidate rank grid around true rank."""
    lower = max(1, true_rank - grid_span)
    upper = true_rank + grid_span
    start = lower if lower % 5 == 0 else lower + (5 - lower % 5)
    grid = list(range(start, upper + 1, 5))
    grid.extend([lower, upper, true_rank])
    return sorted(set(rank for rank in grid if rank >= 1))


def _evaluate_single_condition(
    n: int,
    true_rank: int,
    snr: float,
    seed: int,
    grid_span: int,
    cv_repeats: int,
    n_jobs_inner: int = 1,
) -> dict:
    """Run rank detection for a single condition."""
    # Generate data
    similarity = _make_similarity(n, true_rank, snr, seed)

    # Get candidate ranks
    candidate_ranks = _rank_grid(true_rank, grid_span)

    # Run cross-validation
    cv_results = cross_val_score(
        similarity,
        estimator=SRF(
            init="random_sqrt",
            random_state=seed,
            max_outer=30,
            max_inner=20,
        ),
        param_grid={"rank": candidate_ranks},
        n_repeats=cv_repeats,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        random_state=seed,
        verbose=0,
        n_jobs=n_jobs_inner,
        fit_final_estimator=False,
    )

    # Find best rank
    cv_df = cv_results.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean().sort_index()
    best_rank = int(mean_scores.idxmin())
    best_score = float(mean_scores.loc[best_rank])

    # Find second best
    second_scores = mean_scores.drop(best_rank, errors="ignore")
    if not second_scores.empty:
        second_rank = int(second_scores.idxmin())
        second_score = float(second_scores.loc[second_rank])
    else:
        second_rank = best_rank
        second_score = best_score

    return {
        "true_rank": true_rank,
        "snr": snr,
        "seed": seed,
        "selected_rank": best_rank,
        "best_score": best_score,
        "second_rank": second_rank,
        "second_score": second_score,
        "score_gap": second_score - best_score,
        "abs_error": abs(best_rank - true_rank),
        "signed_error": best_rank - true_rank,
        "is_correct": best_rank == true_rank,
    }


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds."""
    summary = (
        df.groupby(["true_rank", "snr"])
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


def run(cfg: DictConfig) -> None:
    """Run rank detection analysis task."""
    log.info(f"Running rank detection analysis: n={cfg.n}")
    log.info(f"True ranks: {list(cfg.true_ranks)}")
    log.info(f"SNR values: {list(cfg.snrs)}")
    log.info(f"Seeds per condition: {cfg.n_seeds}")

    # Set up output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Outputs: {output_dir}")

    # Build list of conditions
    conditions = [
        (true_rank, snr, seed)
        for true_rank in cfg.true_ranks
        for snr in cfg.snrs
        for seed in range(cfg.n_seeds)
    ]
    n_conditions = len(conditions)
    n_jobs = getattr(cfg, "n_jobs", -1)
    log.info(f"Running {n_conditions} conditions with n_jobs={n_jobs}...")

    # Run parameter sweep in parallel
    records = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_evaluate_single_condition)(
            cfg.n,
            true_rank,
            snr,
            seed,
            cfg.grid_span,
            cfg.cv_repeats,
            n_jobs_inner=1,
        )
        for true_rank, snr, seed in conditions
    )

    # Create results dataframe
    results_df = pd.DataFrame(records)

    # Compute summary statistics
    summary_df = _summarize_results(results_df)

    # Save results
    results_path = output_dir / "rank_detection_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Saved results to {results_path}")

    summary_path = output_dir / "rank_detection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved summary to {summary_path}")

    # Create plot
    log.info("Creating plot...")
    create_rank_detection_plot(results_df, output_dir)

    log.info("Rank detection analysis complete")
