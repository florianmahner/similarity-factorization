"""Estimate optimal rank via cross-validation.

Usage:
    ./scripts/submit experiments/estimate_rank.py dataset=swow
    ./scripts/submit experiments/estimate_rank.py dataset=nsd subject_id=1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pysrf import cross_val_score
from pysrf.bounds import estimate_sampling_bounds_ultra

from similarity import build_similarity, get_rank_grid
from src.utils.figure_theme import CMAP, create_figure, despine, save_figure

log = logging.getLogger(__name__)


def _load_bounds(bounds_path: Path) -> tuple[float, float]:
    with open(bounds_path) as f:
        bounds = json.load(f)
    return bounds["pmin"], bounds["pmax"]


def _get_bounds_path(cfg: DictConfig, subject_id: int | None) -> Path:
    bounds_task = cfg.dataset.get("bounds_task", cfg.dataset.name)
    bounds_dir = Path(cfg.project_root) / "outputs" / "experiments" / "bounds" / bounds_task
    if subject_id is not None:
        new_path = bounds_dir / f"subject_{subject_id}" / "bounds.json"
        old_path = bounds_dir / f"subj{subject_id:02d}" / "bounds.json"
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
        return new_path
    return bounds_dir / "bounds.json"


def run(cfg: DictConfig) -> None:
    """Run cross-validation to estimate optimal rank."""
    subject_id = cfg.get("subject_id")
    output_dir = Path.cwd()

    # Load similarity matrix
    log.info(f"Building similarity matrix for {cfg.dataset.name}...")
    similarity = build_similarity(cfg.dataset, subject_id=subject_id)
    log.info(f"Similarity matrix shape: {similarity.shape}")

    # Get rank grid
    rank_grid = get_rank_grid(cfg.dataset)
    n_ranks = len(rank_grid)
    n_repeats = cfg.cv.n_repeats
    total_fits = n_ranks * n_repeats
    log.info(f"Rank grid: {rank_grid} ({n_ranks} ranks)")
    log.info(f"CV repeats: {n_repeats}")
    log.info(f"Total fits: {total_fits}")

    # Load or estimate bounds
    bounds_file = _get_bounds_path(cfg, subject_id)
    if bounds_file.exists():
        log.info(f"Loading pre-computed bounds from {bounds_file}")
        pmin, pmax = _load_bounds(bounds_file)
    else:
        log.info(f"Bounds not found at {bounds_file}, estimating...")
        pmin, pmax, _ = estimate_sampling_bounds_ultra(
            similarity, random_state=cfg.common.random_state, verbose=True
        )

    sampling_selection = cfg.cv.get("sampling_selection", "mean")
    sampling_fraction = {"min": pmin, "mean": 0.5 * (pmin + pmax), "max": pmax}[sampling_selection]
    log.info(f"Bounds: pmin={pmin:.4f}, pmax={pmax:.4f}")
    log.info(f"Using sampling_selection='{sampling_selection}' -> fraction={sampling_fraction:.4f}")

    # Cross-validation
    log.info(f"Running cross-validation...")
    cv_result = cross_val_score(
        similarity,
        param_grid={"rank": rank_grid},
        n_repeats=n_repeats,
        sampling_fraction=sampling_fraction,
        estimate_sampling_fraction=False,
        fit_final_estimator=False,
        random_state=cfg.common.random_state,
        n_jobs=cfg.common.n_jobs,
        verbose=1,
    )

    optimal_rank = cv_result.best_params_["rank"]
    log.info(f"Optimal rank: {optimal_rank} (CV score: {cv_result.best_score_:.6f})")

    # Aggregate CV scores
    cv_results_df = cv_result.cv_results_
    cv_scores = {
        int(rank): {"mean": float(group["score"].mean()), "std": float(group["score"].std())}
        for rank, group in cv_results_df.groupby("rank")
    }

    # Save results
    summary = {
        "dataset": cfg.dataset.name,
        "n_samples": int(similarity.shape[0]),
        "optimal_rank": int(optimal_rank),
        "best_cv_score": float(cv_result.best_score_),
        "n_ranks": n_ranks,
        "n_repeats": n_repeats,
        "total_fits": total_fits,
        "pmin": float(pmin),
        "pmax": float(pmax),
        "sampling_fraction": float(sampling_fraction),
        "cv_scores": cv_scores,
    }
    if subject_id is not None:
        summary["subject_id"] = int(subject_id)

    (output_dir / "rank_estimation.json").write_text(json.dumps(summary, indent=2))
    cv_results_df.to_csv(output_dir / "cv_results.csv", index=False)

    # Plot
    fig, ax = create_figure("single")
    ranks = sorted(cv_scores.keys())
    means = [cv_scores[r]["mean"] for r in ranks]
    stds = [cv_scores[r]["std"] for r in ranks]

    ax.errorbar(ranks, means, yerr=stds, fmt="o-", color=CMAP[1], capsize=3, markersize=4)
    ax.axvline(optimal_rank, color=CMAP[0], linestyle="--", linewidth=1.5, label=f"Optimal: {optimal_rank}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("CV Score (MSE)")
    ax.legend(fontsize=8)
    despine(ax)
    save_figure(fig, output_dir / "rank_estimation.pdf")
    plt.close(fig)

    log.info(f"Results saved to {output_dir}")
