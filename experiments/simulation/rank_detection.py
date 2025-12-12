"""Rank detection via cross-validation for SRF.

This task evaluates the accuracy of rank selection using cross-validation
across different true ranks, varying either alpha or SNR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF, estimate_sampling_bounds_fast
from pysrf.cross_validation import EntryMaskSplit, fit_and_score

from utils.helpers import add_positive_noise_with_snr
from utils.simulation import simulation_dirichlet

log = logging.getLogger(__name__)


def _make_similarity(n: int, k: int, alpha: float, snr: float, seed: int) -> np.ndarray:
    """Generate synthetic similarity matrix with optional noise."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)

    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)

    similarity = w @ w.T
    return similarity


def _rank_grid(true_rank: int, grid_span: int, step: int = 2) -> list[int]:
    """Generate candidate rank grid around true rank."""
    lower = max(2, true_rank - grid_span)
    upper = true_rank + grid_span
    grid = list(range(lower, upper + 1, step))
    if true_rank not in grid:
        grid.append(true_rank)
    return sorted(set(grid))


def _run_cv_jobs(
    similarity: np.ndarray,
    candidate_ranks: list[int],
    p_mean: float,
    cv_repeats: int,
    seed: int,
    cv_n_jobs: int,
) -> list[dict]:
    """Execute CV fits for a single condition using a single Parallel pool."""
    splitter = EntryMaskSplit(
        n_repeats=cv_repeats,
        sampling_fraction=p_mean,
        random_state=seed,
        missing_values=np.nan,
    )
    masks = list(splitter.split(similarity))

    base_estimator = SRF(
        init="random_sqrt",
        random_state=seed,
        max_outer=100,
        max_inner=30,
    )

    tasks = (
        delayed(fit_and_score)(
            estimator=base_estimator,
            x=similarity,
            train_mask=train_mask,
            validation_mask=validation_mask,
            fit_params={"rank": rank},
            split_idx=split_idx,
        )
        for split_idx, (train_mask, validation_mask) in enumerate(masks)
        for rank in candidate_ranks
    )

    return Parallel(n_jobs=cv_n_jobs, verbose=10)(tasks)


def _evaluate_single_condition(
    n: int,
    true_rank: int,
    alpha: float,
    snr: float,
    seed: int,
    grid_span: int,
    cv_repeats: int,
    cv_n_jobs: int,
) -> dict[str, float | int | bool]:
    """Run rank detection for a single condition."""
    similarity = _make_similarity(n, true_rank, alpha, snr, seed)
    candidate_ranks = _rank_grid(true_rank, grid_span)

    # Estimate sampling bounds and use mean
    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2
    if p_mean <= 0 or p_mean >= 1:
        print(f"Warning: p_mean is out of bounds: {p_mean}")
        print(f"pmin: {pmin}, pmax: {pmax}")
        p_mean = 0.5  # fallback
        print(f"Fallback p_mean: {p_mean}")

    cv_results = _run_cv_jobs(
        similarity=similarity,
        candidate_ranks=candidate_ranks,
        p_mean=p_mean,
        cv_repeats=cv_repeats,
        seed=seed,
        cv_n_jobs=cv_n_jobs,
    )

    cv_df = pd.DataFrame(
        {
            "rank": res["params"]["rank"],
            "score": res["score"],
            "split": res["split"],
        }
        for res in cv_results
    )

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

    return {
        "true_rank": true_rank,
        "alpha": alpha,
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
        "pmin": pmin,
        "pmax": pmax,
        "p_mean": p_mean,
    }


def run(cfg: DictConfig) -> None:
    """Run rank detection analysis task."""
    alphas = list(cfg.get("alphas", [1.0]))
    snrs = list(cfg.get("snrs", [1.0]))

    log.info(f"Running rank detection analysis: n={cfg.n}")
    log.info(f"True ranks: {list(cfg.true_ranks)}")
    log.info(f"Alphas: {alphas}, SNRs: {snrs}")
    log.info(f"Seeds per condition: {cfg.n_seeds}")

    output_dir = Path(cfg.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Outputs: {output_dir}")

    conditions = [
        (true_rank, alpha, snr, seed)
        for true_rank in cfg.true_ranks
        for alpha in alphas
        for snr in snrs
        for seed in range(cfg.n_seeds)
    ]
    n_conditions = len(conditions)
    n_jobs = getattr(cfg, "n_jobs", -1)
    log.info(f"Running {n_conditions} conditions with cv n_jobs={n_jobs}...")

    records = []
    for idx, (true_rank, alpha, snr, seed) in enumerate(conditions, start=1):
        log.info(
            f"[{idx}/{n_conditions}] condition true_rank={true_rank}, alpha={alpha}, snr={snr}, seed={seed}"
        )
        record = _evaluate_single_condition(
            cfg.n,
            true_rank,
            alpha,
            snr,
            seed,
            cfg.grid_span,
            cfg.cv_repeats,
            cv_n_jobs=n_jobs,
        )
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = output_dir / "rank_detection.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {csv_path}")
