"""THINGS low-data coherence analysis for rank selection.

Coherence iteration: for each candidate rank k, impute missing values using rank k,
estimate sampling bounds on the imputed matrix, run CV with those bounds, and check
if CV selects k (i.e., k == k'). Fixed points where k == k' are stable rank estimates.

Usage:
    ./scripts/submit experiments/things_behavior/coherence.py
    ./scripts/submit experiments/things_behavior/coherence.py percentages=[0.05,0.10]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

from experiments.bounds_missing import impute_similarity_matrix
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.cross_validation import cross_val_score

from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

log = logging.getLogger(__name__)


def subsample_triplets(
    triplets: np.ndarray, percentage: float, seed: int
) -> np.ndarray:
    if percentage >= 1.0:
        return triplets
    n_samples = int(len(triplets) * percentage)
    rng = np.random.default_rng(seed)
    return triplets[rng.choice(len(triplets), size=n_samples, replace=False)]


def run_candidate(
    S: np.ndarray,
    k: int,
    cv_ranks: list[int],
    impute_max_outer: int,
    impute_tol: float,
    cv_repeats: int,
    random_state: int,
) -> dict:
    S_imp = impute_similarity_matrix(
        S, rank=k, random_state=random_state, max_outer=impute_max_outer, tol=impute_tol
    )

    pmin, pmax, _ = estimate_sampling_bounds_ultra(
        S_imp, random_state=random_state, verbose=False
    )
    sf = 0.5 * (pmin + pmax)

    cv = cross_val_score(
        S,
        param_grid={"rank": cv_ranks},
        n_repeats=cv_repeats,
        sampling_fraction=sf,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
        missing_values=np.nan,
    )

    cv_df = cv.cv_results_
    mean_scores = cv_df.groupby("rank")["score"].mean()
    std_scores = cv_df.groupby("rank")["score"].std()
    cv_scores = {
        int(r): {"mean": float(mean_scores[r]), "std": float(std_scores[r])}
        for r in mean_scores.index
    }

    return {
        "k": int(k),
        "k_pred": int(cv.best_params_["rank"]),
        "pmin": float(pmin),
        "pmax": float(pmax),
        "sampling_fraction": float(sf),
        "cv_best_score": float(cv.best_score_),
        "cv_scores": cv_scores,
        "is_fixed_point": k == cv.best_params_["rank"],
    }


def select_recommended_rank(rows: list[dict]) -> int:
    fixed = [r for r in rows if r["is_fixed_point"]]
    if fixed:
        return int(fixed[0]["k"])
    return int(min(rows, key=lambda r: abs(r["k"] - r["k_pred"]))["k"])


def build_similarity_for_percentage(
    triplets: np.ndarray,
    percentage: float,
    n_items: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    triplets_sub = subsample_triplets(triplets, percentage, seed)
    S = compute_similarity_matrix_from_triplets(n_items, triplets_sub, alpha=0)
    n_pairs = n_items * (n_items - 1) // 2
    missing_pairs = int(np.isnan(S).sum() // 2)
    missing_frac = missing_pairs / n_pairs
    return S, len(triplets_sub), missing_frac


def run_candidate_for_percentage(
    triplets: np.ndarray,
    percentage: float,
    k: int,
    n_items: int,
    cv_ranks: list[int],
    impute_max_outer: int,
    impute_tol: float,
    cv_repeats: int,
    seed: int,
) -> dict:
    S, n_triplets, missing_frac = build_similarity_for_percentage(
        triplets, percentage, n_items, seed
    )
    result = run_candidate(
        S=S,
        k=k,
        cv_ranks=cv_ranks,
        impute_max_outer=impute_max_outer,
        impute_tol=impute_tol,
        cv_repeats=cv_repeats,
        random_state=seed,
    )
    result["percentage"] = percentage
    result["missing_fraction"] = missing_frac
    result["n_triplets"] = n_triplets
    return result


def run(cfg: DictConfig) -> None:
    log.info(f"Loading triplets from {cfg.dataset_path}")
    triplets, _ = load_triplets(Path(cfg.dataset_path), number=cfg.triplet_version)
    log.info(f"Total triplets: {len(triplets):,}")

    percentages = list(cfg.percentages)
    k_prime = list(cfg.k_prime)
    cv_ranks = list(cfg.cv_ranks)

    tasks = [(pct, k) for pct in percentages for k in k_prime]
    log.info(
        f"Running {len(tasks)} tasks: {len(percentages)} percentages × {len(k_prime)} ranks (n_jobs={cfg.n_jobs})"
    )

    all_rows = Parallel(n_jobs=cfg.n_jobs)(
        delayed(run_candidate_for_percentage)(
            triplets=triplets,
            percentage=pct,
            k=k,
            n_items=cfg.n_items,
            cv_ranks=cv_ranks,
            impute_max_outer=cfg.impute_max_outer,
            impute_tol=cfg.impute_tol,
            cv_repeats=cfg.cv_repeats,
            seed=cfg.seed,
        )
        for pct, k in tasks
    )

    out_dir = Path.cwd()

    df = pd.DataFrame(all_rows)
    cols = [
        "percentage",
        "k",
        "k_pred",
        "is_fixed_point",
        "pmin",
        "pmax",
        "sampling_fraction",
        "cv_best_score",
        "missing_fraction",
        "n_triplets",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["percentage", "k"])
    df.to_csv(out_dir / "results.csv", index=False)

    summary = []
    for pct in percentages:
        pct_rows = [r for r in all_rows if r["percentage"] == pct]
        fixed_points = [r["k"] for r in pct_rows if r["is_fixed_point"]]
        recommended = select_recommended_rank(pct_rows)
        summary.append(
            {
                "percentage": pct,
                "missing_fraction": pct_rows[0]["missing_fraction"],
                "n_triplets": pct_rows[0]["n_triplets"],
                "recommended_rank": recommended,
                "fixed_points": fixed_points,
            }
        )
        log.info(
            f"[{pct*100:.0f}%] missing={pct_rows[0]['missing_fraction']:.1%}, fixed_points={fixed_points}, recommended={recommended}"
        )

    results = {
        "config": {
            "percentages": percentages,
            "k_prime": k_prime,
            "cv_ranks": cv_ranks,
            "cv_repeats": cfg.cv_repeats,
            "imputation": {"max_outer": cfg.impute_max_outer, "tol": cfg.impute_tol},
            "n_items": cfg.n_items,
            "triplet_version": cfg.triplet_version,
            "seed": cfg.seed,
        },
        "summary": summary,
        "results": all_rows,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Saved results to {out_dir}")
