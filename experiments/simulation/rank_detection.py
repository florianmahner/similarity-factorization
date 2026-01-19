"""Rank detection via cross-validation for SRF.

This task evaluates the accuracy of rank selection using cross-validation
across different true ranks, varying either alpha or SNR.

Three experimental slices (all with fixed n):
1. Vary k: alpha=0.1, snr=1.0
2. Vary alpha: k=reference_rank, snr=1.0
3. Vary SNR: k=reference_rank, alpha=0.1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.cross_validation import EntryMaskSplit, fit_and_score

from utils.helpers import add_positive_noise_with_snr
from utils.simulation import simulation_dirichlet

log = logging.getLogger(__name__)


# =============================================================================
# Alternative rank selection methods
# =============================================================================


def _select_rank_kaiser(eigvals: np.ndarray) -> int:
    """Kaiser criterion: keep eigenvalues > mean(eigenvalues).

    Eigenvalue-based method - no model fitting required.
    """
    mean_eig = np.mean(eigvals)
    n_above = np.sum(eigvals > mean_eig)
    return max(1, int(n_above))


def _select_rank_variance(eigvals: np.ndarray, threshold: float = 0.90) -> int:
    """Select smallest rank where cumulative variance explained > threshold.

    Eigenvalue-based method - no model fitting required.
    """
    total_var = np.sum(eigvals ** 2)
    cumvar = np.cumsum(eigvals ** 2) / total_var
    idx = np.searchsorted(cumvar, threshold)
    return int(idx + 1)  # 1-indexed


def _fit_srf_all_ranks(
    similarity: np.ndarray,
    candidate_ranks: list[int],
    seed: int,
) -> dict[int, float]:
    """Fit SRF at each candidate rank on FULL data and return MSE per rank.

    This is used for BIC/AIC computation - fits on complete matrix (no masking).

    Returns:
        Dictionary mapping rank -> MSE (mean squared error of reconstruction)
    """
    mse_by_rank = {}
    for rank in candidate_ranks:
        model = SRF(rank=rank, random_state=seed, max_outer=100, max_inner=30)
        model.fit(similarity)
        recon = model.reconstruct()
        mse = np.mean((similarity - recon) ** 2)
        mse_by_rank[rank] = mse
    return mse_by_rank


def _select_rank_bic(mse_by_rank: dict[int, float], n: int) -> int:
    """Select rank using BIC (Bayesian Information Criterion).

    BIC = n_obs * log(MSE) + n_params * log(n_obs)

    Where:
        n_obs = n² (number of entries in similarity matrix)
        n_params = n * k (embedding matrix W has n*k parameters)

    Args:
        mse_by_rank: MSE from SRF reconstruction at each candidate rank
        n: Matrix dimension
    """
    n_obs = n * n  # Total entries in matrix

    best_rank, best_bic = min(mse_by_rank.keys()), np.inf
    for rank, mse in mse_by_rank.items():
        n_params = n * rank
        bic = n_obs * np.log(mse + 1e-10) + n_params * np.log(n_obs)
        if bic < best_bic:
            best_bic = bic
            best_rank = rank
    return best_rank


def _select_rank_aic(mse_by_rank: dict[int, float], n: int) -> int:
    """Select rank using AIC (Akaike Information Criterion).

    AIC = n_obs * log(MSE) + 2 * n_params

    Args:
        mse_by_rank: MSE from SRF reconstruction at each candidate rank
        n: Matrix dimension
    """
    n_obs = n * n

    best_rank, best_aic = min(mse_by_rank.keys()), np.inf
    for rank, mse in mse_by_rank.items():
        n_params = n * rank
        aic = n_obs * np.log(mse + 1e-10) + 2 * n_params
        if aic < best_aic:
            best_aic = aic
            best_rank = rank
    return best_rank


# =============================================================================
# Simulation and CV infrastructure
# =============================================================================


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
    """Run rank detection for a single condition.

    Structure:
    1. Eigenvalue-based methods (fast, no fitting): Kaiser, Variance
    2. Factorization-based IC (fit SRF on full data): BIC, AIC
    3. CV-based method (fit SRF on masked data): CV
    """
    similarity = _make_similarity(n, true_rank, alpha, snr, seed)
    candidate_ranks = _rank_grid(true_rank, grid_span)

    # -------------------------------------------------------------------------
    # 1. Eigenvalue-based methods (instant)
    # -------------------------------------------------------------------------
    eig = np.linalg.eigvalsh(similarity)[::-1]
    spectral_gap = float(eig[true_rank - 1] - eig[true_rank]) if true_rank < len(eig) else 0.0

    rank_kaiser = _select_rank_kaiser(eig)
    rank_variance = _select_rank_variance(eig, threshold=0.90)

    # -------------------------------------------------------------------------
    # 2. Factorization-based BIC/AIC (fit SRF on FULL data at each rank)
    # -------------------------------------------------------------------------
    mse_by_rank = _fit_srf_all_ranks(similarity, candidate_ranks, seed)
    rank_bic = _select_rank_bic(mse_by_rank, n)
    rank_aic = _select_rank_aic(mse_by_rank, n)

    # -------------------------------------------------------------------------
    # 3. CV-based method (fit SRF on MASKED data, score on held-out)
    # -------------------------------------------------------------------------
    pmin, pmax, _ = estimate_sampling_bounds_ultra(similarity)
    p_mean = (pmin + pmax) / 2
    if p_mean <= 0.1 or p_mean >= 0.95:
        log.warning(f"p_mean={p_mean:.3f} out of usable range, using fallback 0.5")
        p_mean = 0.5

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
    rank_cv = int(mean_scores.idxmin())
    best_score = float(mean_scores.loc[rank_cv])

    second_scores = mean_scores.drop(rank_cv, errors="ignore")
    if not second_scores.empty:
        second_rank = int(second_scores.idxmin())
        second_score = float(second_scores.loc[second_rank])
    else:
        second_rank = rank_cv
        second_score = best_score

    return {
        "true_rank": true_rank,
        "alpha": alpha,
        "snr": snr,
        "seed": seed,
        # CV method (main)
        "rank_cv": rank_cv,
        "error_cv": abs(rank_cv - true_rank),
        "correct_cv": rank_cv == true_rank,
        "best_score": best_score,
        "second_rank": second_rank,
        "second_score": second_score,
        "score_gap": second_score - best_score,
        # Eigenvalue-based methods
        "rank_kaiser": rank_kaiser,
        "error_kaiser": abs(rank_kaiser - true_rank),
        "correct_kaiser": rank_kaiser == true_rank,
        "rank_variance": rank_variance,
        "error_variance": abs(rank_variance - true_rank),
        "correct_variance": rank_variance == true_rank,
        # Factorization-based IC methods
        "rank_bic": rank_bic,
        "error_bic": abs(rank_bic - true_rank),
        "correct_bic": rank_bic == true_rank,
        "rank_aic": rank_aic,
        "error_aic": abs(rank_aic - true_rank),
        "correct_aic": rank_aic == true_rank,
        # Diagnostics
        "pmin": pmin,
        "pmax": pmax,
        "p_mean": p_mean,
        "spectral_gap": spectral_gap,
    }


def _build_condition_slices(cfg: DictConfig) -> list[tuple[int, float, float]]:
    """Build specific condition slices instead of full grid.

    Three slices:
    1. Vary k: alpha=0.1, snr=1.0
    2. Vary alpha: k=reference_rank, snr=1.0
    3. Vary SNR: k=reference_rank, alpha=0.1
    """
    true_ranks = list(cfg.true_ranks)
    alphas = list(cfg.get("alphas", [1.0]))
    snrs = list(cfg.get("snrs", [1.0]))
    reference_rank = cfg.get("reference_rank", 20)

    conditions = set()

    # Slice 1: Vary k (alpha=0.1, snr=1.0)
    for k in true_ranks:
        conditions.add((k, 0.1, 1.0))

    # Slice 2: Vary alpha (k=reference_rank, snr=1.0)
    for alpha in alphas:
        conditions.add((reference_rank, alpha, 1.0))

    # Slice 3: Vary SNR (k=reference_rank, alpha=0.1)
    for snr in snrs:
        conditions.add((reference_rank, 0.1, snr))

    return sorted(conditions)


def _load_completed_conditions(csv_path: Path) -> set[tuple[int, float, float, int]]:
    """Load already completed conditions from existing CSV."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        completed = set()
        for _, row in df.iterrows():
            completed.add((int(row["true_rank"]), float(row["alpha"]), float(row["snr"]), int(row["seed"])))
        return completed
    except Exception as e:
        log.warning(f"Could not load existing results: {e}")
        return set()


def run(cfg: DictConfig) -> None:
    """Run rank detection analysis task with incremental saving."""
    import os

    log.info(f"Running rank detection analysis: n={cfg.n}")
    log.info(f"True ranks: {list(cfg.true_ranks)}")
    log.info(f"Alphas: {list(cfg.get('alphas', [1.0]))}")
    log.info(f"SNRs: {list(cfg.get('snrs', [1.0]))}")
    log.info(f"Seeds per condition: {cfg.n_seeds}")

    output_dir = Path(cfg.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rank_detection.csv"
    log.info(f"Outputs: {output_dir}")

    # Ensure stable working directory for joblib workers
    os.chdir(cfg.project_root)

    # Build specific condition slices
    base_conditions = _build_condition_slices(cfg)
    log.info(f"Condition slices: {len(base_conditions)} unique (k, alpha, snr) combinations")

    conditions = [
        (true_rank, alpha, snr, seed)
        for true_rank, alpha, snr in base_conditions
        for seed in range(cfg.n_seeds)
    ]

    # Load existing results and skip completed conditions
    completed = _load_completed_conditions(csv_path)
    if completed:
        log.info(f"Resuming: {len(completed)} conditions already completed")
        conditions = [c for c in conditions if c not in completed]

    n_conditions = len(conditions)
    n_jobs = getattr(cfg, "n_jobs", -1)
    log.info(f"Running {n_conditions} remaining conditions with cv n_jobs={n_jobs}...")

    if n_conditions == 0:
        log.info("All conditions already completed!")
        return

    # Load existing records for appending
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        records = existing_df.to_dict("records")
    else:
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

        # Save incrementally after each condition
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        log.info(f"Saved progress: {len(records)} conditions completed")

    log.info(f"Done! Final results: {csv_path}")
