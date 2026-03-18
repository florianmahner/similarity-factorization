"""Rank selection benchmark: kappa coherence, BIC, elbow, and variance-90 across a full grid.

Compares 4 rank selection methods on synthetic similarity matrices (n=300) across
a grid of true ranks, sparsity (alpha), and SNR. Results are saved as a single CSV.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kneed import KneeLocator
from pysrf import SRF

from src.coherence import (
    _estimate_kappa_hat as estimate_kappa,
    compute_incremental_coherence_multi_k_eig_anisotropic as compute_coherence,
    kappa_changepoint,
)
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# =============================================================================
# Grid constants
# =============================================================================

N = 300
N_SEEDS = 5
TRUE_RANKS = [5, 10, 15, 20, 25, 30, 35, 40]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SNRS = [0.3, 0.5, 0.7, 1.0]

OUTPUT_DIR = Path("outputs/experiments/simulation/kappa_rank_detection")

# =============================================================================
# Shared data generation
# =============================================================================


def _make_similarity(n: int, k: int, alpha: float, snr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)
    return w @ w.T


# =============================================================================
# Rank selection methods
# =============================================================================


def _select_kappa(similarity: np.ndarray, true_rank: int) -> int:
    max_k = min(true_rank + 30, N // 2)
    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, 20)

    result = compute_coherence(
        similarity,
        k_list=k_list,
        p_list=p_list,
        B=50,
        B_null=0,
        compute_null=False,
        use_baseline_correction=False,
        show_progress=False,
        n_jobs=1,
        random_state=42,
        visualize=False,
    )

    x_median = result["diagnostics"]["x_median"]
    kappa, _ = estimate_kappa(x_median, result["p"], hi_band_quantile=0.85)
    k_cut, _ = kappa_changepoint(kappa, result["k_list"])
    return int(k_cut)


def _select_bic(similarity: np.ndarray, true_rank: int, seed: int) -> int:
    n = similarity.shape[0]
    n_obs = n * n
    lower = max(2, true_rank - 15)
    upper = true_rank + 15
    candidate_ranks = list(range(lower, upper + 1, 2))
    if true_rank not in candidate_ranks:
        candidate_ranks.append(true_rank)
    candidate_ranks = sorted(set(candidate_ranks))

    best_rank = candidate_ranks[0]
    best_bic = np.inf

    for rank in candidate_ranks:
        model = SRF(rank=rank, random_state=seed, max_outer=100, max_inner=30)
        model.fit(similarity)
        recon = model.reconstruct()
        mse = np.mean((similarity - recon) ** 2)
        n_params = n * rank
        bic = n_obs * np.log(mse + 1e-10) + n_params * np.log(n_obs)
        if bic < best_bic:
            best_bic = bic
            best_rank = rank

    return best_rank


def _select_elbow(eigvals: np.ndarray) -> int:
    x = np.arange(1, len(eigvals) + 1)
    kn = KneeLocator(x, eigvals, curve="convex", direction="decreasing")
    return int(kn.knee) if kn.knee is not None else 1


def _select_var90(eigvals: np.ndarray, threshold: float = 0.90) -> int:
    total = np.sum(eigvals ** 2)
    cumvar = np.cumsum(eigvals ** 2) / total
    idx = np.searchsorted(cumvar, threshold)
    return int(idx + 1)


# =============================================================================
# Per-condition runner
# =============================================================================


def _run_one(true_rank: int, alpha: float, snr: float, seed: int) -> dict:
    similarity = _make_similarity(N, true_rank, alpha, snr, seed)

    eigvals = np.linalg.eigvalsh(similarity)[::-1]
    eigvals = eigvals[eigvals > 0]

    rank_kappa = _select_kappa(similarity, true_rank)
    rank_bic = _select_bic(similarity, true_rank, seed)
    rank_elbow = _select_elbow(eigvals)
    rank_var90 = _select_var90(eigvals)

    return {
        "true_rank": true_rank,
        "alpha": alpha,
        "snr": snr,
        "seed": seed,
        "rank_kappa": rank_kappa,
        "rank_bic": rank_bic,
        "rank_elbow": rank_elbow,
        "rank_var90": rank_var90,
        "error_kappa": abs(rank_kappa - true_rank),
        "error_bic": abs(rank_bic - true_rank),
        "error_elbow": abs(rank_elbow - true_rank),
        "error_var90": abs(rank_var90 - true_rank),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "kappa_rank_detection.csv"

    conditions = [
        (true_rank, alpha, snr, seed)
        for true_rank in TRUE_RANKS
        for alpha in ALPHAS
        for snr in SNRS
        for seed in range(N_SEEDS)
    ]

    n_total = len(conditions)
    log.info(
        f"Grid: {len(TRUE_RANKS)} ranks x {len(ALPHAS)} alphas x {len(SNRS)} SNRs x "
        f"{N_SEEDS} seeds = {n_total} conditions, n={N}"
    )
    log.info(f"Output: {csv_path}")

    records = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_one)(true_rank, alpha, snr, seed)
        for true_rank, alpha, snr, seed in conditions
    )

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
