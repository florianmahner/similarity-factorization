"""Rank selection benchmark: kappa coherence vs parallel analysis vs elbow.

Generates synthetic similarity matrices (S = WW^T, W ~ Dirichlet) across a grid
of true ranks, sparsity (alpha), and noise (SNR). Compares three rank selection
methods and saves detected ranks as CSV.

Usage:
    poetry run python experiments/analyses/simulation/rank_detection/run.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kneed import KneeLocator

from src.coherence import (
    _estimate_kappa_hat as estimate_kappa,
    compute_incremental_coherence_multi_k_eig_anisotropic as compute_coherence,
    kappa_changepoint,
)
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"

N = 300
N_SEEDS = 5
TRUE_RANKS = list(range(2, 42, 2))
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
SNRS = [0.3, 0.5, 0.7, 1.0]


def _make_similarity(n: int, k: int, alpha: float, snr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)
    return w @ w.T


def _select_kappa(similarity: np.ndarray, true_rank: int) -> int:
    max_k = min(true_rank + 30, N // 2)
    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, 20)

    result = compute_coherence(
        similarity, k_list=k_list, p_list=p_list,
        B=100, B_null=0, compute_null=False,
        use_baseline_correction=False, show_progress=False,
        n_jobs=1, random_state=42, visualize=False,
    )

    x_median = result["diagnostics"]["x_median"]
    kappa, _ = estimate_kappa(x_median, result["p"], hi_band_quantile=0.90)
    k_cut, _ = kappa_changepoint(kappa, result["k_list"])
    return int(k_cut)


def _select_parallel_analysis(eigvals: np.ndarray, n: int, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    k_max = len(eigvals)
    null_eigvals = np.zeros((100, k_max))
    for i in range(100):
        x_rand = rng.standard_normal((n, n))
        s_rand = x_rand @ x_rand.T / n
        null_eigvals[i] = np.linalg.eigvalsh(s_rand)[::-1][:k_max]
    threshold = np.percentile(null_eigvals, 95, axis=0)
    return max(1, int(np.sum(eigvals[:k_max] > threshold)))


def _select_elbow(eigvals: np.ndarray) -> int:
    x = np.arange(1, len(eigvals) + 1)
    kn = KneeLocator(x, eigvals, curve="convex", direction="decreasing")
    return int(kn.knee) if kn.knee is not None else 1


def _run_one(true_rank: int, alpha: float, snr: float, seed: int) -> dict:
    similarity = _make_similarity(N, true_rank, alpha, snr, seed)
    eigvals = np.linalg.eigvalsh(similarity)[::-1]
    eigvals = eigvals[eigvals > 0]

    rank_kappa = _select_kappa(similarity, true_rank)
    rank_parallel = _select_parallel_analysis(eigvals, N, seed=seed)
    rank_elbow = _select_elbow(eigvals)

    return {
        "true_rank": true_rank, "alpha": alpha, "snr": snr, "seed": seed,
        "rank_kappa": rank_kappa, "rank_parallel": rank_parallel, "rank_elbow": rank_elbow,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "rank_detection.csv"

    conditions = [
        (k, a, s, seed)
        for k in TRUE_RANKS for a in ALPHAS for s in SNRS for seed in range(N_SEEDS)
    ]

    log.info(f"Grid: {len(TRUE_RANKS)} ranks x {len(ALPHAS)} alphas x {len(SNRS)} SNRs x {N_SEEDS} seeds = {len(conditions)}")
    records = Parallel(n_jobs=-1, verbose=10)(delayed(_run_one)(*c) for c in conditions)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
