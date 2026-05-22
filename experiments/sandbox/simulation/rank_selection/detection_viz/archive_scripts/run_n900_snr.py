"""Run rank detection simulation with n=900 - SNR effect only.

Fixed alpha=1.0, varying SNR.

Usage:
    poetry run python sandbox/rank_detection_viz/run_n900_snr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf.cross_validation import cross_val_score
from utils.simulation import simulation_dirichlet

OUTPUT_DIR = get_output_dir()

# Parameters
N = 900
TRUE_RANKS = [5, 10, 15, 20, 25, 30]
ALPHA = 1.0  # Fixed
SNRS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 5


def add_noise_snr(w: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add positive noise to achieve target SNR."""
    if snr >= 1.0:
        return w
    signal_var = np.var(w)
    noise_var = signal_var * (1 - snr) / snr
    noise = np.abs(rng.normal(0, np.sqrt(noise_var), w.shape))
    return w + noise


def run_single(n: int, k: int, alpha: float, snr: float, seed: int) -> dict:
    """Run single rank detection."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)

    if snr < 1.0:
        w = add_noise_snr(w, snr, rng)

    S = w @ w.T

    ranks = list(range(max(2, k - 8), min(k + 10, n // 2), 2))
    if k not in ranks:
        ranks.append(k)
    ranks = sorted(ranks)

    result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=5,
        estimate_sampling_fraction=True,
        verbose=0,
        n_jobs=1,
    )

    selected = result.best_params_["rank"]

    return {
        "n": n,
        "true_rank": k,
        "alpha": alpha,
        "snr": snr,
        "seed": seed,
        "selected_rank": selected,
        "abs_error": abs(selected - k),
        "signed_error": selected - k,
        "is_correct": selected == k,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = [
        (N, k, ALPHA, snr, seed)
        for k in TRUE_RANKS
        for snr in SNRS
        for seed in range(N_SEEDS)
    ]

    n_conditions = len(conditions)
    print(f"Running {n_conditions} conditions with n={N}, alpha={ALPHA}...")
    print(f"True ranks: {TRUE_RANKS}")
    print(f"SNRs: {SNRS}")
    print(f"Seeds: {N_SEEDS}")

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, alpha, snr, seed)
        for n, k, alpha, snr, seed in conditions
    )

    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "rank_detection_n900_snr.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} results to {output_path}")

    # Quick summary
    print("\nMAE by SNR:")
    print(df.groupby("snr")["abs_error"].mean())


if __name__ == "__main__":
    main()
