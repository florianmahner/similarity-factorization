"""Coherence kappa rank detection on simulated data -- full grid.

Full grid of true_rank x alpha x SNR to show rank recovery across all conditions.
n=300, 3 seeds per condition. Pure eigenspace coherence, no SRF fitting.
"""

import logging
import time

import numpy as np
import pandas as pd

from src.coherence import compute_coherence
from src.coherence.analysis import estimate_kappa, kappa_changepoint
from src.utils import get_output_dir
from src.utils.helpers import add_positive_noise_with_snr
from src.utils.simulation import simulation_dirichlet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUT_DIR = get_output_dir()

N = 300
N_SEEDS = 3
TRUE_RANKS = [5, 10, 15, 20, 25, 30, 40]
ALPHAS = [0.1, 0.5, 2.0, 5.0]
SNRS = [0.4, 0.7, 1.0]


def _make_similarity(n: int, k: int, alpha: float, snr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    if snr < 1.0:
        w = add_positive_noise_with_snr(w, snr, rng)
    return w @ w.T


def _coherence_kappa_rank(similarity: np.ndarray, max_k: int) -> int:
    k_list = list(range(1, max_k + 1))
    p_list = np.linspace(0.05, 0.95, 20)

    result = compute_coherence(
        similarity,
        k_list=k_list,
        p_list=p_list,
        b=50,
        b_null=0,
        compute_null=False,
        use_baseline_correction=False,
        show_progress=False,
        n_jobs=-1,
        random_state=42,
    )

    x_median = result["diagnostics"]["x_median"]
    k_arr = result["k_list"]
    kappa, _ = estimate_kappa(x_median, result["p"], hi_band_quantile=0.85)
    k_cut, _ = kappa_changepoint(kappa, k_arr)
    return k_cut


def main():
    conditions = [
        (k, alpha, snr)
        for k in TRUE_RANKS
        for alpha in ALPHAS
        for snr in SNRS
    ]
    total = len(conditions) * N_SEEDS
    log.info(f"Running {len(conditions)} conditions x {N_SEEDS} seeds = {total} runs, n={N}")

    records = []
    idx = 0

    for true_rank, alpha, snr in conditions:
        for seed in range(N_SEEDS):
            idx += 1
            t0 = time.time()

            max_k = min(true_rank + 20, N // 2)
            similarity = _make_similarity(N, true_rank, alpha, snr, seed)
            k_star = _coherence_kappa_rank(similarity, max_k)
            elapsed = time.time() - t0

            error = k_star - true_rank
            log.info(
                f"[{idx}/{total}] k={true_rank}, a={alpha}, snr={snr}, s={seed} "
                f"-> k*={k_star}, err={error:+d}, {elapsed:.1f}s"
            )

            records.append({
                "true_rank": true_rank,
                "alpha": alpha,
                "snr": snr,
                "seed": seed,
                "k_kappa": k_star,
                "error": error,
                "abs_error": abs(error),
                "correct": k_star == true_rank,
                "runtime_sec": round(elapsed, 1),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "coherence_rank_detection.csv", index=False)
    log.info(f"Results saved to {OUTPUT_DIR / 'coherence_rank_detection.csv'}")

    # Summary
    print("\n" + "=" * 70)
    g = df.groupby(["alpha", "snr"]).agg(
        accuracy=("correct", "mean"),
        mean_abs_error=("abs_error", "mean"),
    ).reset_index()
    print(g.to_string(index=False))
    print(f"\nOverall accuracy: {df['correct'].mean():.1%}")
    print(f"Overall MAE: {df['abs_error'].mean():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
