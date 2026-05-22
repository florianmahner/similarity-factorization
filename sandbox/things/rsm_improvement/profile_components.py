"""Time individual SRF components directly to find the real bottleneck."""

import logging
import time
from pathlib import Path

import numpy as np
from pysrf._bsum import update_w_blas_blocked
from pysrf.model import update_v_

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")


def time_op(name, fn, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    log.info("  %-30s %7.3f s  (min over %d runs)", name, min(times), n_runs)
    return min(times)


def main():
    train, _ = load_triplets(DATA_DIR)
    s = compute_similarity_matrix_from_triplets(1854, train, alpha=0.0)
    n = 1854
    k = 45
    rho = 3.0

    # Setup state matching what _fit_missing_data uses
    rng = np.random.default_rng(0)
    w = rng.random((n, k)).astype(np.float64) * np.sqrt(s.mean() / k)
    observed_mask = ~np.isnan(s)
    s_clean = np.where(observed_mask, s, 0.0)
    s_clean = (s_clean + s_clean.T) / 2  # ensure symmetric
    observed_mask = observed_mask & observed_mask.T

    x_hat = w @ w.T
    lam = np.zeros_like(s_clean)
    v = s_clean.copy()
    target = lam / rho + v

    log.info("Matrix: n=%d, k=%d, observed=%.1f%%", n, k, 100 * observed_mask.mean())
    log.info("")

    # 1. The Cython W update with max_inner=50
    log.info("--- Cython BSUM W update (the suspected bottleneck) ---")
    def w_update_50():
        return update_w_blas_blocked(target.copy(), w.copy(), max_iter=50, tol=1e-4)
    t_bsum_50 = time_op("BSUM max_inner=50", w_update_50)

    def w_update_200():
        return update_w_blas_blocked(target.copy(), w.copy(), max_iter=200, tol=1e-4)
    t_bsum_200 = time_op("BSUM max_inner=200", w_update_200)

    def w_update_20():
        return update_w_blas_blocked(target.copy(), w.copy(), max_iter=20, tol=1e-4)
    t_bsum_20 = time_op("BSUM max_inner=20", w_update_20)

    # 2. The matmul w @ w.T
    log.info("\n--- np.dot(w, w.T) ---")
    x_hat_buf = np.empty((n, n))
    def matmul():
        np.dot(w, w.T, out=x_hat_buf)
    t_matmul = time_op("w @ w.T", matmul)

    # 3. update_v_
    log.info("\n--- update_v_ ---")
    v_buf = np.empty_like(s_clean)
    def update_v():
        update_v_(observed_mask, s_clean, x_hat, lam, rho, None, None, v_buf)
    t_v = time_op("update_v_", update_v)

    # 4. The array ops in _fit_missing_data outer loop
    log.info("\n--- Array ops in outer loop ---")
    state = {
        "v_old_buf": np.empty_like(v),
        "target_buf": np.empty_like(v),
        "pr_buf": np.empty_like(v),
        "lam": lam.copy(),
        "v": v.copy(),
        "x_hat": x_hat.copy(),
    }

    def array_ops():
        state["v_old_buf"][:] = state["v"]
        np.copyto(state["target_buf"], state["lam"])
        np.divide(state["target_buf"], rho, out=state["target_buf"])
        np.add(state["target_buf"], state["v"], out=state["target_buf"])
        np.copyto(state["pr_buf"], state["v"])
        np.subtract(state["pr_buf"], state["x_hat"], out=state["pr_buf"])
        np.multiply(state["target_buf"], rho, out=state["target_buf"])
    t_array = time_op("array ops (v_old, target, pr)", array_ops)

    # 5. Real SRF fit for comparison
    log.info("\n--- Real SRF.fit() with max_outer=10, max_inner=50 ---")
    from pysrf import SRF
    def srf_fit():
        SRF(rank=k, random_state=0, max_outer=10, max_inner=50, tol=1e-4, verbose=0).fit(s)
    t_srf = time_op("SRF.fit", srf_fit, n_runs=2)
    log.info("  -> per outer iter: %.3f s", t_srf / 10)

    log.info("")
    log.info("=== Per outer iteration ===")
    log.info("BSUM (max_inner=200):     %7.3f s", t_bsum_200)
    log.info("BSUM (max_inner=50):      %7.3f s", t_bsum_50)
    log.info("BSUM (max_inner=20):      %7.3f s", t_bsum_20)
    log.info("w @ w.T matmul:           %7.3f s", t_matmul)
    log.info("update_v_:                %7.3f s", t_v)
    log.info("array ops:                %7.3f s", t_array)

    log.info("")
    log.info("=== Total per outer iter ===")
    log.info("With max_inner=200:       %7.3f s  (BSUM is %.0f%%)",
             t_bsum_200 + t_matmul + t_v + t_array,
             100 * t_bsum_200 / (t_bsum_200 + t_matmul + t_v + t_array))
    log.info("With max_inner=50:        %7.3f s  (BSUM is %.0f%%)",
             t_bsum_50 + t_matmul + t_v + t_array,
             100 * t_bsum_50 / (t_bsum_50 + t_matmul + t_v + t_array))
    log.info("With max_inner=20:        %7.3f s  (BSUM is %.0f%%)",
             t_bsum_20 + t_matmul + t_v + t_array,
             100 * t_bsum_20 / (t_bsum_20 + t_matmul + t_v + t_array))


if __name__ == "__main__":
    main()
