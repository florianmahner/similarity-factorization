"""Test BSUM speed with different BLAS thread counts."""

import logging
import os
import time
from pathlib import Path

# CRITICAL: must be set BEFORE numpy/scipy/pysrf import
N_THREADS = int(os.environ.get("BLAS_TEST_THREADS", "8"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

import numpy as np
from pysrf._bsum import update_w_blas_blocked
from pysrf.model import update_v_, _initialize_w, _get_observed_mask

from src.utils import get_output_dir
from src.utils.helpers import compute_similarity_matrix_from_triplets
from src.utils.io import load_triplets

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()
DATA_DIR = Path("data/things")


def main():
    log.info("BLAS threads requested: %d", N_THREADS)
    log.info("OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s MKL_NUM_THREADS=%s",
             os.environ.get("OMP_NUM_THREADS"),
             os.environ.get("OPENBLAS_NUM_THREADS"),
             os.environ.get("MKL_NUM_THREADS"))

    train, _ = load_triplets(DATA_DIR)
    s = compute_similarity_matrix_from_triplets(1854, train, alpha=0.0)
    n = 1854
    k = 45
    rho = 3.0
    max_outer = 5
    max_inner = 50

    observed_mask = _get_observed_mask(s, np.nan)
    s = np.where(observed_mask, s, 0.0)
    s = (s + s.T) / 2

    w = _initialize_w(s, k, "random_sqrt", 0)
    lam = np.zeros_like(s)
    v = s.copy()
    x_hat = np.empty_like(s)
    np.dot(w, w.T, out=x_hat)
    target = np.empty_like(s)
    primal_residual = np.empty_like(s)

    bsum_times = []
    update_v_times = []
    matmul_times = []

    for outer in range(max_outer):
        target[:] = lam
        target /= rho
        target += v

        t = time.perf_counter()
        w = update_w_blas_blocked(target, w, max_iter=max_inner, tol=1e-4)
        bsum_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        np.dot(w, w.T, out=x_hat)
        matmul_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        update_v_(observed_mask, s, x_hat, lam, rho, None, None, v)
        update_v_times.append(time.perf_counter() - t)

        primal_residual[:] = v
        primal_residual -= x_hat
        target[:] = primal_residual
        target *= rho
        lam += target

    log.info("\n=== Mean times per outer iter (n=1854, k=45, max_inner=50) ===")
    log.info("  bsum:      %7.3f s", np.mean(bsum_times))
    log.info("  update_v:  %7.3f s", np.mean(update_v_times))
    log.info("  matmul:    %7.3f s", np.mean(matmul_times))


if __name__ == "__main__":
    main()
