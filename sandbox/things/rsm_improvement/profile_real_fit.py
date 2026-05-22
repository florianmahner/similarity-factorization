"""Instrument an actual SRF outer loop to find the real bottleneck."""

import logging
import time
from collections import defaultdict
from pathlib import Path

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
    train, _ = load_triplets(DATA_DIR)
    s = compute_similarity_matrix_from_triplets(1854, train, alpha=0.0)
    n = 1854
    k = 45
    rho = 3.0
    max_outer = 10
    max_inner = 50

    observed_mask = _get_observed_mask(s, np.nan)
    s = np.where(observed_mask, s, 0.0)
    s = (s + s.T) / 2

    log.info("Matrix: n=%d, k=%d, observed=%.3f%%", n, k, 100 * observed_mask.mean())

    w = _initialize_w(s, k, "random_sqrt", 0)
    lam = np.zeros_like(s)
    v = s.copy()
    x_hat = np.empty_like(s)
    np.dot(w, w.T, out=x_hat)
    v_old = np.empty_like(s)
    target = np.empty_like(s)
    primal_residual = np.empty_like(s)

    timings = defaultdict(list)

    for outer in range(max_outer):
        t = time.perf_counter()
        v_old[:] = v
        timings["v_old_copy"].append(time.perf_counter() - t)

        t = time.perf_counter()
        target[:] = lam
        target /= rho
        target += v
        timings["build_target"].append(time.perf_counter() - t)

        t = time.perf_counter()
        w = update_w_blas_blocked(target, w, max_iter=max_inner, tol=1e-4)
        timings["bsum"].append(time.perf_counter() - t)

        t = time.perf_counter()
        np.dot(w, w.T, out=x_hat)
        timings["matmul"].append(time.perf_counter() - t)

        t = time.perf_counter()
        update_v_(observed_mask, s, x_hat, lam, rho, None, None, v)
        timings["update_v"].append(time.perf_counter() - t)

        t = time.perf_counter()
        primal_residual[:] = v
        primal_residual -= x_hat
        target[:] = primal_residual
        target *= rho
        lam += target
        timings["dual_update"].append(time.perf_counter() - t)

        log.info("outer=%d  bsum=%.3f  update_v=%.3f  matmul=%.3f  total~=%.3f",
                 outer,
                 timings["bsum"][-1],
                 timings["update_v"][-1],
                 timings["matmul"][-1],
                 sum(timings[k][-1] for k in timings))

    log.info("\n=== Mean per outer iter (over %d iters) ===", max_outer)
    total = 0
    for name in ["v_old_copy", "build_target", "bsum", "matmul", "update_v", "dual_update"]:
        mean_t = np.mean(timings[name])
        total += mean_t
        log.info("  %-20s %7.3f s", name, mean_t)
    log.info("  %-20s %7.3f s", "TOTAL", total)


if __name__ == "__main__":
    main()
