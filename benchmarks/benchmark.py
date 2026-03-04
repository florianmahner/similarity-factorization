"""Performance benchmark: all BSUM implementations.

Compares scalar, blas, and blocked (multiple block sizes) across
problem sizes. Reports wall-clock time per iteration and speedup vs scalar.

Supports multi-threaded BLAS via OMP_NUM_THREADS / OPENBLAS_NUM_THREADS.

Usage:
    python -u benchmark.py
    OMP_NUM_THREADS=4 python -u benchmark.py
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

RESULTS_CSV = Path(__file__).parent / "benchmark.csv"

CSV_COLUMNS = [
    "n",
    "rank",
    "max_iter",
    "impl",
    "threads",
    "total_s",
    "per_iter_s",
    "speedup_vs_scalar",
]


def get_num_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        val = os.environ.get(var)
        if val:
            return int(val)
    return 1


def load_implementations():
    from pysrf._bsum import update_w, update_w_blas, update_w_blas_blocked

    return {
        "scalar": update_w,
        "blas": update_w_blas,
        "blocked": update_w_blas_blocked,
    }


def make_data(n: int, rank: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    w_true = rng.rand(n, rank).astype(np.float64)
    m = w_true @ w_true.T
    m = (m + m.T) / 2
    w0 = rng.rand(n, rank).astype(np.float64) * np.sqrt(m.mean() / rank)
    return np.ascontiguousarray(m), np.ascontiguousarray(w0)


def time_impl(impl_fn, m, w0, max_iter, **kwargs):
    w0_copy = w0.copy()

    # Warmup
    impl_fn(m, w0.copy(), max_iter=1, tol=0.0, **kwargs)

    t0 = time.perf_counter()
    impl_fn(m, w0_copy, max_iter=max_iter, tol=0.0, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    impls = load_implementations()
    threads = get_num_threads()

    sizes = [(5000, 50), (10000, 50), (20000, 50)]
    max_iter = 5
    blocked_sizes = [50, 100, 200]

    results = []
    scalar_times = {}

    print(f"Threads: {threads}")
    print(f"Implementations: {', '.join(impls.keys())}")
    print()
    print(
        f"{'n':>6} {'rank':>4} {'iter':>4} {'impl':>16} "
        f"{'total_s':>10} {'per_iter':>10} {'speedup':>8}"
    )
    print("-" * 75)

    for n, rank in sizes:
        m, w0 = make_data(n, rank)

        # Scalar baseline
        if n <= 10000:
            elapsed = time_impl(impls["scalar"], m, w0, max_iter)
        else:
            # For n=20k, scalar is very slow; time 1 iteration and extrapolate
            t1 = time_impl(impls["scalar"], m, w0, 1)
            elapsed = t1 * max_iter

        scalar_times[n] = elapsed
        per_iter = elapsed / max_iter
        row = {
            "n": n,
            "rank": rank,
            "max_iter": max_iter,
            "impl": "scalar",
            "threads": threads,
            "total_s": f"{elapsed:.3f}",
            "per_iter_s": f"{per_iter:.3f}",
            "speedup_vs_scalar": "1.00x",
        }
        results.append(row)
        print(
            f"{n:>6} {rank:>4} {max_iter:>4} {'scalar':>16} "
            f"{elapsed:>10.3f} {per_iter:>10.3f} {'1.00x':>8}",
            flush=True,
        )

        # BLAS
        elapsed = time_impl(impls["blas"], m, w0, max_iter)
        per_iter = elapsed / max_iter
        speedup = scalar_times[n] / elapsed
        row = {
            "n": n,
            "rank": rank,
            "max_iter": max_iter,
            "impl": "blas",
            "threads": threads,
            "total_s": f"{elapsed:.3f}",
            "per_iter_s": f"{per_iter:.3f}",
            "speedup_vs_scalar": f"{speedup:.2f}x",
        }
        results.append(row)
        print(
            f"{n:>6} {rank:>4} {max_iter:>4} {'blas':>16} "
            f"{elapsed:>10.3f} {per_iter:>10.3f} {f'{speedup:.2f}x':>8}",
            flush=True,
        )

        # Blocked with different block sizes
        for bs in blocked_sizes:
            if bs > n:
                continue
            label = f"blocked(B={bs})"
            elapsed = time_impl(impls["blocked"], m, w0, max_iter, block_size=bs)
            per_iter = elapsed / max_iter
            speedup = scalar_times[n] / elapsed
            row = {
                "n": n,
                "rank": rank,
                "max_iter": max_iter,
                "impl": label,
                "threads": threads,
                "total_s": f"{elapsed:.3f}",
                "per_iter_s": f"{per_iter:.3f}",
                "speedup_vs_scalar": f"{speedup:.2f}x",
            }
            results.append(row)
            print(
                f"{n:>6} {rank:>4} {max_iter:>4} {label:>16} "
                f"{elapsed:>10.3f} {per_iter:>10.3f} {f'{speedup:.2f}x':>8}",
                flush=True,
            )

        print(flush=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
