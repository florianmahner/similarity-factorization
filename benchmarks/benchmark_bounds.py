"""Benchmark comparing original vs optimized bounds implementation."""

from __future__ import annotations

import numpy as np
import time

from pysrf import bounds
from pysrf import bounds_optimized


def generate_test_matrix(
    n: int, rank: int, noise_level: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Generate a low-rank test matrix with noise."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((rank, n))
    S = U @ V
    S = (S + S.T) / 2
    S += noise_level * rng.standard_normal((n, n))
    S = (S + S.T) / 2
    return S


def benchmark_function(func, *args, **kwargs):
    """Time a function call and return result and elapsed time."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def main():
    # sizes = [50, 100, 200, 300, 500]
    sizes = [5000]
    methods = ["dyson"]

    results = []

    print("Benchmarking bounds estimation (original vs optimized)")
    print("=" * 80)

    for n in sizes:
        print(f"\nMatrix size: {n}×{n}")
        print("-" * 80)

        S = generate_test_matrix(n, rank=min(5, n // 10), noise_level=0.1)

        for method in methods:
            print(f"  Method: {method}")

            print("    Original implementation...", end=" ", flush=True)
            try:
                result_orig, time_orig = benchmark_function(
                    bounds.estimate_sampling_bounds_fast,
                    S,
                    method=method,
                    n_jobs=1,
                )
                pmin_orig, pmax_orig, _ = result_orig
                print(f"{time_orig:.2f}s (pmin={pmin_orig:.4f}, pmax={pmax_orig:.4f})")
            except Exception as e:
                print(f"FAILED: {e}")
                time_orig = None
                pmin_orig = pmax_orig = None

            print("    Optimized implementation...", end=" ", flush=True)
            try:
                result_opt, time_opt = benchmark_function(
                    bounds_optimized.estimate_sampling_bounds_fast,
                    S,
                    method=method,
                    n_jobs=1,
                )
                pmin_opt, pmax_opt, _ = result_opt
                print(f"{time_opt:.2f}s (pmin={pmin_opt:.4f}, pmax={pmax_opt:.4f})")
            except Exception as e:
                print(f"FAILED: {e}")
                time_opt = None
                pmin_opt = pmax_opt = None

            if time_orig is not None and time_opt is not None:
                speedup = time_orig / time_opt
                pmin_diff = abs(pmin_orig - pmin_opt) if pmin_orig is not None else None
                pmax_diff = abs(pmax_orig - pmax_opt) if pmax_orig is not None else None

                print(f"    Speedup: {speedup:.2f}x")
                if pmin_diff is not None:
                    print(f"    pmin difference: {pmin_diff:.6f}")
                if pmax_diff is not None:
                    print(f"    pmax difference: {pmax_diff:.6f}")

                results.append(
                    {
                        "n": n,
                        "method": method,
                        "time_orig": time_orig,
                        "time_opt": time_opt,
                        "speedup": speedup,
                        "pmin_orig": pmin_orig,
                        "pmin_opt": pmin_opt,
                        "pmax_orig": pmax_orig,
                        "pmax_opt": pmax_opt,
                        "pmin_diff": pmin_diff,
                        "pmax_diff": pmax_diff,
                    }
                )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if results:
        avg_speedup = np.mean([r["speedup"] for r in results])
        max_speedup = np.max([r["speedup"] for r in results])
        avg_pmin_diff = np.mean(
            [r["pmin_diff"] for r in results if r["pmin_diff"] is not None]
        )
        avg_pmax_diff = np.mean(
            [r["pmax_diff"] for r in results if r["pmax_diff"] is not None]
        )

        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Average pmin difference: {avg_pmin_diff:.6f}")
        print(f"Average pmax difference: {avg_pmax_diff:.6f}")

        print("\nDetailed results:")
        print(
            f"{'Size':>6} {'Method':>10} {'Time (orig)':>12} {'Time (opt)':>12} {'Speedup':>9} {'pmin diff':>11} {'pmax diff':>11}"
        )
        for r in results:
            print(
                f"{r['n']:6d} {r['method']:>10} {r['time_orig']:12.2f}s {r['time_opt']:12.2f}s {r['speedup']:9.2f}x {r['pmin_diff']:11.6f} {r['pmax_diff']:11.6f}"
            )
    else:
        print("No successful benchmark runs")


if __name__ == "__main__":
    main()
