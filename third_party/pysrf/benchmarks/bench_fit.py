"""Benchmark SRF.fit() end-to-end performance."""

from __future__ import annotations

import argparse
import time
import numpy as np

from pysrf import SRF


def generate_test_matrix(
    n: int, rank: int, noise_level: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Generate a low-rank symmetric test matrix with noise."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal((n, rank))
    s = w_true @ w_true.T
    s += noise_level * rng.standard_normal((n, n))
    s = (s + s.T) / 2
    return s


def add_missing_values(s: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    """Add missing values (NaN) to upper triangle of matrix."""
    rng = np.random.default_rng(seed)
    s_missing = s.copy()
    n = s.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_entries = len(triu_i)
    n_missing = int(n_entries * fraction)
    missing_idx = rng.choice(n_entries, size=n_missing, replace=False)
    for idx in missing_idx:
        i, j = triu_i[idx], triu_j[idx]
        s_missing[i, j] = np.nan
        s_missing[j, i] = np.nan
    return s_missing


def benchmark_fit(
    s: np.ndarray,
    rank: int,
    n_repeats: int = 3,
    **fit_kwargs,
) -> dict:
    """Benchmark SRF.fit() and return timing statistics."""
    times = []
    iterations = []

    for i in range(n_repeats):
        model = SRF(rank=rank, random_state=i, **fit_kwargs)
        start = time.perf_counter()
        model.fit(s)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        iterations.append(len(model.history_))

    return {
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "time_min": np.min(times),
        "iter_mean": np.mean(iterations),
        "iter_std": np.std(iterations),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SRF.fit() performance")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Matrix sizes to test",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Ranks to test",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per configuration",
    )
    parser.add_argument(
        "--missing-fraction",
        type=float,
        default=0.3,
        help="Fraction of missing values for missing data test",
    )
    args = parser.parse_args()

    print("SRF.fit() Benchmark")
    print("=" * 80)
    print(f"Sizes: {args.sizes}, Ranks: {args.ranks}, Repeats: {args.repeats}")
    print("=" * 80)

    results = []

    for n in args.sizes:
        for rank in args.ranks:
            if rank >= n:
                continue

            print(f"\nMatrix size: {n}x{n}, rank: {rank}")
            print("-" * 60)

            # Generate test data once per configuration
            s_complete = generate_test_matrix(n, rank, seed=42)
            s_missing = add_missing_values(s_complete, args.missing_fraction, seed=42)

            # Benchmark complete data
            print("  Complete data...", end=" ", flush=True)
            stats_complete = benchmark_fit(s_complete, rank, args.repeats)
            print(
                f"{stats_complete['time_mean']:.3f}s (+/- {stats_complete['time_std']:.3f}s), "
                f"{stats_complete['iter_mean']:.0f} iters"
            )

            # Benchmark missing data
            print("  Missing data (~30%)...", end=" ", flush=True)
            stats_missing = benchmark_fit(s_missing, rank, args.repeats)
            print(
                f"{stats_missing['time_mean']:.3f}s (+/- {stats_missing['time_std']:.3f}s), "
                f"{stats_missing['iter_mean']:.0f} iters"
            )

            overhead = stats_missing["time_mean"] / stats_complete["time_mean"]
            print(f"  Missing data overhead: {overhead:.2f}x")

            results.append(
                {
                    "n": n,
                    "rank": rank,
                    "complete_time": stats_complete["time_mean"],
                    "complete_std": stats_complete["time_std"],
                    "complete_iters": stats_complete["iter_mean"],
                    "missing_time": stats_missing["time_mean"],
                    "missing_std": stats_missing["time_std"],
                    "missing_iters": stats_missing["iter_mean"],
                    "overhead": overhead,
                }
            )

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"{'Size':>6} {'Rank':>6} {'Complete (s)':>14} {'Missing (s)':>14} {'Overhead':>10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['n']:6d} {r['rank']:6d} "
            f"{r['complete_time']:10.3f} +/- {r['complete_std']:.2f} "
            f"{r['missing_time']:10.3f} +/- {r['missing_std']:.2f} "
            f"{r['overhead']:10.2f}x"
        )


if __name__ == "__main__":
    main()
