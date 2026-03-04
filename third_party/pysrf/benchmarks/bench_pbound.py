import argparse
import timeit
import numpy as np
from pysrf.bounds import estimate_sampling_bounds_fast


def make_similarity_matrix(
    n: int, rank: int, noise_level: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Generate a synthetic similarity matrix for benchmarking."""
    rng = np.random.default_rng(seed)

    # Generate low-rank matrix
    U = rng.random((n, rank))
    S = U @ U.T

    # Add noise
    noise = rng.normal(0, noise_level, (n, n))
    noise = (noise + noise.T) * 0.5  # Make symmetric

    return S + noise


def benchmark_pbound(n: int, number: int = 20, repeat: int = 10):
    """Benchmark p-bound estimation for a given matrix size."""

    def run_pbound():
        S = make_similarity_matrix(n)
        return estimate_sampling_bounds_fast(S)

    times = timeit.repeat(run_pbound, repeat=repeat, number=number)
    best_time = min(times) / number
    mean_time = sum(times) / (repeat * number)

    return best_time, mean_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark p-bound estimation performance."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 2000],
        help="Matrix sizes to test",
    )
    parser.add_argument(
        "--number", type=int, default=20, help="Number of executions per repeat"
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Number of repeats for timeit"
    )
    args = parser.parse_args()

    print("P-bound Estimation Benchmark")
    print("=" * 50)
    print(f"Testing {len(args.sizes)} matrix sizes")
    print(f"Iterations: {args.repeat} × {args.number}")
    print("")

    results = []
    for n in args.sizes:
        print(f"Testing N={n}...", end=" ")
        try:
            best_time, mean_time = benchmark_pbound(n, args.number, args.repeat)
            results.append((n, best_time, mean_time))
            print(f"Best: {best_time * 1e3:.3f}ms, Mean: {mean_time * 1e3:.3f}ms")
        except Exception as e:
            print(f"Failed: {e}")

    if results:
        print("\nResults Summary:")
        print("N\tBest(ms)\tMean(ms)")
        print("-" * 30)
        for n, best_time, mean_time in results:
            print(f"{n}\t{best_time * 1e3:.3f}\t\t{mean_time * 1e3:.3f}")


if __name__ == "__main__":
    main()
