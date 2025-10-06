import argparse
import timeit
import numpy as np
import os

# Set up pyximport for Cython compilation
os.environ[
    "CPPFLAGS"
] = f"-I{np.get_include()} -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
import pyximport

pyximport.install(
    setup_args={
        "include_dirs": [np.get_include()],
        "script_args": ["--quiet"],
    },
    language_level=3,
)

from pysrf.model import update_w_python
from pysrf._bsum import update_w as update_w_cython


def make_data(
    n: int = 100, r: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))
    return m, x0


def benchmark_size(n: int, r: int, number: int = 20, repeat: int = 10):
    def run_cython():
        m, x0 = make_data(n, r)
        return update_w_cython(m, x0, tol=0.0)

    def run_python():
        m, x0 = make_data(n, r)
        return update_w_python(m, x0, tol=0.0)

    cython_times = timeit.repeat(run_cython, repeat=repeat, number=number)
    python_times = timeit.repeat(run_python, repeat=repeat, number=number)

    cython_best = min(cython_times) / number
    python_best = min(python_times) / number
    speedup = python_best / cython_best

    return cython_best, python_best, speedup


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cython performance across matrix sizes."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500],
        help="Matrix sizes to test",
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=[5, 10, 20], help="Ranks to test"
    )
    parser.add_argument(
        "--number", type=int, default=20, help="Number of executions per repeat"
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Number of repeats for timeit"
    )
    args = parser.parse_args()

    print("Cython Performance Benchmark")
    print("=" * 60)
    print(
        f"Testing {len(args.sizes)} sizes × {len(args.ranks)} ranks = {len(args.sizes) * len(args.ranks)} combinations"
    )
    print(f"Iterations: {args.repeat} × {args.number}")
    print("")

    results = []
    for n in args.sizes:
        for r in args.ranks:
            if r >= n:
                continue

            print(f"Testing N={n}, R={r}...", end=" ")
            try:
                cython_time, python_time, speedup = benchmark_size(
                    n, r, args.number, args.repeat
                )
                results.append((n, r, cython_time, python_time, speedup))
                print(f"Speedup: {speedup:.1f}x")
            except Exception as e:
                print(f"Failed: {e}")

    if results:
        print("\nResults Summary:")
        print("N\tR\tCython(ms)\tPython(ms)\tSpeedup")
        print("-" * 50)
        for n, r, cython_time, python_time, speedup in results:
            print(
                f"{n}\t{r}\t{cython_time*1e3:.3f}\t\t{python_time*1e3:.3f}\t\t{speedup:.1f}x"
            )


if __name__ == "__main__":
    main()
