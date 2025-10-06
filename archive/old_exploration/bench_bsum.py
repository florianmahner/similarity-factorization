import numpy as np
import timeit
import os

# Set up pyximport for Cython compilation
os.environ["CPPFLAGS"] = (
    f"-I{np.get_include()} -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
)

import pyximport

pyximport.install(
    setup_args={
        "include_dirs": [np.get_include()],
        "script_args": ["--quiet"],
    },
    language_level=3,
)

from models.bsum import update_w as update_w_cython
from models.admm import update_w_python


def make_data(n=100, r=10, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))
    return m, x0


def run_cython():
    m, x0 = make_data()
    return update_w_cython(m, x0, tol=0.0)


def run_python():
    m, x0 = make_data()
    return update_w_python(m, x0, tol=0.0)


def verify_implementations():
    try:
        m, x0 = make_data(seed=1234)

        cython_result = update_w_cython(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)

        error = np.max(np.abs(cython_result - python_result))
        print(f"Maximum difference between implementations: {error:.4e}")

        if error > 1e-10:
            print("Warning: Large discrepancy between implementations")
            return False
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def benchmark_function(func, number=20, repeat=10):
    try:
        times = timeit.repeat(func, repeat=repeat, number=number)
        return min(times) / number, sum(times) / (repeat * number)
    except Exception as e:
        print(f"Benchmark failed for {func.__name__}: {e}")
        return None, None


def main():
    print("Benchmarking update_w implementations")
    print("=" * 40)

    if not verify_implementations():
        return

    results = {}
    implementations = [("Python", run_python), ("Cython", run_cython)]

    for label, func in implementations:
        print(f"Benchmarking {label}...")
        best, mean = benchmark_function(func)

        if best is not None:
            results[label] = best
            print(f"{label:<8}  best: {best*1e3:8.3f} ms   mean: {mean*1e3:8.3f} ms")
        else:
            print(f"{label:<8}  failed")

    if "Python" in results and "Cython" in results:
        speedup = results["Python"] / results["Cython"]
        print(f"\nSpeedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
