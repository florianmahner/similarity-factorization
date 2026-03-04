"""Correctness test: all BSUM implementations vs scalar (ground truth).

Compares blas and blocked (multiple block sizes) against the scalar
Cython implementation across problem sizes and iteration counts.

Checks:
  1. W matrix element-wise agreement (max |W_test - W_scalar|)
  2. Reconstruction quality agreement (||M - W@W.T||_F / ||M||_F)
  3. Whether differences grow, stay bounded, or diverge with iterations

Usage:
    python -u correctness.py
"""

import csv
import time
from pathlib import Path

import numpy as np

RESULTS_CSV = Path(__file__).parent / "correctness.csv"

CSV_COLUMNS = [
    "n",
    "rank",
    "max_iter",
    "impl",
    "max_w_diff",
    "mean_w_diff",
    "recon_scalar",
    "recon_test",
    "recon_diff",
    "bit_identical",
]


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


def compare(m, w0, impl_fn, scalar_fn, max_iter: int, **kwargs):
    w_scalar = scalar_fn(m, w0.copy(), max_iter=max_iter, tol=0.0)
    w_test = impl_fn(m, w0.copy(), max_iter=max_iter, tol=0.0, **kwargs)

    norm_m = np.linalg.norm(m, "fro")
    max_w_diff = np.max(np.abs(w_test - w_scalar))
    mean_w_diff = np.mean(np.abs(w_test - w_scalar))
    recon_scalar = np.linalg.norm(m - w_scalar @ w_scalar.T, "fro") / norm_m
    recon_test = np.linalg.norm(m - w_test @ w_test.T, "fro") / norm_m
    identical = np.array_equal(w_test, w_scalar)

    return {
        "max_w_diff": max_w_diff,
        "mean_w_diff": mean_w_diff,
        "recon_scalar": recon_scalar,
        "recon_test": recon_test,
        "recon_diff": recon_test - recon_scalar,
        "bit_identical": identical,
    }


def main():
    impls = load_implementations()
    scalar_fn = impls["scalar"]

    sizes = [(500, 50), (1854, 50), (1854, 100), (5000, 50)]
    iter_counts = [1, 5, 10, 50, 100, 500]
    blocked_sizes = [1, 50, 100, 200]

    results = []

    print(
        f"{'n':>5} {'rank':>4} {'iter':>5} {'impl':>16} "
        f"{'max_W_diff':>12} {'mean_W_diff':>12} "
        f"{'recon_scalar':>14} {'recon_test':>14} {'recon_diff':>12} "
        f"{'identical':>9}"
    )
    print("-" * 125)

    for n, rank in sizes:
        m, w0 = make_data(n, rank)

        for max_iter in iter_counts:
            if n >= 5000 and max_iter > 100:
                continue

            t0 = time.perf_counter()

            # BLAS
            result = compare(m, w0, impls["blas"], scalar_fn, max_iter)
            row = {
                "n": n,
                "rank": rank,
                "max_iter": max_iter,
                "impl": "blas",
                "max_w_diff": f"{result['max_w_diff']:.2e}",
                "mean_w_diff": f"{result['mean_w_diff']:.2e}",
                "recon_scalar": f"{result['recon_scalar']:.12f}",
                "recon_test": f"{result['recon_test']:.12f}",
                "recon_diff": f"{result['recon_diff']:+.2e}",
                "bit_identical": result["bit_identical"],
            }
            results.append(row)
            print(
                f"{n:>5} {rank:>4} {max_iter:>5} {'blas':>16} "
                f"{row['max_w_diff']:>12} {row['mean_w_diff']:>12} "
                f"{row['recon_scalar']:>14} {row['recon_test']:>14} "
                f"{row['recon_diff']:>12} {str(result['bit_identical']):>9}",
                flush=True,
            )

            # Blocked with different block sizes
            for bs in blocked_sizes:
                if bs > n:
                    continue
                label = f"blocked(B={bs})"
                result = compare(
                    m, w0, impls["blocked"], scalar_fn, max_iter, block_size=bs
                )
                row = {
                    "n": n,
                    "rank": rank,
                    "max_iter": max_iter,
                    "impl": label,
                    "max_w_diff": f"{result['max_w_diff']:.2e}",
                    "mean_w_diff": f"{result['mean_w_diff']:.2e}",
                    "recon_scalar": f"{result['recon_scalar']:.12f}",
                    "recon_test": f"{result['recon_test']:.12f}",
                    "recon_diff": f"{result['recon_diff']:+.2e}",
                    "bit_identical": result["bit_identical"],
                }
                results.append(row)
                print(
                    f"{n:>5} {rank:>4} {max_iter:>5} {label:>16} "
                    f"{row['max_w_diff']:>12} {row['mean_w_diff']:>12} "
                    f"{row['recon_scalar']:>14} {row['recon_test']:>14} "
                    f"{row['recon_diff']:>12} {str(result['bit_identical']):>9}",
                    flush=True,
                )

            elapsed = time.perf_counter() - t0
            if elapsed > 1:
                print(f"  ({elapsed:.1f}s)", flush=True)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to: {RESULTS_CSV}")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for impl_name in sorted({r["impl"] for r in results}):
        impl_rows = [r for r in results if r["impl"] == impl_name]
        all_identical = all(r["bit_identical"] for r in impl_rows)
        max_diff = max(float(r["max_w_diff"]) for r in impl_rows)
        max_recon_diff = max(abs(float(r["recon_diff"])) for r in impl_rows)

        print(f"\n{impl_name} vs scalar:")
        print(f"  Bit-identical:        {'YES' if all_identical else 'NO'}")
        print(f"  Max W element diff:   {max_diff:.2e}")
        print(f"  Max recon error diff: {max_recon_diff:.2e}")

        if all_identical:
            print("  VERDICT: EXACT MATCH")
        elif max_recon_diff < 1e-10:
            print("  VERDICT: Numerically equivalent (BLAS accumulation order only)")
        else:
            print("  VERDICT: DIFFERENCES DETECTED - investigate!")


if __name__ == "__main__":
    main()
