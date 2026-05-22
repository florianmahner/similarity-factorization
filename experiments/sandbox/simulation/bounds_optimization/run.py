"""Benchmark comparing all bounds estimation implementations.

This script compares:
1. estimate_sampling_bounds (original)
2. estimate_sampling_bounds_fast (current "fast" version)
3. estimate_sampling_bounds_ultra (new optimized version)

It verifies numerical equivalence and measures speedups.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pysrf.bounds import (
    estimate_sampling_bounds,
    estimate_sampling_bounds_fast,
    pmin_bound,
    p_upper_only_k,
    p_upper_only_k_fast,
    lambda_bulk_dyson_raw,
    lambda_bulk_dyson_raw_fast,
)

from bounds_ultra import (
    estimate_sampling_bounds_ultra,
    precompute_matrix_info,
    pmin_bound_ultra,
    p_upper_only_k_ultra,
    lambda_bulk_dyson_ultra,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_test_matrix(n: int, rank: int, random_state: int = 42) -> np.ndarray:
    """Generate a low-rank symmetric positive semidefinite matrix."""
    rng = np.random.RandomState(random_state)
    W = rng.rand(n, rank)
    S = W @ W.T
    return (S + S.T) / 2


def benchmark_function(func, name: str, *args, **kwargs):
    """Benchmark a function and return (result, elapsed_time)."""
    logger.info(f"  Running {name}...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.info(f"  {name}: {elapsed:.3f}s")
    return result, elapsed


def compare_results(res1, res2, name1: str, name2: str, rtol: float = 1e-4):
    """Compare two results and report differences."""
    if isinstance(res1, tuple) and isinstance(res2, tuple):
        for i, (r1, r2) in enumerate(zip(res1, res2)):
            if isinstance(r1, np.ndarray):
                if not np.allclose(r1, r2, rtol=rtol, equal_nan=True):
                    max_diff = np.max(np.abs(r1 - r2))
                    logger.warning(f"  Array diff at position {i}: max_diff={max_diff:.6e}")
                    return False
            elif isinstance(r1, (int, float, np.floating)):
                if not np.isclose(r1, r2, rtol=rtol):
                    logger.warning(f"  Scalar diff at position {i}: {r1} vs {r2}")
                    return False
    elif isinstance(res1, (int, float, np.floating)):
        if not np.isclose(res1, res2, rtol=rtol):
            logger.warning(f"  {name1}={res1:.6f} vs {name2}={res2:.6f}")
            return False
    return True


def run_component_benchmarks(S: np.ndarray, random_state: int = 42):
    """Benchmark individual components to identify bottlenecks."""
    logger.info("\n=== Component Benchmarks ===")
    n = S.shape[0]
    results = {}

    logger.info(f"\nMatrix size: {n}x{n}")

    logger.info("\n1. pmin_bound comparison:")
    pmin_orig, t_orig = benchmark_function(
        pmin_bound, "pmin_bound (original)",
        S, random_state=random_state
    )

    info = precompute_matrix_info(S)
    pmin_ultra_result, t_ultra = benchmark_function(
        pmin_bound_ultra, "pmin_bound_ultra",
        S, info
    )
    pmin_ultra = pmin_ultra_result[0]

    logger.info(f"  pmin original: {pmin_orig[0]:.6f}")
    logger.info(f"  pmin ultra:    {pmin_ultra:.6f}")
    logger.info(f"  Speedup: {t_orig/t_ultra:.1f}x")
    results["pmin"] = {"original": t_orig, "ultra": t_ultra}

    logger.info("\n2. lambda_bulk_dyson comparison (p=0.5):")
    p_test = 0.5

    edge_orig, t_orig = benchmark_function(
        lambda_bulk_dyson_raw, "lambda_bulk_dyson_raw",
        S, p_test
    )

    from numpy.linalg import eigvalsh
    s2_max = np.max(eigvalsh(S**2))
    s_norm = np.linalg.norm(S, 2)

    edge_fast, t_fast = benchmark_function(
        lambda_bulk_dyson_raw_fast, "lambda_bulk_dyson_raw_fast",
        S, p_test, s2_max=s2_max, s_norm=s_norm
    )

    edge_ultra, t_ultra = benchmark_function(
        lambda_bulk_dyson_ultra, "lambda_bulk_dyson_ultra",
        S, p_test, info
    )

    logger.info(f"  edge original: {edge_orig:.6f}")
    logger.info(f"  edge fast:     {edge_fast:.6f}")
    logger.info(f"  edge ultra:    {edge_ultra:.6f}")
    logger.info(f"  Speedup (fast vs original):  {t_orig/t_fast:.1f}x")
    logger.info(f"  Speedup (ultra vs original): {t_orig/t_ultra:.1f}x")
    results["lambda_bulk"] = {"original": t_orig, "fast": t_fast, "ultra": t_ultra}

    logger.info("\n3. p_upper_only_k comparison:")
    eff_dim = info.eff_dim
    logger.info(f"  Using k={eff_dim} (effective dimension)")

    pmax_orig, t_orig = benchmark_function(
        p_upper_only_k, "p_upper_only_k (original)",
        S, k=eff_dim, seed=random_state
    )

    pmax_fast, t_fast = benchmark_function(
        p_upper_only_k_fast, "p_upper_only_k_fast",
        S, k=eff_dim, seed=random_state, n_jobs=1
    )

    pmax_ultra, t_ultra = benchmark_function(
        p_upper_only_k_ultra, "p_upper_only_k_ultra",
        S, k=eff_dim, info=info
    )

    logger.info(f"  pmax original: {pmax_orig:.6f}")
    logger.info(f"  pmax fast:     {pmax_fast:.6f}")
    logger.info(f"  pmax ultra:    {pmax_ultra:.6f}")
    logger.info(f"  Speedup (fast vs original):  {t_orig/t_fast:.1f}x")
    logger.info(f"  Speedup (ultra vs original): {t_orig/t_ultra:.1f}x")
    results["p_upper_only_k"] = {"original": t_orig, "fast": t_fast, "ultra": t_ultra}

    return results


def run_full_benchmarks(S: np.ndarray, random_state: int = 42):
    """Benchmark the full estimate_sampling_bounds functions."""
    logger.info("\n=== Full estimate_sampling_bounds Benchmarks ===")
    n = S.shape[0]
    results = {}

    logger.info(f"\nMatrix size: {n}x{n}")

    (pmin_orig, pmax_orig, _), t_orig = benchmark_function(
        estimate_sampling_bounds, "estimate_sampling_bounds (original)",
        S, random_state=random_state, verbose=False
    )

    (pmin_fast, pmax_fast, _), t_fast = benchmark_function(
        estimate_sampling_bounds_fast, "estimate_sampling_bounds_fast",
        S, random_state=random_state, verbose=False, n_jobs=1
    )

    (pmin_ultra, pmax_ultra, _), t_ultra = benchmark_function(
        estimate_sampling_bounds_ultra, "estimate_sampling_bounds_ultra",
        S, random_state=random_state, verbose=False
    )

    logger.info(f"\nResults:")
    logger.info(f"  Original: pmin={pmin_orig:.6f}, pmax={pmax_orig:.6f}, time={t_orig:.3f}s")
    logger.info(f"  Fast:     pmin={pmin_fast:.6f}, pmax={pmax_fast:.6f}, time={t_fast:.3f}s")
    logger.info(f"  Ultra:    pmin={pmin_ultra:.6f}, pmax={pmax_ultra:.6f}, time={t_ultra:.3f}s")

    logger.info(f"\nSpeedups:")
    logger.info(f"  Fast vs Original:  {t_orig/t_fast:.1f}x")
    logger.info(f"  Ultra vs Original: {t_orig/t_ultra:.1f}x")
    logger.info(f"  Ultra vs Fast:     {t_fast/t_ultra:.1f}x")

    results["times"] = {"original": t_orig, "fast": t_fast, "ultra": t_ultra}
    results["pmin"] = {"original": pmin_orig, "fast": pmin_fast, "ultra": pmin_ultra}
    results["pmax"] = {"original": pmax_orig, "fast": pmax_fast, "ultra": pmax_ultra}

    pmin_match = np.isclose(pmin_orig, pmin_ultra, rtol=0.01)
    pmax_match = np.isclose(pmax_orig, pmax_ultra, rtol=0.01)

    logger.info(f"\nNumerical agreement (rtol=0.01):")
    logger.info(f"  pmin: {'PASS' if pmin_match else 'FAIL'}")
    logger.info(f"  pmax: {'PASS' if pmax_match else 'FAIL'}")

    results["agreement"] = {"pmin": pmin_match, "pmax": pmax_match}

    return results


def run_scaling_benchmark(sizes: list[int], rank: int = 5, random_state: int = 42):
    """Benchmark scaling across different matrix sizes."""
    logger.info("\n=== Scaling Benchmark ===")

    records = []
    for n in sizes:
        logger.info(f"\n--- Matrix size: {n}x{n} ---")
        S = generate_test_matrix(n, rank, random_state)

        info = precompute_matrix_info(S)
        eff_dim = info.eff_dim

        _, t_orig = benchmark_function(
            p_upper_only_k, "p_upper_only_k (original)",
            S, k=eff_dim, seed=random_state
        )

        _, t_ultra = benchmark_function(
            p_upper_only_k_ultra, "p_upper_only_k_ultra",
            S, k=eff_dim, info=info
        )

        speedup = t_orig / t_ultra
        logger.info(f"  Speedup: {speedup:.1f}x")

        records.append({
            "n": n,
            "rank": rank,
            "eff_dim": eff_dim,
            "time_original": t_orig,
            "time_ultra": t_ultra,
            "speedup": speedup,
        })

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Benchmark bounds estimation")
    parser.add_argument("--size", type=int, default=100, help="Matrix size for single benchmark")
    parser.add_argument("--rank", type=int, default=5, help="True rank of test matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--components", action="store_true", help="Benchmark individual components")
    parser.add_argument("--full", action="store_true", help="Run full bounds estimation")
    args = parser.parse_args()

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    if args.scaling:
        sizes = [30, 50, 100, 200, 300, 500]
        df = run_scaling_benchmark(sizes, args.rank, args.seed)
        df.to_csv(output_dir / "scaling_benchmark.csv", index=False)
        logger.info(f"\nScaling results saved to {output_dir / 'scaling_benchmark.csv'}")
        logger.info(f"\n{df.to_string()}")
    else:
        S = generate_test_matrix(args.size, args.rank, args.seed)
        logger.info(f"Generated test matrix: {S.shape}, rank={args.rank}")

        all_results = {}

        if args.components:
            comp_results = run_component_benchmarks(S, args.seed)
            all_results["components"] = comp_results

        if args.full or not args.components:
            full_results = run_full_benchmarks(S, args.seed)
            all_results["full"] = full_results

        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        logger.info(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
