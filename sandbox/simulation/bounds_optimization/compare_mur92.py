"""Compare original vs ultra bounds estimation on mur92."""

import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import load_dataset
from pysrf.bounds import (
    estimate_sampling_bounds,
    estimate_sampling_bounds_fast,
    estimate_sampling_bounds_ultra,
)

MUR92_PATH = "/SSD/datasets/similarity_datasets/mur92"


def main():
    print("Loading mur92 dataset...")
    ds = load_dataset("mur92", root=MUR92_PATH)
    S = ds.rsm
    print(f"Shape: {S.shape}")

    # Check for NaN
    nan_count = np.isnan(S).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values, setting to 0")
        S = np.nan_to_num(S, nan=0.0)

    print("\n" + "="*60)
    print("Running ORIGINAL estimate_sampling_bounds...")
    print("="*60)
    t0 = time.time()
    pmin_orig, pmax_orig, _ = estimate_sampling_bounds(
        S, verbose=True, random_state=0
    )
    t_orig = time.time() - t0
    print(f"\nOriginal: pmin={pmin_orig:.6f}, pmax={pmax_orig:.6f}")
    print(f"Time: {t_orig:.2f}s")

    print("\n" + "="*60)
    print("Running FAST estimate_sampling_bounds_fast...")
    print("="*60)
    t0 = time.time()
    pmin_fast, pmax_fast, _ = estimate_sampling_bounds_fast(
        S, verbose=True, random_state=0
    )
    t_fast = time.time() - t0
    print(f"\nFast: pmin={pmin_fast:.6f}, pmax={pmax_fast:.6f}")
    print(f"Time: {t_fast:.2f}s")

    print("\n" + "="*60)
    print("Running ULTRA estimate_sampling_bounds_ultra...")
    print("="*60)
    t0 = time.time()
    pmin_ultra, pmax_ultra, _ = estimate_sampling_bounds_ultra(
        S, verbose=True, random_state=0
    )
    t_ultra = time.time() - t0
    print(f"\nUltra: pmin={pmin_ultra:.6f}, pmax={pmax_ultra:.6f}")
    print(f"Time: {t_ultra:.2f}s")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Method':<10} {'pmin':>10} {'pmax':>10}")
    print("-" * 32)
    print(f"{'Original':<10} {pmin_orig:>10.6f} {pmax_orig:>10.6f}")
    print(f"{'Fast':<10} {pmin_fast:>10.6f} {pmax_fast:>10.6f}")
    print(f"{'Ultra':<10} {pmin_ultra:>10.6f} {pmax_ultra:>10.6f}")

    # Check if they all match
    all_pmax = [pmax_orig, pmax_fast, pmax_ultra]
    all_pmin = [pmin_orig, pmin_fast, pmin_ultra]

    pmax_range = max(all_pmax) - min(all_pmax)
    pmin_range = max(all_pmin) - min(all_pmin)

    print(f"\npmax range: {pmax_range:.6f}")
    print(f"pmin range: {pmin_range:.6f}")

    if pmax_range < 0.01 and pmin_range < 0.01:
        print("\n✓ All methods MATCH within 1% tolerance")
    else:
        print("\n✗ Methods DIFFER!")


if __name__ == "__main__":
    main()
