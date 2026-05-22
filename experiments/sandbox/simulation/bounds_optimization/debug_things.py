"""Debug THINGS bounds discrepancy."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.similarity.datasets import dispatch_dataset_builder as build_similarity_matrix
from omegaconf import OmegaConf
from pysrf.bounds import estimate_sampling_bounds_fast


def main():
    print("Loading THINGS behavioral similarity matrix...")
    cfg = OmegaConf.create({
        "type": "triplet",
        "name": "things",
        "path": "/LOCAL/fmahner/similarity-factorization/data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    S = build_similarity_matrix(cfg)
    print(f"Shape: {S.shape}")

    # Check for NaN
    nan_count = np.isnan(S).sum()
    print(f"NaN count: {nan_count}")
    if nan_count > 0:
        nan_indices = np.argwhere(np.isnan(S))
        print(f"NaN locations (first 10): {nan_indices[:10]}")
        # Check if diagonal
        diag_nan = np.isnan(np.diag(S)).sum()
        print(f"NaN on diagonal: {diag_nan}")
        print(f"NaN off-diagonal: {nan_count - diag_nan}")

    # Test 1: WITH NaN (like old code did)
    print("\n" + "="*60)
    print("Test 1: WITH NaN values (like embedding_generation)")
    print("="*60)
    try:
        pmin1, pmax1, _ = estimate_sampling_bounds_fast(
            S, random_state=0, verbose=True
        )
        print(f"Result: pmin={pmin1:.6f}, pmax={pmax1:.6f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: NaN -> 0 (like our bounds experiment)
    print("\n" + "="*60)
    print("Test 2: NaN -> 0 (like bounds experiment)")
    print("="*60)
    S_clean = np.nan_to_num(S, nan=0.0)
    pmin2, pmax2, _ = estimate_sampling_bounds_fast(
        S_clean, random_state=42, verbose=True
    )
    print(f"Result: pmin={pmin2:.6f}, pmax={pmax2:.6f}")


if __name__ == "__main__":
    main()
