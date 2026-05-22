"""Quick test: compare random_state 0 vs 42 on THINGS."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.similarity.datasets import dispatch_dataset_builder
from omegaconf import OmegaConf
from pysrf.bounds import p_upper_only_k_fast, pmin_bound


def main():
    print("Loading THINGS behavioral similarity matrix...")
    cfg = OmegaConf.create({
        "type": "triplet",
        "name": "things",
        "path": "/LOCAL/fmahner/similarity-factorization/data/things",
        "triplet_number": "4.7mio",
        "n_objects": 1854,
    })
    S = dispatch_dataset_builder(cfg)
    print(f"Shape: {S.shape}")

    # Replace NaN with 0 (like both old and new code)
    nan_count = np.isnan(S).sum()
    print(f"NaN count: {nan_count}")
    S = np.nan_to_num(S, nan=0.0)

    # Compute basic stats
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]
    print(f"Top 5 eigenvalues: {eigvals[:5]}")
    print(f"Spectral norm: {eigvals[0]:.4f}")
    print(f"Frobenius norm: {np.linalg.norm(S, 'fro'):.4f}")

    # Compare pmax with different random_states
    for rs in [0, 42]:
        print(f"\n--- random_state={rs} ---")
        pmin, _, _, _, _ = pmin_bound(S, random_state=rs, verbose=False)
        pmax = p_upper_only_k_fast(S, k=1, seed=rs, verbose=True)
        print(f"pmin={pmin:.6f}, pmax={pmax:.6f}")


if __name__ == "__main__":
    main()
